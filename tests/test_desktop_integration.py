"""Opt-in real Docker tests. Only uniquely named synthetic fixtures are changed.

RLM_DOCKER_INTEGRATION=1 python3 -m unittest discover -s tests -p test_desktop_integration.py -v
Requires a locally available standard Agent Zero image and Docker Desktop/Engine.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.getenv("RLM_TEST_A0_IMAGE", "agent0ai/agent-zero:latest")
PYTHON = "/opt/venv-a0/bin/python3"


def docker(*args, check=True, input=None):
    return subprocess.run(["docker", *args], text=True, input=input, capture_output=True, check=check, timeout=300)


@unittest.skipUnless(os.getenv("RLM_DOCKER_INTEGRATION") == "1", "opt-in Docker integration test")
class DesktopIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.name = "rlm-test-" + uuid.uuid4().hex[:10]
        self.network = self.name + "-network"
        self.volume = self.name + "-data"
        self.ids = []
        docker("network", "create", self.network)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        # Only fixtures owned by this test, identified by unique labels/names.
        ids = docker("ps", "-aq", "--filter", f"label=rlm.integration={self.name}").stdout.split()
        for target in ids:
            docker("rm", "-f", target, check=False)
        docker("network", "rm", self.network, check=False)
        docker("volume", "rm", self.volume, check=False)
        for target in self.ids:
            docker("volume", "rm", f"rlm-{target[:12]}-usr", check=False)
            docker("image", "rm", f"rlm-local/agent-zero:{target[:12]}", check=False)
        for image in set(docker("image", "ls", "-aq", "--filter", f"label=rlm.integration={self.name}").stdout.split()):
            docker("image", "rm", image, check=False)

    def inspect(self):
        return json.loads(docker("inspect", self.name).stdout)[0]

    def fixture(self, storage="volume"):
        args = ["run", "-d", "--init", "--name", self.name, "--label", f"rlm.integration={self.name}",
                "--network", self.network, "--network-alias", "agent-fixture", "--restart", "unless-stopped",
                "-e", "RLM_TEST_VALUE=synthetic-setting", "-p", "127.0.0.1::80"]
        if storage == "volume":
            args += ["--mount", f"type=volume,source={self.volume},target=/a0/usr"]
        args += ["--entrypoint", PYTHON, IMAGE, "-m", "http.server", "80", "--bind", "0.0.0.0"]
        self.ids.append(docker(*args).stdout.strip())
        docker("exec", self.name, PYTHON, "-c", "from pathlib import Path; Path('/a0/usr/plugins').mkdir(parents=True,exist_ok=True); Path('/a0/usr/fixture.txt').write_text('persistent-data'); Path('/opt/rlm-fixture.txt').write_text('installed-software')")
        docker("cp", str(ROOT), f"{self.name}:/a0/usr/plugins/rlm")
        return self.inspect()

    def launch(self, *, apply=True, bad_probe=False):
        if bad_probe:
            return docker("run", "--rm", "-i", "--user", "0", "--network", "none",
                "--mount", "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock",
                "--entrypoint", PYTHON, IMAGE, "-", "--container", self.name, "--apply", "--yes",
                "--probe-image", "docker:28-cli", input=(ROOT / "setup/docker_desktop_setup.py").read_text(), check=False)
        return subprocess.run(["sh", str(ROOT / "setup/enable-docker-access.sh"), "--container", self.name,
            "--apply" if apply else "--check", "--yes"], capture_output=True, text=True)

    def assert_preserved(self, before, *, same_port=True):
        after = self.inspect()
        if same_port:
            self.assertEqual(after["NetworkSettings"]["Ports"], before["NetworkSettings"]["Ports"])
        self.assertEqual(after["HostConfig"]["RestartPolicy"], before["HostConfig"]["RestartPolicy"])
        self.assertIn("RLM_TEST_VALUE=synthetic-setting", after["Config"]["Env"])
        self.assertIn("agent-fixture", after["NetworkSettings"]["Networks"][self.network]["Aliases"])
        docker("exec", self.name, PYTHON, "-c", "from pathlib import Path; assert Path('/a0/usr/fixture.txt').read_text()=='persistent-data'; assert Path('/opt/rlm-fixture.txt').read_text()=='installed-software'")
        return after

    def test_named_volume_install_probe_and_repeat(self):
        before = self.fixture()
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        after = self.assert_preserved(before)
        self.assertNotEqual(after["Id"], before["Id"])
        docker("exec", self.name, PYTHON, "-c", "import json; from pathlib import Path; assert json.loads(Path('/a0/usr/plugins/rlm/data/docker-probe.json').read_text())['success']")
        repeat = self.launch()
        self.assertEqual(repeat.returncode, 0, repeat.stdout + repeat.stderr)
        self.assertEqual(self.inspect()["Id"], after["Id"])
        docker("stop", "-t", "1", self.name)
        restart = self.launch()
        self.assertEqual(restart.returncode, 0, restart.stdout + restart.stderr)
        self.assertEqual(self.inspect()["Id"], after["Id"])

    def test_unmounted_data_migrates_to_a_volume(self):
        before = self.fixture(storage="none")
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        after = self.assert_preserved(before)
        self.assertTrue(any(m["Destination"] == "/a0/usr" and m["Type"] == "volume" for m in after["Mounts"]))

    def test_probe_failure_restores_original_container(self):
        before = self.fixture()
        result = self.launch(bad_probe=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Restoring the original", result.stdout)
        after = self.assert_preserved(before, same_port=False)
        self.assertEqual(after["Id"], before["Id"])
        self.assertIn(after["NetworkSettings"]["Ports"]["80/tcp"][0]["HostPort"], result.stdout)
        self.assertTrue(after["State"]["Running"])

    def test_check_does_not_change_running_container(self):
        before = self.fixture()
        result = self.launch(apply=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.inspect()["Id"], before["Id"])
        self.assert_preserved(before)


if __name__ == "__main__":
    unittest.main()
