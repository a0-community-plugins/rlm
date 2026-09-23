from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("desktop_setup", Path(__file__).resolve().parents[1] / "setup/docker_desktop_setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def original():
    return {
        "Id": "abcdef123456" * 5 + "abcd", "Name": "/my-agent", "Image": "sha256:original",
        "State": {"Running": True},
        "Config": {"Image": "agent0ai/agent-zero:latest", "Env": ["PRIVATE=do-not-print", "PATH=/usr/local/bin:/usr/bin", "DOCKER_CONTEXT=wrong"], "Labels": {"owner": "kept"}, "Hostname": "abcdef123456", "Cmd": ["/exe/initialize.sh", "$BRANCH"]},
        "HostConfig": {"AutoRemove": False, "NetworkMode": "bridge", "RestartPolicy": {"Name": "unless-stopped"}, "Binds": ["/home/user/Agent Zero/usr:/a0/usr"], "PortBindings": {"80/tcp": [{"HostIp": "127.0.0.1", "HostPort": "51234"}]}, "Memory": 536870912, "CapAdd": ["SYS_PTRACE"]},
        "Mounts": [{"Type": "bind", "Source": "/home/user/Agent Zero/usr", "Destination": "/a0/usr", "RW": True}],
        "NetworkSettings": {"Networks": {"bridge": {"IPAddress": "172.17.0.2", "IPAMConfig": None, "Aliases": None}}, "Ports": {"80/tcp": [{"HostIp": "127.0.0.1", "HostPort": "51234"}]}},
    }


class ConfigTests(unittest.TestCase):
    def test_preserves_existing_data_settings_and_does_not_mutate_inspect(self):
        before = original()
        saved = copy.deepcopy(before)
        result = setup.replacement_config(before, "sha256:derived")
        self.assertEqual(before, saved)
        self.assertIn("PRIVATE=do-not-print", result["Env"])
        self.assertNotIn("DOCKER_CONTEXT=wrong", result["Env"])
        self.assertEqual(result["Cmd"], before["Config"]["Cmd"])
        self.assertEqual(result["HostConfig"]["Binds"], before["HostConfig"]["Binds"])
        for field in ("PortBindings", "Memory", "CapAdd", "RestartPolicy"):
            self.assertEqual(result["HostConfig"][field], before["HostConfig"][field])
        self.assertEqual(result["Hostname"], "")
        self.assertEqual(result["Labels"]["owner"], "kept")
        self.assertIn("RLM_AGENT_ZERO_CONTAINER=my-agent", result["Env"])

    def test_reuses_named_and_anonymous_volumes(self):
        data = original()
        data["HostConfig"]["Binds"] = []
        data["Mounts"] = [{"Type": "volume", "Name": "anonymous-existing", "Source": "/var/lib/docker/volumes/anonymous-existing/_data", "Destination": "/a0/usr", "RW": True}]
        result = setup.replacement_config(data, "derived")
        self.assertIn({"Type": "volume", "Source": "anonymous-existing", "Target": "/a0/usr", "ReadOnly": False}, result["HostConfig"]["Mounts"])

    def test_preserves_mount_options_and_does_not_duplicate_explicit_volume(self):
        data = original()
        mount = {"Type": "volume", "Source": "a0", "Target": "/a0/usr", "VolumeOptions": {"NoCopy": True}}
        data["HostConfig"]["Binds"] = []
        data["HostConfig"]["Mounts"] = [mount]
        data["Mounts"] = [{"Type": "volume", "Name": "a0", "Destination": "/a0/usr", "RW": True}]
        result = setup.replacement_config(data, "derived")
        self.assertEqual([m for m in result["HostConfig"]["Mounts"] if m["Target"] == "/a0/usr"], [mount])

    def test_unmounted_user_data_gets_persistent_volume_from_snapshot(self):
        data = original()
        data["HostConfig"]["Binds"] = []
        data["Mounts"] = []
        result = setup.replacement_config(data, "derived")
        self.assertIn({"Type": "volume", "Source": "rlm-abcdef123456-usr", "Target": "/a0/usr"}, result["HostConfig"]["Mounts"])

    def test_dynamic_web_port_stays_the_same(self):
        data = original()
        data["HostConfig"]["PortBindings"]["80/tcp"][0]["HostPort"] = "0"
        result = setup.replacement_config(data, "derived")
        self.assertEqual(result["HostConfig"]["PortBindings"], data["NetworkSettings"]["Ports"])

    def test_static_network_alias_and_ip_are_preserved_without_runtime_fields(self):
        data = original()
        data["NetworkSettings"]["Networks"] = {"custom": {"IPAMConfig": {"IPv4Address": "172.30.0.5"}, "IPAddress": "172.30.0.5", "Aliases": ["my-agent", data["Id"][:12]], "EndpointID": "do-not-reuse"}}
        result = setup.replacement_config(data, "derived")
        self.assertEqual(result["NetworkingConfig"]["EndpointsConfig"]["custom"], {"IPAMConfig": {"IPv4Address": "172.30.0.5"}, "Aliases": ["my-agent"]})

    def test_rejects_destructive_or_unreproducible_layout_before_changes(self):
        for field, value in (("AutoRemove", True), ("NetworkMode", "host"), ("NetworkMode", "container:other"), ("VolumesFrom", ["other"]), ("ReadonlyRootfs", True), ("IpcMode", "container:other")):
            with self.subTest(field=field, value=value):
                data = original()
                data["HostConfig"][field] = value
                with self.assertRaises(setup.SetupError):
                    setup.replacement_config(data, "derived")

    def test_replaces_old_socket_without_duplicates(self):
        data = original()
        data["HostConfig"]["Binds"].append("/old/docker.sock:/var/run/docker.sock")
        result = setup.replacement_config(data, "derived", "/run/user/1000/docker.sock")
        self.assertNotIn("/old/docker.sock:/var/run/docker.sock", result["HostConfig"]["Binds"])
        sockets = [m for m in result["HostConfig"]["Mounts"] if m["Target"] == setup.SOCKET]
        self.assertEqual(len(sockets), 1)
        self.assertEqual(sockets[0]["Source"], "/run/user/1000/docker.sock")


class InstallTests(unittest.TestCase):
    def engine(self):
        engine = Mock()
        engine.inspect.side_effect = [setup.SetupError("not found"), {"State": {"Running": True}}, {"NetworkSettings": {"Ports": {}}}]
        engine.pull.return_value = "cli-image"
        engine.create.side_effect = ["donor", "staging", "replacement"]
        def request(method, path, *args, **kwargs):
            if path.startswith("/images/"):
                return {"Architecture": "arm64"}
            if path.startswith("/commit"):
                return {"Id": "snapshot"}
            if "/archive?" in path and method == "GET":
                return b"archive"
        engine.request.side_effect = request
        return engine

    def test_success_preserves_original_as_non_restarting_backup(self):
        engine = self.engine()
        with patch.object(setup, "probe") as probe:
            setup.install(engine, original())
        engine.container.assert_any_call(original()["Id"], "update", {"RestartPolicy": {"Name": "no"}})
        engine.container.assert_any_call("replacement", "start")
        probe.assert_called_once_with(engine, "replacement", "python:3.11-slim")
        self.assertNotIn(original()["Id"], [call.args[0] for call in engine.remove.call_args_list])

    def test_failed_probe_restores_name_network_restart_policy_and_running_state(self):
        engine = self.engine()
        with patch.object(setup, "probe", side_effect=setup.SetupError("probe failed")):
            with self.assertRaisesRegex(setup.SetupError, "probe failed"):
                setup.install(engine, original())
        engine.remove.assert_any_call("replacement")
        engine.container.assert_any_call(original()["Id"], "rename", query={"name": "my-agent"})
        engine.container.assert_any_call(original()["Id"], "update", {"RestartPolicy": {"Name": "unless-stopped"}})
        engine.container.assert_any_call(original()["Id"], "start")
        self.assertFalse(any("/disconnect" in call.args[1] for call in engine.request.call_args_list))

    def test_download_failure_does_not_stop_original(self):
        engine = self.engine()
        engine.pull.side_effect = setup.SetupError("offline")
        with self.assertRaises(setup.SetupError):
            setup.install(engine, original())
        engine.container.assert_not_called()

    def test_existing_backup_blocks_setup(self):
        engine = self.engine()
        engine.inspect.side_effect = None
        engine.inspect.return_value = {}
        with self.assertRaisesRegex(setup.SetupError, "backup already exists"):
            setup.install(engine, original())
        engine.pull.assert_not_called()


if __name__ == "__main__":
    unittest.main()
