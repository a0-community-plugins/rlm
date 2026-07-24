from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import support  # noqa: F401

from usr.plugins.rlm.helpers import docker_setup
from usr.plugins.rlm.helpers.docker_shim import rewrite_rlm_run_args


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _inspection() -> dict:
    return {
        "NetworkSettings": {
            "Networks": {
                "agent-zero_default": {
                    "IPAddress": "172.30.0.4",
                }
            }
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/host/agent-zero",
                "Destination": "/a0",
            }
        ],
    }


class DockerShimTests(unittest.TestCase):
    def test_rlm_sandbox_joins_outer_network_and_reaches_callback(self):
        args = [
            "run",
            "-d",
            "--rm",
            "-v",
            "/a0/.rlm_workspace/docker_repl_123:/workspace",
            "--add-host",
            "host.docker.internal:host-gateway",
            "python:3.11-slim",
            "tail",
            "-f",
            "/dev/null",
        ]

        rewritten = rewrite_rlm_run_args(args, _inspection())

        self.assertEqual(rewritten[1:3], ["--network", "agent-zero_default"])
        self.assertIn("host.docker.internal:172.30.0.4", rewritten)
        self.assertIn(
            "/host/agent-zero/.rlm_workspace/docker_repl_123:/workspace",
            rewritten,
        )

    def test_unrelated_docker_commands_are_not_changed(self):
        args = ["run", "--rm", "hello-world"]
        self.assertEqual(rewrite_rlm_run_args(args, _inspection()), args)

    def test_configured_network_must_be_attached(self):
        with self.assertRaisesRegex(RuntimeError, "not attached"):
            rewrite_rlm_run_args(
                [
                    "run",
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "python:3.11-slim",
                ],
                _inspection(),
                requested_network="missing",
            )

    def test_workspace_must_be_visible_to_external_daemon(self):
        with self.assertRaisesRegex(RuntimeError, "not inside a bind mount"):
            rewrite_rlm_run_args(
                [
                    "run",
                    "-v",
                    "/private/runtime/probe:/workspace:ro",
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "python:3.11-slim",
                ],
                _inspection(),
            )


class DockerSetupTests(unittest.TestCase):
    def test_status_separates_cli_endpoint_and_daemon_readiness(self):
        with (
            patch.object(docker_setup, "activate_docker_cli_shim", return_value="shim"),
            patch.object(
                docker_setup,
                "find_real_docker_cli",
                return_value="/usr/local/libexec/rlm-docker/docker",
            ),
            patch.object(docker_setup, "detect_containerized_runtime", return_value=True),
            patch.object(docker_setup, "_endpoint_summary", return_value=("Unix socket", True)),
            patch.object(docker_setup, "docker_info_available", return_value=True),
            patch.object(docker_setup, "read_probe_record", return_value=None),
        ):
            status = docker_setup.get_docker_setup_status()

        self.assertTrue(status["cli_available"])
        self.assertTrue(status["endpoint_present"])
        self.assertTrue(status["daemon_reachable"])
        self.assertFalse(status["setup_required"])
        self.assertEqual(status["cli_source"], "RLM derived image")

    def test_repository_ships_host_setup_and_compose_overlay(self):
        script = PLUGIN_ROOT / "setup" / "enable-docker-access.sh"
        self.assertTrue(script.is_file())
        self.assertTrue(script.stat().st_mode & 0o111)
        self.assertTrue((PLUGIN_ROOT / "setup" / "Dockerfile.rlm").is_file())
        overlay = (PLUGIN_ROOT / "setup" / "docker-compose.rlm.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("/var/run/docker.sock", overlay)
        self.assertIn("RLM_DOCKER_SOCKET", overlay)


if __name__ == "__main__":
    unittest.main()
