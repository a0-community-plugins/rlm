from __future__ import annotations

import unittest
from unittest.mock import patch

import support  # noqa: F401

from usr.plugins.rlm.helpers import environment


class EnvironmentTests(unittest.TestCase):
    def test_auto_selects_docker_when_available_on_host(self):
        with (
            patch.object(environment, "is_docker_available", return_value=True),
            patch.object(environment, "detect_containerized_runtime", return_value=False),
            patch.object(environment, "has_external_docker_access", return_value=False),
        ):
            result = environment.resolve_environment({"environment_mode": "auto"})

        self.assertTrue(result.usable)
        self.assertEqual(result.environment, "docker")

    def test_auto_blocks_instead_of_falling_back_to_local(self):
        with (
            patch.object(environment, "is_docker_available", return_value=False),
            patch.object(environment, "detect_containerized_runtime", return_value=False),
            patch.object(environment, "has_external_docker_access", return_value=False),
        ):
            result = environment.resolve_environment({"environment_mode": "auto"})

        self.assertFalse(result.usable)
        self.assertEqual(result.environment, "docker")
        self.assertIn("requires an isolated Docker REPL", result.reason)

    def test_container_requires_external_docker_access(self):
        with (
            patch.object(environment, "is_docker_available", return_value=True),
            patch.object(environment, "detect_containerized_runtime", return_value=True),
            patch.object(environment, "has_external_docker_access", return_value=False),
        ):
            result = environment.resolve_environment({"environment_mode": "auto"})

        self.assertFalse(result.usable)
        self.assertEqual(result.environment, "docker")

    def test_local_mode_is_explicit_and_warns_about_isolation(self):
        with (
            patch.object(environment, "is_docker_available", return_value=False),
            patch.object(environment, "detect_containerized_runtime", return_value=False),
            patch.object(environment, "has_external_docker_access", return_value=False),
        ):
            result = environment.resolve_environment({"environment_mode": "local"})

        self.assertTrue(result.usable)
        self.assertEqual(result.environment, "local")
        self.assertIn("without Docker isolation", result.reason)


if __name__ == "__main__":
    unittest.main()
