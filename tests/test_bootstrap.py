from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import support  # noqa: F401

from usr.plugins.rlm.helpers import bootstrap


class BootstrapTests(unittest.TestCase):
    def test_dependency_status_requires_exact_stable_version(self):
        with (
            patch.object(bootstrap, "find_spec", return_value=object()),
            patch.object(
                bootstrap,
                "version",
                return_value=bootstrap.DEPENDENCY_TARGET_VERSION,
            ),
        ):
            status = bootstrap.get_dependency_status()

        self.assertTrue(status["dependency_satisfied"])
        self.assertEqual(status["dependency_package"], "rlms==0.1.3")
        self.assertEqual(status["dependency_target_version"], "0.1.3")

    def test_dependency_status_rejects_outdated_version(self):
        with (
            patch.object(bootstrap, "find_spec", return_value=object()),
            patch.object(bootstrap, "version", return_value="0.1.1"),
        ):
            status = bootstrap.get_dependency_status()

        self.assertTrue(status["dependency_installed"])
        self.assertFalse(status["dependency_satisfied"])

    def test_framework_python_override_must_be_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            python_path = Path(temp_dir) / "framework-python"
            python_path.write_text("#!/bin/sh\n", encoding="utf-8")
            python_path.chmod(0o700)

            with patch.dict(
                os.environ,
                {bootstrap.FRAMEWORK_PYTHON_ENV: str(python_path)},
            ):
                self.assertEqual(bootstrap.find_framework_python(), str(python_path))

    def test_install_uses_the_current_framework_interpreter(self):
        before = {
            "dependency_satisfied": False,
            "framework_python": "/framework/python",
        }
        after = {
            "dependency_satisfied": True,
            "framework_python": "/framework/python",
        }
        completed = type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with (
            patch.object(bootstrap, "get_dependency_status", side_effect=[before, after]),
            patch.object(bootstrap.subprocess, "run", return_value=completed) as run,
        ):
            result = bootstrap.ensure_rlm_dependency()

        self.assertIs(result, after)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["/framework/python", "-m", "pip", "install"])
        self.assertEqual(command[-1], "rlms==0.1.3")


if __name__ == "__main__":
    unittest.main()
