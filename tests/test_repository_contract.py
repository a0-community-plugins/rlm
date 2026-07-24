from __future__ import annotations

from pathlib import Path
import unittest

import support


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_public_plugin_has_required_contribution_files(self):
        for filename in (
            "plugin.yaml",
            "plugin.json",
            "README.md",
            "LICENSE",
            "hooks.py",
        ):
            self.assertTrue((PLUGIN_ROOT / filename).is_file(), filename)

    def test_manifest_identity_and_version_match_repository(self):
        manifest = (PLUGIN_ROOT / "plugin.yaml").read_text(encoding="utf-8")
        self.assertIn("name: rlm", manifest)
        self.assertIn('version: "2.0.0"', manifest)

    def test_runtime_configuration_is_not_committed_as_source(self):
        self.assertFalse((PLUGIN_ROOT / "config.json").exists())
        self.assertEqual(list(PLUGIN_ROOT.glob(".toggle-*")), [])
        ignore_rules = (PLUGIN_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("config.json", ignore_rules.splitlines())
        self.assertIn("data/", ignore_rules.splitlines())

    def test_plugin_is_developed_inside_agent_zero_user_plugins(self):
        expected = (support.PROJECT_ROOT / "usr" / "plugins" / "rlm").resolve()
        self.assertEqual(PLUGIN_ROOT.resolve(), expected)

    def test_explorer_long_notices_wrap_inside_narrow_modals(self):
        explorer = (PLUGIN_ROOT / "webui" / "main.html").read_text(encoding="utf-8")
        store = (PLUGIN_ROOT / "webui" / "rlm-context-store.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('<template x-if="$store.rlmContextDashboard">', explorer)
        self.assertNotIn("<div x-data x-init=", explorer)
        self.assertIn('class="rcd-notice is-warning"', explorer)
        self.assertIn("overflow-wrap: anywhere;", explorer)
        self.assertIn("grid-template-columns: minmax(0, 1fr);", explorer)
        self.assertNotIn('class="rcd-pill is-warning" x-text="item"', explorer)
        self.assertNotIn("async init()", store)
        self.assertIn("this.openPromise = Promise.all", store)

    def test_install_hook_owns_automatic_dependency_setup(self):
        hooks = (PLUGIN_ROOT / "hooks.py").read_text(encoding="utf-8")
        self.assertIn("def install() -> None:", hooks)
        self.assertIn("_ensure_rlm_dependency()", hooks)
        self.assertFalse((PLUGIN_ROOT / "execute.py").exists())
        readme = (PLUGIN_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("install hook automatically installs", readme)
        self.assertIn("There is no separate Execute step", readme)


if __name__ == "__main__":
    unittest.main()
