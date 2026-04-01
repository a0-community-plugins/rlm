from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from usr.plugins.rlm.helpers.bootstrap import ensure_rlm_dependency, get_dependency_status


def main() -> int:
    before = get_dependency_status()
    print(f"Framework Python: {before['framework_python']}")
    print(f"Expected package: {before['dependency_package']}")
    print(f"Expected module: {before['dependency_module']}")
    print(f"Installed before run: {before['dependency_installed']}")
    print(f"Satisfied before run: {before['dependency_satisfied']}")
    if before.get("dependency_target_revision"):
        print(f"Target revision: {before['dependency_target_revision']}")

    try:
        after = ensure_rlm_dependency()
    except Exception as exc:
        print(f"Dependency install failed: {exc}")
        return 1

    print(f"Installed after run: {after['dependency_installed']}")
    print(f"Satisfied after run: {after['dependency_satisfied']}")
    if after.get("dependency_version"):
        print(f"Installed version: {after['dependency_version']}")
    if after.get("dependency_revision"):
        print(f"Installed revision: {after['dependency_revision']}")
    print("RLM dependency is ready in the Agent Zero framework runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
