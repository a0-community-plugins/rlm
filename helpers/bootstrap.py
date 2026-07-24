from __future__ import annotations

from importlib import invalidate_caches
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys


DEPENDENCY_MODULE = "rlm"
DEPENDENCY_DISTRIBUTION = "rlms"
DEPENDENCY_TARGET_VERSION = "0.1.3"
DEPENDENCY_PACKAGE = f"{DEPENDENCY_DISTRIBUTION}=={DEPENDENCY_TARGET_VERSION}"
DEPENDENCY_UPSTREAM_URL = "https://github.com/alexzhang13/rlm"
FRAMEWORK_PYTHON_ENV = "A0_FRAMEWORK_PYTHON"


def get_dependency_status() -> dict[str, object]:
    try:
        spec = find_spec(DEPENDENCY_MODULE)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    dependency_version = None
    try:
        dependency_version = version(DEPENDENCY_DISTRIBUTION)
    except PackageNotFoundError:
        dependency_version = None

    dependency_installed = spec is not None
    dependency_satisfied = (
        dependency_installed and dependency_version == DEPENDENCY_TARGET_VERSION
    )

    return {
        "dependency_installed": dependency_installed,
        "dependency_satisfied": dependency_satisfied,
        "dependency_module": DEPENDENCY_MODULE,
        "dependency_package": DEPENDENCY_PACKAGE,
        "dependency_version": dependency_version,
        "dependency_target_version": DEPENDENCY_TARGET_VERSION,
        "dependency_upstream_url": DEPENDENCY_UPSTREAM_URL,
        "framework_python": sys.executable,
        "preferred_framework_python": find_framework_python(),
    }


def ensure_rlm_dependency() -> dict[str, object]:
    status = get_dependency_status()
    if status["dependency_satisfied"]:
        return status

    result = subprocess.run(
        [
            str(status["framework_python"]),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            DEPENDENCY_PACKAGE,
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
        raise RuntimeError(output or f"Failed to install {DEPENDENCY_PACKAGE}.")

    invalidate_caches()
    status = get_dependency_status()
    if not status["dependency_satisfied"]:
        raise RuntimeError(
            f"{DEPENDENCY_PACKAGE} was installed, but the required version is still unavailable."
        )
    return status


def find_framework_python() -> str:
    configured = str(os.getenv(FRAMEWORK_PYTHON_ENV, "") or "").strip()
    candidates = [
        configured,
        "/opt/venv-a0/bin/python3",
        "/opt/venv-a0/bin/python",
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return sys.executable
