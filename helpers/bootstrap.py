from __future__ import annotations

from importlib import invalidate_caches
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
import json
import subprocess
import sys


DEPENDENCY_MODULE = "rlm"
DEPENDENCY_SOURCE_URL = "https://github.com/alexzhang13/rlm.git"
DEPENDENCY_TARGET_REVISION = "95bff825c11909bde7fb9d1257606f44886df869"
DEPENDENCY_PACKAGE = (
    f"rlms @ git+{DEPENDENCY_SOURCE_URL}@{DEPENDENCY_TARGET_REVISION}"
)
DEPENDENCY_DISTRIBUTION = "rlms"


def get_dependency_status() -> dict[str, object]:
    spec = find_spec(DEPENDENCY_MODULE)
    dependency_version = None
    dependency_source = None
    dependency_revision = None
    try:
        dependency_version = version(DEPENDENCY_DISTRIBUTION)
        dependency_source, dependency_revision = _read_dependency_direct_url()
    except PackageNotFoundError:
        dependency_version = None

    dependency_installed = spec is not None
    dependency_satisfied = dependency_installed and _matches_target_revision(
        dependency_source,
        dependency_revision,
    )

    return {
        "dependency_installed": dependency_installed,
        "dependency_satisfied": dependency_satisfied,
        "dependency_module": DEPENDENCY_MODULE,
        "dependency_package": DEPENDENCY_PACKAGE,
        "dependency_version": dependency_version,
        "dependency_source": dependency_source,
        "dependency_revision": dependency_revision,
        "dependency_target_source": DEPENDENCY_SOURCE_URL,
        "dependency_target_revision": DEPENDENCY_TARGET_REVISION,
        "framework_python": sys.executable,
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
            f"{DEPENDENCY_PACKAGE} was installed, but the required revision is still unavailable."
        )
    return status


def _read_dependency_direct_url() -> tuple[str | None, str | None]:
    try:
        direct_url_text = distribution(DEPENDENCY_DISTRIBUTION).read_text("direct_url.json")
    except Exception:
        return None, None
    if not direct_url_text:
        return None, None
    try:
        payload = json.loads(direct_url_text)
    except Exception:
        return None, None
    source = str(payload.get("url") or "").strip() or None
    vcs_info = dict(payload.get("vcs_info") or {})
    revision = str(vcs_info.get("commit_id") or "").strip() or None
    return source, revision


def _matches_target_revision(source: str | None, revision: str | None) -> bool:
    if not source or not revision:
        return False
    normalized_source = source.removesuffix("/")
    normalized_target = DEPENDENCY_SOURCE_URL.removesuffix("/")
    return (
        normalized_source == normalized_target
        and revision == DEPENDENCY_TARGET_REVISION
    )
