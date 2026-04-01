from __future__ import annotations

import importlib
import shutil
from pathlib import Path
import sys

from usr.plugins.rlm.helpers.bootstrap import ensure_rlm_dependency

PLUGIN_PACKAGE_PREFIX = "usr.plugins.rlm"
PLUGIN_ROOT = Path(__file__).resolve().parent


def install() -> None:
    _ensure_rlm_dependency()
    _clear_plugin_modules()
    _clear_plugin_bytecode()


def pre_update() -> None:
    _clear_plugin_modules()
    _clear_plugin_bytecode()


def _ensure_rlm_dependency() -> None:
    ensure_rlm_dependency()


def _clear_plugin_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == PLUGIN_PACKAGE_PREFIX or module_name.startswith(
            f"{PLUGIN_PACKAGE_PREFIX}."
        ):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _clear_plugin_bytecode() -> None:
    for cache_dir in PLUGIN_ROOT.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
