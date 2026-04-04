from __future__ import annotations

from pathlib import Path

from usr.plugins.rlm.helpers.persistence import RunStore


DEFAULTS = {
    "auto_enabled": True,
    "manual_tool_enabled": True,
    "trigger_threshold_pct": 0.8,
    "min_block_chars": 4000,
    "attachment_max_chars": 2_000_000,
    "environment_mode": "auto",
    "docker_image": "python:3.11-slim",
    "max_depth": 2,
    "max_iterations": 20,
    "max_budget": 0.0,
    "max_timeout": 120.0,
    "max_tokens": 0,
    "max_errors": 3,
    "max_concurrent_subcalls": 4,
    "subcall_model_source": "utility",
    "persistence_enabled": True,
    "retention_count": 25,
}


def get_plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_plugin_config(agent=None) -> dict:
    try:
        from helpers import plugins

        configured = plugins.get_plugin_config("rlm", agent=agent) or {}
    except Exception:
        configured = {}
    return {**DEFAULTS, **configured}


def get_runs_root() -> Path:
    return get_plugin_root() / "data" / "runs"


def get_run_store(agent=None) -> RunStore:
    config = get_plugin_config(agent)
    return RunStore(get_runs_root(), retention_count=int(config.get("retention_count", 25) or 25))


def get_chat_and_utility_configs(agent) -> tuple[dict, dict]:
    try:
        from plugins._model_config.helpers.model_config import (
            get_chat_model_config,
            get_utility_model_config,
        )

        return get_chat_model_config(agent) or {}, get_utility_model_config(agent) or {}
    except Exception:
        return {}, {}

