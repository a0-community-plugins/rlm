from __future__ import annotations

from typing import Any

from usr.plugins.rlm.helpers.bootstrap import get_dependency_status
from usr.plugins.rlm.helpers.config import (
    get_chat_and_utility_configs,
    get_plugin_config,
)
from usr.plugins.rlm.helpers.environment import resolve_environment
from usr.plugins.rlm.helpers.provider_mapping import (
    ProviderMapping,
    map_agent_zero_config_to_rlm,
)


def build_runtime_readiness(agent=None) -> dict[str, Any]:
    plugin_config = get_plugin_config(agent)
    dependency_status = get_dependency_status()
    environment = resolve_environment(plugin_config)
    chat_config, utility_config = get_chat_and_utility_configs(agent) if agent else ({}, {})
    chat_model = _safe_get_model(agent, "get_chat_model")
    utility_model = _safe_get_model(agent, "get_utility_model")

    root_mapping = _resolve_mapping(
        chat_config,
        runtime_model=chat_model,
        missing_reason="No active chat model is configured for RLM routing.",
    )
    utility_mapping = _resolve_mapping(
        utility_config,
        runtime_model=utility_model,
        missing_reason="No utility model is configured for recursive subcalls.",
    )

    subcall_source = str(plugin_config.get("subcall_model_source", "utility") or "utility").lower()
    if subcall_source == "utility":
        subcall_mapping = utility_mapping
        subcall_reason = utility_mapping.reason
    else:
        subcall_mapping = root_mapping
        subcall_reason = "Recursive subcalls will reuse the root model."

    blockers: list[str] = []
    advisories: list[str] = []

    dependency_ready = bool(
        dependency_status.get(
            "dependency_satisfied",
            dependency_status.get("dependency_installed", False),
        )
    )
    if not dependency_ready:
        blockers.append("RLM dependency is missing or outdated in the framework runtime.")
    if not root_mapping.supported:
        blockers.append(root_mapping.reason or "The active chat model does not map to an RLM client.")
    if not environment.usable:
        blockers.append(environment.reason or "The configured RLM environment is unavailable.")
    if subcall_source == "utility" and not utility_mapping.supported:
        advisories.append(
            utility_mapping.reason
            or "The utility model does not map cleanly to an RLM backend, so subcalls will reuse the root model."
        )

    auto_ready = (
        bool(plugin_config.get("auto_enabled", True))
        and dependency_ready
        and root_mapping.supported
        and environment.usable
    )
    manual_ready = (
        bool(plugin_config.get("manual_tool_enabled", True))
        and dependency_ready
        and root_mapping.supported
        and environment.usable
    )

    return {
        "auto_enabled": bool(plugin_config.get("auto_enabled", True)),
        "manual_tool_enabled": bool(plugin_config.get("manual_tool_enabled", True)),
        "auto_ready": auto_ready,
        "manual_ready": manual_ready,
        "blockers": blockers,
        "advisories": advisories,
        "environment": {
            "mode": str(plugin_config.get("environment_mode", "auto") or "auto"),
            "resolved": environment.environment,
            "usable": environment.usable,
            "reason": environment.reason,
        },
        "root_model": _mapping_summary(
            chat_config,
            chat_model,
            root_mapping,
        ),
        "subcall_model": {
            **_mapping_summary(
                utility_config if subcall_source == "utility" else chat_config,
                utility_model if subcall_source == "utility" else chat_model,
                subcall_mapping,
            ),
            "source": subcall_source,
            "reason": subcall_reason,
        },
    }


def _resolve_mapping(
    config: dict[str, Any] | None,
    *,
    runtime_model: Any | None,
    missing_reason: str,
) -> ProviderMapping:
    has_config = bool(config)
    if not has_config and runtime_model is None:
        return ProviderMapping(False, reason=missing_reason)
    return map_agent_zero_config_to_rlm(config, runtime_model=runtime_model)


def _mapping_summary(
    config: dict[str, Any] | None,
    runtime_model: Any | None,
    mapping: ProviderMapping,
) -> dict[str, Any]:
    provider = ""
    model_name = ""
    if isinstance(config, dict):
        provider = str(config.get("provider") or "").strip()
        model_name = str(config.get("name") or "").strip()
    provider = provider or str(getattr(runtime_model, "provider", "") or "").strip()
    model_name = model_name or str(getattr(runtime_model, "model_name", "") or "").strip()
    return {
        "provider": provider,
        "model_name": model_name,
        "backend": mapping.backend,
        "supported": mapping.supported,
        "reason": mapping.reason,
    }


def _safe_get_model(agent: Any, method_name: str) -> Any | None:
    if agent is None:
        return None
    getter = getattr(agent, method_name, None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None
