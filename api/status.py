from __future__ import annotations

from agent import AgentContext
from helpers.api import ApiHandler, Request

from usr.plugins.rlm_context.helpers.bootstrap import get_dependency_status
from usr.plugins.rlm_context.helpers.config import get_plugin_config
from usr.plugins.rlm_context.helpers.environment import (
    detect_containerized_runtime,
    is_docker_available,
)
from usr.plugins.rlm_context.helpers.readiness import build_runtime_readiness


class StatusApi(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        active_context = AgentContext.first()
        agent = getattr(active_context, "agent0", None) if active_context else None
        try:
            config = get_plugin_config(agent)
        except TypeError:
            config = get_plugin_config()
        dependency_status = get_dependency_status()
        return {
            "success": True,
            **dependency_status,
            "docker_available": is_docker_available(),
            "containerized_runtime": detect_containerized_runtime(),
            "config": config,
            "active_context": {
                "id": getattr(active_context, "id", ""),
                "name": getattr(active_context, "name", "") or "",
            }
            if active_context
            else None,
            "readiness": _build_readiness(agent),
        }


def _build_readiness(agent):
    try:
        return build_runtime_readiness(agent)
    except TypeError:
        return build_runtime_readiness()
