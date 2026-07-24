from __future__ import annotations

from helpers.api import ApiHandler, Request, Response

from usr.plugins.rlm.helpers.config import get_plugin_config
from usr.plugins.rlm.helpers.docker_setup import (
    get_docker_setup_status,
    run_docker_sandbox_probe,
)


class DockerSetupApi(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        action = str(input.get("action", "status") or "status").lower()
        if action == "status":
            return {"success": True, "docker_setup": get_docker_setup_status()}
        if action == "probe":
            config = get_plugin_config()
            result = run_docker_sandbox_probe(
                str(config.get("docker_image", "python:3.11-slim") or "python:3.11-slim")
            )
            return {
                "success": bool(result.get("success")),
                "probe": result,
                "docker_setup": get_docker_setup_status(),
            }
        return {"success": False, "error": f"Unknown action: {action}"}
