from __future__ import annotations

from helpers import plugins
from helpers.api import ApiHandler, Request

from usr.plugins.rlm.helpers.bootstrap import get_dependency_status
from usr.plugins.rlm.helpers.docker_setup import get_docker_setup_status


class InstallSetup(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        plugins.call_plugin_hook("rlm", "install")
        return {
            "success": True,
            **get_dependency_status(),
            "docker_setup": get_docker_setup_status(),
        }
