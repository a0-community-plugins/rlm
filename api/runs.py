from __future__ import annotations

from helpers.api import ApiHandler, Request, Response

from usr.plugins.rlm_context.helpers.config import get_run_store
from usr.plugins.rlm_context.helpers.trajectory_view import build_run_view


class RunsApi(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        action = str(input.get("action", "list") or "list").lower()
        store = get_run_store()

        if action == "list":
            return {"success": True, "runs": store.list_runs()}
        if action == "get":
            run_id = str(input.get("run_id", "") or "")
            run = store.get_run(run_id)
            if run is None:
                return {"success": False, "error": "Run not found."}
            run = {**run, "view": build_run_view(run)}
            return {"success": True, "run": run}
        if action == "prune":
            keep = int(input.get("keep", store.retention_count) or 0)
            return {"success": True, "removed": store.prune_runs(keep=keep)}

        return {"success": False, "error": f"Unknown action: {action}"}
