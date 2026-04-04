from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from helpers.tool import Response, Tool
from helpers.history import output_text
from usr.plugins.rlm.helpers.config import (
    get_chat_and_utility_configs,
    get_plugin_config,
    get_run_store,
)
from usr.plugins.rlm.helpers.context_packer import pack_messages_for_rlm
from usr.plugins.rlm.helpers.environment import resolve_environment
from usr.plugins.rlm.helpers.provider_mapping import map_agent_zero_config_to_rlm
from usr.plugins.rlm.helpers.readiness import build_runtime_readiness
from usr.plugins.rlm.helpers.runtime import RoutePayload, run_manual_tool


class RLMContextTool(Tool):
    async def execute(
        self,
        question: str = "",
        source: str = "recent_history",
        history_turns: int = 6,
        **kwargs,
    ) -> Response:
        plugin_config = get_plugin_config(self.agent)
        if not plugin_config.get("manual_tool_enabled", True):
            return Response(
                message="RLM manual analysis is disabled in the plugin settings.",
                break_loop=False,
            )

        readiness = build_runtime_readiness(self.agent)
        if not readiness.get("manual_ready", False):
            blocker = (readiness.get("blockers") or ["RLM is not ready for this agent."])[0]
            return Response(
                message=f"RLM is not ready for manual analysis: {blocker}",
                break_loop=False,
                additional={"rlm_readiness": readiness},
            )

        chat_config, utility_config = get_chat_and_utility_configs(self.agent)
        chat_model = self.agent.get_chat_model() if self.agent else None
        utility_model = self.agent.get_utility_model() if self.agent else None

        outputs = list(self.agent.history.output())
        selected = outputs[-history_turns:] if history_turns > 0 else outputs
        messages = []
        for item in selected:
            content = _langchain_safe_content(item.get("content", ""))
            if item.get("ai"):
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        packed = pack_messages_for_rlm(
            messages,
            {
                **plugin_config,
                "ctx_length": int(chat_config.get("ctx_length", 128000) or 128000),
            },
        )
        try:
            root_mapping = map_agent_zero_config_to_rlm(
                chat_config,
                runtime_model=chat_model,
            )
            utility_mapping = map_agent_zero_config_to_rlm(
                utility_config,
                runtime_model=utility_model,
            )
        except TypeError:
            root_mapping = map_agent_zero_config_to_rlm(chat_config)
            utility_mapping = map_agent_zero_config_to_rlm(utility_config)
        payload = RoutePayload(
            agent=self.agent,
            packed=packed,
            root_mapping=root_mapping,
            subcall_mapping=utility_mapping if utility_mapping.supported else None,
            environment=resolve_environment(plugin_config),
            plugin_config=plugin_config,
            call_kwargs={
                "question": question or f"Analyze Agent Zero {source} context.",
                "source": source,
                "history_excerpt": output_text(selected) if selected else "",
            },
            finalizer_model=getattr(chat_model, "base_model", chat_model) or utility_model,
        )

        result = await run_manual_tool(payload)
        get_run_store(self.agent).save_run(result["run_record"])

        message = (
            f"RLM analysis result:\n{result['response']}\n\n"
            f"Run ID: {result['run_record']['run_id']}"
        )
        return Response(message=message, break_loop=False)


def _langchain_safe_content(content):
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)
