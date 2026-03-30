from helpers.extension import Extension

from usr.plugins.rlm_context.helpers.config import (
    get_chat_and_utility_configs,
    get_plugin_config,
)
from usr.plugins.rlm_context.helpers.runtime import RLMChatWrapper


class RLMContextChatModel(Extension):
    def execute(self, data: dict = {}, **kwargs):
        if not self.agent:
            return
        base_model = data.get("result")
        if base_model is None:
            return

        plugin_config = get_plugin_config(self.agent)
        if not plugin_config.get("auto_enabled", True):
            return

        chat_config, utility_config = get_chat_and_utility_configs(self.agent)
        utility_model = None
        try:
            utility_model = self.agent.get_utility_model()
        except Exception:
            utility_model = None
        data["result"] = RLMChatWrapper(
            agent=self.agent,
            base_model=base_model,
            utility_model=utility_model,
            chat_model_config=chat_config,
            utility_model_config=utility_config,
            plugin_config=plugin_config,
        )
