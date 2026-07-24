from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support  # noqa: F401

from usr.plugins.rlm.helpers.context_packer import PackedContext
from usr.plugins.rlm.helpers.environment import EnvironmentResolution
from usr.plugins.rlm.helpers.provider_mapping import ProviderMapping
from usr.plugins.rlm.helpers import runtime


class FakeRLM:
    received_kwargs = None
    received_context = None
    received_prompt = None

    def __init__(self, **kwargs):
        type(self).received_kwargs = kwargs

    def completion(self, context, root_prompt):
        type(self).received_context = context
        type(self).received_prompt = root_prompt
        return SimpleNamespace(response="final response", metadata={"iterations": []})


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def payload(self):
        return runtime.RoutePayload(
            agent=None,
            packed=PackedContext(
                should_route=True,
                trigger_reason="oversized_external_context",
                visible_messages=[{"role": "user", "content": "placeholder"}],
                offloaded_blocks=[{"id": "block-1", "content": "evidence"}],
                approx_tokens_before=100,
                approx_tokens_after=10,
                threshold_tokens=50,
            ),
            root_mapping=ProviderMapping(
                supported=True,
                backend="openai",
                backend_kwargs={"model_name": "example"},
            ),
            subcall_mapping=None,
            environment=EnvironmentResolution(
                environment="docker",
                environment_kwargs={"image": "python:3.11-slim"},
            ),
            plugin_config={
                "max_depth": 2,
                "max_iterations": 5,
                "max_errors": 2,
                "max_concurrent_subcalls": 3,
            },
            call_kwargs={},
            finalizer_model=None,
        )

    def test_prompt_overlay_uses_upstream_answer_contract(self):
        overlay = runtime.AGENT_ZERO_RLM_PROMPT_OVERLAY
        self.assertIn('answer["content"]', overlay)
        self.assertIn('answer["ready"] = True', overlay)
        self.assertNotIn("FINAL(", overlay)
        self.assertNotIn("FINAL_VAR(", overlay)

    async def test_execute_uses_public_upstream_constructor_and_completion(self):
        with (
            patch.object(runtime, "_load_rlm_class", return_value=FakeRLM),
            patch.object(runtime, "_load_rlm_logger_class", return_value=None),
        ):
            completion, run_record = await runtime._execute_rlm(
                self.payload(),
                root_prompt="Answer from the evidence.",
            )

        self.assertEqual(completion["response"], "final response")
        self.assertEqual(FakeRLM.received_kwargs["environment"], "docker")
        self.assertEqual(FakeRLM.received_kwargs["max_concurrent_subcalls"], 3)
        self.assertEqual(
            FakeRLM.received_context["offloaded_blocks"][0]["id"],
            "block-1",
        )
        self.assertEqual(FakeRLM.received_prompt, "Answer from the evidence.")
        self.assertEqual(run_record["summary"]["environment"], "docker")

    def test_constructor_filter_preserves_only_supported_keywords(self):
        class StrictRLM:
            def __init__(self, backend, max_depth=1):
                pass

        filtered = runtime._filter_constructor_kwargs(
            StrictRLM,
            {"backend": "openai", "max_depth": 2, "unknown": True},
        )
        self.assertEqual(filtered, {"backend": "openai", "max_depth": 2})

    def test_internal_answer_assignment_requires_cleanup(self):
        self.assertTrue(
            runtime._response_needs_finalization(
                '```repl\nanswer["ready"] = True\n```'
            )
        )
        self.assertFalse(runtime._response_needs_finalization("A complete answer."))


if __name__ == "__main__":
    unittest.main()
