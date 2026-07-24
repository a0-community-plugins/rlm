from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
import unittest

import support  # noqa: F401

from usr.plugins.rlm.helpers.upstream_compat import patch_nullable_usage_tracking


class FakeOpenAIClient:
    valid_calls = 0

    def __init__(self):
        self.model_call_counts = defaultdict(int)
        self.model_input_tokens = defaultdict(int)
        self.model_output_tokens = defaultdict(int)
        self.model_total_tokens = defaultdict(int)
        self.model_costs = defaultdict(float)

    def _track_cost(self, response, model):
        type(self).valid_calls += 1
        usage = response.usage
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += usage.prompt_tokens
        self.model_output_tokens[model] += usage.completion_tokens
        self.model_total_tokens[model] += usage.total_tokens
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_completion_tokens = usage.completion_tokens
        self.last_cost = None


class UpstreamCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch_nullable_usage_tracking(FakeOpenAIClient)

    def test_nullable_usage_is_counted_as_zero_instead_of_crashing(self):
        client = FakeOpenAIClient()
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                model_extra=None,
            )
        )

        client._track_cost(response, "gpt-test")

        self.assertEqual(client.model_call_counts["gpt-test"], 1)
        self.assertEqual(client.model_input_tokens["gpt-test"], 0)
        self.assertEqual(client.model_output_tokens["gpt-test"], 0)
        self.assertEqual(client.model_total_tokens["gpt-test"], 0)
        self.assertEqual(client.last_prompt_tokens, 0)
        self.assertEqual(client.last_completion_tokens, 0)

    def test_complete_usage_keeps_the_upstream_tracking_path(self):
        client = FakeOpenAIClient()
        before = FakeOpenAIClient.valid_calls
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            )
        )

        client._track_cost(response, "gpt-test")

        self.assertEqual(FakeOpenAIClient.valid_calls, before + 1)
        self.assertEqual(client.model_total_tokens["gpt-test"], 14)

    def test_patch_is_idempotent(self):
        self.assertFalse(patch_nullable_usage_tracking(FakeOpenAIClient))


if __name__ == "__main__":
    unittest.main()
