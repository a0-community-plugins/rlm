from __future__ import annotations

import unittest

import support  # noqa: F401

from usr.plugins.rlm.helpers.trajectory_view import build_run_view


class TrajectoryViewTests(unittest.TestCase):
    def test_upstream_logger_trajectory_is_projected_for_the_explorer(self):
        record = {
            "summary": {
                "status": "completed",
                "offloaded_block_count": 2,
                "approx_tokens_before": 9000,
                "approx_tokens_after": 1200,
            },
            "trajectory": {
                "run_metadata": {
                    "root_model": "gpt-example",
                    "environment_type": "docker",
                },
                "iterations": [
                    {
                        "type": "iteration",
                        "iteration": 1,
                        "timestamp": "2026-07-23T12:00:00Z",
                        "prompt": [{"role": "user", "content": "Inspect evidence"}],
                        "response": "I will inspect the offloaded block.",
                        "iteration_time": 1.25,
                        "final_answer": None,
                        "code_blocks": [
                            {
                                "code": "print(context['offloaded_blocks'][0]['id'])",
                                "result": {
                                    "stdout": "block-1\n",
                                    "stderr": "",
                                    "execution_time": 0.02,
                                    "final_answer": None,
                                    "rlm_calls": [
                                        {
                                            "root_model": "gpt-example",
                                            "prompt": "Summarize block-1",
                                            "response": "Relevant evidence",
                                            "execution_time": 0.4,
                                            "usage_summary": {
                                                "model_usage_summaries": {
                                                    "gpt-example": {
                                                        "total_input_tokens": 30,
                                                        "total_output_tokens": 4,
                                                        "total_calls": 1,
                                                    }
                                                },
                                                "total_cost": None,
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
            },
        }

        view = build_run_view(record)

        self.assertEqual(view["metrics"]["iteration_count"], 1)
        self.assertEqual(view["metrics"]["code_block_count"], 1)
        self.assertEqual(view["metrics"]["subcall_count"], 1)
        self.assertEqual(view["metrics"]["root_model"], "gpt-example")
        self.assertEqual(view["metrics"]["environment"], "docker")
        self.assertEqual(view["iterations"][0]["code_blocks"][0]["stdout"], "block-1\n")
        self.assertEqual(view["subcalls"][0]["total_input_tokens"], 30)
        self.assertEqual(view["subcalls"][0]["total_output_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
