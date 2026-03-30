from __future__ import annotations

from typing import Any


def build_run_view(record: dict[str, Any] | None) -> dict[str, Any]:
    record = dict(record or {})
    summary = dict(record.get("summary") or {})
    trajectory = dict(record.get("trajectory") or {})
    run_metadata = dict(trajectory.get("run_metadata") or {})
    iterations_raw = list(trajectory.get("iterations") or [])

    iterations: list[dict[str, Any]] = []
    subcalls: list[dict[str, Any]] = []

    total_code_blocks = 0
    total_subcalls = 0
    error_iteration_count = 0
    final_answer_iteration: int | None = None

    for fallback_index, raw_iteration in enumerate(iterations_raw, start=1):
        iteration_number = int(raw_iteration.get("iteration") or fallback_index)
        code_blocks_raw = list(raw_iteration.get("code_blocks") or [])
        code_blocks: list[dict[str, Any]] = []
        iteration_subcalls: list[dict[str, Any]] = []
        had_error = False

        for block_index, raw_block in enumerate(code_blocks_raw, start=1):
            result = dict(raw_block.get("result") or {})
            stderr = _to_text(result.get("stderr"))
            stdout = _to_text(result.get("stdout"))
            had_block_error = bool(stderr.strip())
            had_error = had_error or had_block_error

            block_subcalls: list[dict[str, Any]] = []
            for subcall_index, raw_call in enumerate(result.get("rlm_calls") or [], start=1):
                usage = dict(raw_call.get("usage_summary") or {})
                call_summary = _usage_summary(usage)
                subcall = {
                    "iteration": iteration_number,
                    "code_block_index": block_index,
                    "subcall_index": subcall_index,
                    "model": _to_text(raw_call.get("root_model")),
                    "prompt_preview": _preview(raw_call.get("prompt"), 220),
                    "response_preview": _preview(raw_call.get("response"), 220),
                    "execution_time": _to_float(raw_call.get("execution_time")),
                    **call_summary,
                }
                block_subcalls.append(subcall)
                iteration_subcalls.append(subcall)
                subcalls.append(subcall)

            code_blocks.append(
                {
                    "index": block_index,
                    "code": _to_text(raw_block.get("code")),
                    "stdout": stdout,
                    "stderr": stderr,
                    "had_error": had_block_error,
                    "execution_time": _to_float(result.get("execution_time")),
                    "subcall_count": len(block_subcalls),
                    "subcalls": block_subcalls,
                    "final_answer": _to_text(result.get("final_answer")),
                }
            )

        final_answer = _to_text(raw_iteration.get("final_answer"))
        if had_error:
            error_iteration_count += 1
        if final_answer and final_answer_iteration is None:
            final_answer_iteration = iteration_number

        total_code_blocks += len(code_blocks)
        total_subcalls += len(iteration_subcalls)

        iteration_view = {
            "iteration": iteration_number,
            "timestamp": _to_text(raw_iteration.get("timestamp")),
            "prompt_message_count": _message_count(raw_iteration.get("prompt")),
            "response": _to_text(raw_iteration.get("response")),
            "response_preview": _preview(raw_iteration.get("response"), 220),
            "response_length": len(_to_text(raw_iteration.get("response"))),
            "code_block_count": len(code_blocks),
            "subcall_count": len(iteration_subcalls),
            "had_error": had_error,
            "final_answer": final_answer,
            "iteration_time": _to_float(raw_iteration.get("iteration_time")),
            "activity_score": max(1, len(code_blocks) + len(iteration_subcalls)),
            "code_blocks": code_blocks,
        }
        iterations.append(iteration_view)

    max_iteration_time = max([item["iteration_time"] for item in iterations] or [0.0])
    max_activity_score = max([item["activity_score"] for item in iterations] or [1])
    max_subcalls_per_iteration = max([item["subcall_count"] for item in iterations] or [0])

    for item in iterations:
        item["time_pct"] = _ratio_pct(item["iteration_time"], max_iteration_time)
        item["activity_pct"] = _ratio_pct(item["activity_score"], max_activity_score)
        item["subcall_pct"] = _ratio_pct(item["subcall_count"], max_subcalls_per_iteration)

    total_iteration_time = round(sum(item["iteration_time"] for item in iterations), 3)

    return {
        "run_metadata": run_metadata,
        "metrics": {
            "status": _to_text(summary.get("status")) or "completed",
            "error_type": _to_text(summary.get("error_type")),
            "error_message": _to_text(summary.get("error_message")),
            "iteration_count": len(iterations),
            "code_block_count": total_code_blocks,
            "subcall_count": total_subcalls,
            "error_iteration_count": error_iteration_count,
            "final_answer_iteration": final_answer_iteration,
            "total_iteration_time": total_iteration_time,
            "partial_response": bool(summary.get("partial_response")),
            "offloaded_block_count": int(summary.get("offloaded_block_count", 0) or 0),
            "approx_tokens_before": int(summary.get("approx_tokens_before", 0) or 0),
            "approx_tokens_after": int(summary.get("approx_tokens_after", 0) or 0),
            "root_model": _to_text(summary.get("root_model") or run_metadata.get("root_model")),
            "environment": _to_text(summary.get("environment") or run_metadata.get("environment_type")),
        },
        "chart": {
            "max_iteration_time": max_iteration_time,
            "max_activity_score": max_activity_score,
            "max_subcalls_per_iteration": max_subcalls_per_iteration,
        },
        "iterations": iterations,
        "subcalls": subcalls,
    }


def _usage_summary(usage: dict[str, Any]) -> dict[str, Any]:
    model_usage = dict(usage.get("model_usage_summaries") or {})
    total_input = 0
    total_output = 0
    total_calls = 0
    for summary in model_usage.values():
        total_input += int(summary.get("total_input_tokens", 0) or 0)
        total_output += int(summary.get("total_output_tokens", 0) or 0)
        total_calls += int(summary.get("total_calls", 0) or 0)
    return {
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost": usage.get("total_cost"),
    }


def _ratio_pct(value: float | int, maximum: float | int) -> float:
    max_value = float(maximum or 0)
    if max_value <= 0:
        return 0.0
    return round((float(value or 0) / max_value) * 100, 2)


def _message_count(prompt: Any) -> int:
    if isinstance(prompt, list):
        return len(prompt)
    if prompt:
        return 1
    return 0


def _preview(value: Any, limit: int) -> str:
    text = _to_text(value).strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}…"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
