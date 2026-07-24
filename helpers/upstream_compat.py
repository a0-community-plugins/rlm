from __future__ import annotations

from collections import defaultdict
from importlib import import_module
from typing import Any


PATCH_MARKER = "_a0_nullable_usage_compat"


def apply_upstream_compatibility() -> list[str]:
    patched: list[str] = []
    targets = (
        ("rlm.clients.openai", "OpenAIClient"),
        ("rlm.clients.azure_openai", "AzureOpenAIClient"),
    )
    for module_name, class_name in targets:
        try:
            client_class = getattr(import_module(module_name), class_name)
        except Exception:
            continue
        if patch_nullable_usage_tracking(client_class):
            patched.append(f"{module_name}.{class_name}")
    return patched


def patch_nullable_usage_tracking(client_class: type[Any]) -> bool:
    original = getattr(client_class, "_track_cost", None)
    if original is None or getattr(original, PATCH_MARKER, False):
        return False

    def compatible_track_cost(self, response: Any, model: str):
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = (
            getattr(usage, "completion_tokens", None) if usage is not None else None
        )
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        if (
            usage is not None
            and prompt_tokens is not None
            and completion_tokens is not None
            and total_tokens is not None
        ):
            return original(self, response, model)

        _ensure_usage_counters(self)
        prompt_value = _safe_nonnegative_int(prompt_tokens)
        completion_value = _safe_nonnegative_int(completion_tokens)
        total_value = _safe_nonnegative_int(total_tokens)
        if total_tokens is None:
            total_value = prompt_value + completion_value

        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += prompt_value
        self.model_output_tokens[model] += completion_value
        self.model_total_tokens[model] += total_value
        self.last_prompt_tokens = prompt_value
        self.last_completion_tokens = completion_value
        self.last_cost = None
        _track_optional_cost(self, usage, model)
        return None

    setattr(compatible_track_cost, PATCH_MARKER, True)
    setattr(compatible_track_cost, "_a0_original", original)
    client_class._track_cost = compatible_track_cost
    return True


def _ensure_usage_counters(client: Any) -> None:
    defaults = (
        ("model_call_counts", int),
        ("model_input_tokens", int),
        ("model_output_tokens", int),
        ("model_total_tokens", int),
        ("model_costs", float),
    )
    for name, factory in defaults:
        if not hasattr(client, name):
            setattr(client, name, defaultdict(factory))


def _safe_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _track_optional_cost(client: Any, usage: Any, model: str) -> None:
    if usage is None:
        return
    cost = getattr(usage, "cost", None)
    if not cost:
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            cost = extra.get("cost")
            if not cost:
                details = extra.get("cost_details")
                if isinstance(details, dict):
                    cost = details.get("upstream_inference_cost")
    try:
        numeric_cost = float(cost)
    except (TypeError, ValueError):
        return
    if numeric_cost <= 0:
        return
    client.last_cost = numeric_cost
    client.model_costs[model] += numeric_cost
