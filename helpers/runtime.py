from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
import inspect
import json
import os
import re
from typing import Any
from uuid import uuid4

from usr.plugins.rlm.helpers.config import get_run_store
from usr.plugins.rlm.helpers.context_packer import PackedContext, pack_messages_for_rlm
from usr.plugins.rlm.helpers.environment import EnvironmentResolution, resolve_environment
from usr.plugins.rlm.helpers.provider_mapping import ProviderMapping, map_agent_zero_config_to_rlm


AGENT_ZERO_RLM_PROMPT_OVERLAY = """
Agent Zero specific instructions:
- You are producing the single next assistant message for Agent Zero from the visible conversation and the offloaded external context.
- Keep using the REPL and the recursive query tools from this prompt. When you need deeper reasoning over large offloaded blocks, cross-block synthesis, or a sub-problem that benefits from its own iterative reasoning, prefer `rlm_query` / `rlm_query_batched` over only one-shot calls.
- Use `llm_query` for lightweight extraction or summarization. Use `rlm_query` for decomposition, deeper analysis, and recursive investigation.
- When you finalize with `FINAL(...)` or `FINAL_VAR(...)`, the returned value must be exactly one of:
  1. the next assistant response text Agent Zero should send, or
  2. the exact JSON tool call object Agent Zero should consume next.
- Do not wrap the final output in markdown fences, and do not mention RLM, recursion, or internal implementation details in the final output.
""".strip()
_BACKEND_ENV_EXPORTS = {
    "anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
    },
    "azure_openai": {
        "api_key": "AZURE_OPENAI_API_KEY",
        "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        "api_version": "AZURE_OPENAI_API_VERSION",
        "azure_deployment": "AZURE_OPENAI_DEPLOYMENT",
    },
    "gemini": {
        "api_key": "GEMINI_API_KEY",
    },
    "openrouter": {
        "api_key": "OPENROUTER_API_KEY",
    },
    "portkey": {
        "api_key": "PORTKEY_API_KEY",
    },
}


@dataclass
class RoutePayload:
    agent: Any
    packed: PackedContext
    root_mapping: ProviderMapping
    subcall_mapping: ProviderMapping | None
    environment: EnvironmentResolution
    plugin_config: dict[str, Any]
    call_kwargs: dict[str, Any]
    finalizer_model: Any | None = None


class RLMChatWrapper:
    def __init__(
        self,
        *,
        agent,
        base_model,
        utility_model=None,
        chat_model_config: dict[str, Any],
        utility_model_config: dict[str, Any],
        plugin_config: dict[str, Any],
        runner=None,
    ) -> None:
        self.agent = agent
        self.base_model = base_model
        self.utility_model = utility_model
        self.chat_model_config = chat_model_config or {}
        self.utility_model_config = utility_model_config or {}
        self.plugin_config = plugin_config or {}
        self.runner = runner or run_routed_completion
        self.a0_model_conf = getattr(base_model, "a0_model_conf", None)

    def __getattr__(self, name: str):
        return getattr(self.base_model, name)

    async def unified_call(self, **kwargs: Any):
        if not self.plugin_config.get("auto_enabled", True):
            return await self.base_model.unified_call(**kwargs)

        messages = list(kwargs.get("messages") or [])
        packed = pack_messages_for_rlm(
            messages,
            {
                **self.plugin_config,
                "ctx_length": int(self.chat_model_config.get("ctx_length", 128000) or 128000),
            },
        )
        if not packed.should_route:
            _log_auto_route_skip(self.agent, packed)
            return await self.base_model.unified_call(**kwargs)

        root_mapping = map_agent_zero_config_to_rlm(
            self.chat_model_config,
            runtime_model=self.base_model,
        )
        if not root_mapping.supported:
            _log_agent_event(
                self.agent,
                heading="RLM auto-routing unavailable",
                content=root_mapping.reason or "The active chat model does not map to a supported RLM backend.",
            )
            return await self.base_model.unified_call(**kwargs)

        subcall_mapping = None
        if str(self.plugin_config.get("subcall_model_source", "utility")) == "utility":
            utility_mapping = map_agent_zero_config_to_rlm(
                self.utility_model_config,
                runtime_model=self.utility_model,
            )
            if utility_mapping.supported:
                subcall_mapping = utility_mapping
            else:
                _log_agent_event(
                    self.agent,
                    heading="RLM subcalls will reuse the root model",
                    content=utility_mapping.reason or "The utility model does not map to a supported RLM backend.",
                )

        environment = resolve_environment(self.plugin_config)
        if not environment.usable:
            _log_agent_event(
                self.agent,
                heading="RLM auto-routing unavailable",
                content=environment.reason or "The configured RLM environment is unavailable.",
            )
            return await self.base_model.unified_call(**kwargs)

        payload = RoutePayload(
            agent=self.agent,
            packed=packed,
            root_mapping=root_mapping,
            subcall_mapping=subcall_mapping,
            environment=environment,
            plugin_config=self.plugin_config,
            call_kwargs=kwargs,
            finalizer_model=_unwrap_finalizer_model(self.base_model),
        )
        _log_agent_event(
            self.agent,
            heading="RLM auto-routing engaged",
            content=(
                f"Offloaded {len(packed.offloaded_blocks)} block(s). "
                f"Estimated prompt pressure {packed.approx_tokens_before} -> {packed.approx_tokens_after} "
                f"tokens using backend {root_mapping.backend} in {environment.environment} mode."
            ),
        )

        try:
            result = self.runner(payload)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as exc:
            _log_agent_event(
                self.agent,
                heading="RLM auto-routing fell back to the base model",
                content=f"{type(exc).__name__}: {exc}",
                event_type="error",
            )
            return await self.base_model.unified_call(**kwargs)


async def run_routed_completion(payload: RoutePayload) -> tuple[str, str]:
    completion, run_record = await _execute_rlm(payload, root_prompt=_default_root_prompt())
    if payload.plugin_config.get("persistence_enabled", True):
        get_run_store(payload.agent).save_run(run_record)
    return completion.get("response", ""), ""


async def run_manual_tool(payload: RoutePayload) -> dict[str, Any]:
    completion, run_record = await _execute_rlm(
        payload,
        root_prompt=payload.call_kwargs.get("question") or _default_root_prompt(),
        allow_limit_recovery=True,
        emit_limit_message=True,
    )
    if payload.plugin_config.get("persistence_enabled", True):
        get_run_store(payload.agent).save_run(run_record)
    return {"response": completion.get("response", ""), "run_record": run_record}


async def _execute_rlm(
    payload: RoutePayload,
    *,
    root_prompt: str,
    allow_limit_recovery: bool = False,
    emit_limit_message: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _prime_backend_env_vars(payload.root_mapping, payload.subcall_mapping)
    RLM = _load_rlm_class()
    logger_cls = _load_rlm_logger_class()

    logger = logger_cls() if logger_cls is not None else None
    root_kwargs = dict(payload.root_mapping.backend_kwargs or {})
    environment_kwargs = dict(payload.environment.environment_kwargs or {})

    rlm_kwargs: dict[str, Any] = {
        "backend": payload.root_mapping.backend,
        "backend_kwargs": root_kwargs,
        "environment": payload.environment.environment,
        "environment_kwargs": environment_kwargs,
        "max_depth": int(payload.plugin_config.get("max_depth", 2) or 2),
        "max_iterations": int(payload.plugin_config.get("max_iterations", 20) or 20),
        "max_errors": int(payload.plugin_config.get("max_errors", 3) or 3),
        "max_concurrent_subcalls": int(payload.plugin_config.get("max_concurrent_subcalls", 4) or 4),
        "custom_system_prompt": _build_agent_zero_system_prompt(),
        "logger": logger,
    }

    if payload.subcall_mapping is not None:
        rlm_kwargs["other_backends"] = [payload.subcall_mapping.backend]
        rlm_kwargs["other_backend_kwargs"] = [payload.subcall_mapping.backend_kwargs]

    if (budget := float(payload.plugin_config.get("max_budget", 0.0) or 0.0)) > 0:
        rlm_kwargs["max_budget"] = budget
    if (timeout := float(payload.plugin_config.get("max_timeout", 0.0) or 0.0)) > 0:
        rlm_kwargs["max_timeout"] = timeout
    if (max_tokens := int(payload.plugin_config.get("max_tokens", 0) or 0)) > 0:
        rlm_kwargs["max_tokens"] = max_tokens

    _attach_callbacks(payload, rlm_kwargs)

    context_payload = {
        "visible_messages": payload.packed.visible_messages,
        "offloaded_blocks": payload.packed.offloaded_blocks,
        "trigger_reason": payload.packed.trigger_reason,
        "prompt_budget": {
            "before": payload.packed.approx_tokens_before,
            "after": payload.packed.approx_tokens_after,
            "threshold": payload.packed.threshold_tokens,
        },
    }

    rlm = None
    try:
        rlm = RLM(**_filter_constructor_kwargs(RLM, rlm_kwargs))
        completion = await asyncio.to_thread(rlm.completion, context_payload, root_prompt)
    except Exception as exc:
        if _should_retry_with_local_environment(payload, exc):
            local_payload = replace(
                payload,
                environment=EnvironmentResolution(
                    environment="local",
                    environment_kwargs={},
                    reason=(
                        "Auto mode retried with local REPL after Docker environment "
                        "startup failed."
                    ),
                    usable=True,
                ),
            )
            return await _execute_rlm(
                local_payload,
                root_prompt=root_prompt,
                allow_limit_recovery=allow_limit_recovery,
                emit_limit_message=emit_limit_message,
            )
        recovered = _recover_limit_result(
            payload,
            root_kwargs=root_kwargs,
            logger=logger,
            rlm=rlm,
            exc=exc,
            allow_limit_recovery=allow_limit_recovery,
            emit_limit_message=emit_limit_message,
        )
        if recovered is not None:
            return recovered
        raise

    response_text, postprocess_reason = await _prepare_agent_zero_response(
        payload,
        root_prompt=root_prompt,
        response_text=getattr(completion, "response", ""),
        trajectory=getattr(completion, "metadata", None) or {},
    )
    run_record = _build_run_record(
        payload,
        root_kwargs=root_kwargs,
        response_text=response_text,
        trajectory=getattr(completion, "metadata", None) or {},
        summary_extra={
            "response_postprocessed": bool(postprocess_reason),
            "response_postprocess_reason": postprocess_reason,
        }
        if postprocess_reason
        else None,
    )
    return {"response": response_text}, run_record


def _load_rlm_class():
    try:
        rlm_module = import_module("rlm")
    except Exception as exc:
        raise RuntimeError("RLM dependency is not installed.") from exc
    _patch_rlm_runtime_safety()
    return rlm_module.RLM


def _load_rlm_logger_class():
    try:
        return import_module("rlm.logger").RLMLogger
    except Exception:
        return None


def _build_agent_zero_system_prompt() -> str:
    try:
        base_prompt = str(import_module("rlm.utils.prompts").RLM_SYSTEM_PROMPT)
    except Exception:
        base_prompt = (
            "Use the REPL to inspect the `context` payload and rely on `rlm_query` for "
            "deeper recursive analysis when a sub-problem needs its own reasoning loop."
        )
    if AGENT_ZERO_RLM_PROMPT_OVERLAY in base_prompt:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{AGENT_ZERO_RLM_PROMPT_OVERLAY}"


def _patch_rlm_runtime_safety() -> None:
    _patch_rlm_openai_client_kwargs()
    _patch_rlm_handler_responses()
    _patch_rlm_direct_completion_paths()


def _prime_backend_env_vars(*mappings: ProviderMapping | None) -> None:
    for mapping in mappings:
        if mapping is None or not mapping.supported:
            continue
        env_exports = _BACKEND_ENV_EXPORTS.get(mapping.backend or "", {})
        backend_kwargs = dict(mapping.backend_kwargs or {})
        for field_name, env_name in env_exports.items():
            if os.environ.get(env_name):
                continue
            value = backend_kwargs.get(field_name)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                os.environ[env_name] = text


def _log_auto_route_skip(agent: Any, packed: PackedContext) -> None:
    if packed.trigger_reason == "below_threshold":
        return

    details = {
        "no_eligible_blocks": "Large prompt pressure was detected, but there were no eligible external blocks to offload.",
        "insufficient_reduction": "External blocks were identified, but offloading them still did not reduce the visible prompt enough to justify RLM routing.",
    }
    content = (
        f"Reason: {packed.trigger_reason}. "
        f"{details.get(packed.trigger_reason, 'RLM auto-routing was skipped.')} "
        f"Estimated prompt pressure stayed at {packed.approx_tokens_after} of {packed.threshold_tokens} tokens "
        f"after examining {len(packed.offloaded_blocks)} candidate block(s)."
    )
    _log_agent_event(agent, heading="RLM auto-routing skipped", content=content)


def _log_agent_event(
    agent: Any,
    *,
    heading: str,
    content: str,
    event_type: str = "util",
) -> None:
    logger = getattr(getattr(agent, "context", None), "log", None)
    if logger is None or not hasattr(logger, "log"):
        return
    try:
        logger.log(type=event_type, heading=heading, content=content)
    except Exception:
        return


def _patch_rlm_openai_client_kwargs() -> None:
    try:
        openai_client_module = import_module("rlm.clients.openai")
    except Exception:
        return

    openai_client_cls = getattr(openai_client_module, "OpenAIClient", None)
    if openai_client_cls is None or getattr(
        openai_client_cls, "_a0_constructor_passthrough_patch_applied", False
    ):
        return

    def patched_init(self, api_key=None, model_name=None, base_url=None, **kwargs):
        openai_client_module.BaseLM.__init__(self, model_name=model_name, **kwargs)

        if api_key is None:
            if base_url == "https://api.openai.com/v1" or base_url is None:
                api_key_value = getattr(openai_client_module, "DEFAULT_OPENAI_API_KEY", None)
            elif base_url == "https://openrouter.ai/api/v1":
                api_key_value = getattr(openai_client_module, "DEFAULT_OPENROUTER_API_KEY", None)
            elif base_url == "https://ai-gateway.vercel.sh/v1":
                api_key_value = getattr(openai_client_module, "DEFAULT_VERCEL_API_KEY", None)
            elif base_url == getattr(
                openai_client_module,
                "DEFAULT_PRIME_INTELLECT_BASE_URL",
                None,
            ):
                api_key_value = getattr(openai_client_module, "DEFAULT_PRIME_API_KEY", None)
            else:
                api_key_value = None
            api_key = api_key_value

        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            **{key: value for key, value in self.kwargs.items() if key != "model_name"},
        }
        self.client = openai_client_module.openai.OpenAI(**client_kwargs)
        self.async_client = openai_client_module.openai.AsyncOpenAI(**client_kwargs)
        self.model_name = model_name
        self.base_url = base_url

        self.model_call_counts = defaultdict(int)
        self.model_input_tokens = defaultdict(int)
        self.model_output_tokens = defaultdict(int)
        self.model_total_tokens = defaultdict(int)
        self.model_costs = defaultdict(float)

    openai_client_cls.__init__ = patched_init
    openai_client_cls._a0_constructor_passthrough_patch_applied = True


def _patch_rlm_handler_responses() -> None:
    try:
        lm_handler_module = import_module("rlm.core.lm_handler")
    except Exception:
        return

    if getattr(lm_handler_module, "_a0_text_response_patch_applied", False):
        return

    original_completion = lm_handler_module.LMHandler.completion
    original_handle_single = lm_handler_module.LMRequestHandler._handle_single
    original_handle_batched = lm_handler_module.LMRequestHandler._handle_batched

    def patched_completion(self, prompt, model=None):
        return _normalize_text_response(original_completion(self, prompt, model))

    def patched_handle_single(self, request, handler):
        response = original_handle_single(self, request, handler)
        _normalize_lm_response_object(response)
        return response

    def patched_handle_batched(self, request, handler):
        response = original_handle_batched(self, request, handler)
        _normalize_lm_response_object(response)
        return response

    lm_handler_module.LMHandler.completion = patched_completion
    lm_handler_module.LMRequestHandler._handle_single = patched_handle_single
    lm_handler_module.LMRequestHandler._handle_batched = patched_handle_batched
    lm_handler_module._a0_text_response_patch_applied = True


def _patch_rlm_direct_completion_paths() -> None:
    try:
        rlm_core_module = import_module("rlm.core.rlm")
    except Exception:
        return

    rlm_class = getattr(rlm_core_module, "RLM", None)
    if rlm_class is None or getattr(rlm_class, "_a0_text_response_patch_applied", False):
        return

    original_fallback_answer = rlm_class._fallback_answer
    original_subcall = rlm_class._subcall

    def patched_fallback_answer(self, message):
        return _normalize_text_response(original_fallback_answer(self, message))

    def patched_subcall(self, prompt, model=None):
        completion = original_subcall(self, prompt, model)
        if completion is not None and hasattr(completion, "response"):
            completion.response = _normalize_text_response(getattr(completion, "response", ""))
        return completion

    rlm_class._fallback_answer = patched_fallback_answer
    rlm_class._subcall = patched_subcall
    rlm_class._a0_text_response_patch_applied = True


def _normalize_lm_response_object(response: Any) -> None:
    if response is None:
        return
    chat_completion = getattr(response, "chat_completion", None)
    if chat_completion is not None and hasattr(chat_completion, "response"):
        chat_completion.response = _normalize_text_response(getattr(chat_completion, "response", ""))
    chat_completions = getattr(response, "chat_completions", None) or []
    for item in chat_completions:
        if item is not None and hasattr(item, "response"):
            item.response = _normalize_text_response(getattr(item, "response", ""))


def _normalize_text_response(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


async def _prepare_agent_zero_response(
    payload: RoutePayload,
    *,
    root_prompt: str,
    response_text: Any,
    trajectory: dict[str, Any] | None,
) -> tuple[str, str | None]:
    normalized = _normalize_text_response(response_text).strip()
    extracted = _extract_wrapped_final_answer(normalized)
    if extracted is not None:
        normalized = extracted

    if not _response_needs_finalization(normalized):
        return normalized, None

    finalizer_model = _unwrap_finalizer_model(payload.finalizer_model)
    if finalizer_model is None or not hasattr(finalizer_model, "unified_call"):
        return normalized, None

    try:
        finalized, _reasoning = await finalizer_model.unified_call(
            system_message=(
                "You are preparing the single next assistant message for Agent Zero. "
                "Convert internal RLM working notes into the final user-facing answer. "
                "Output only the final assistant text or the exact JSON tool call object. "
                "Do not include REPL code, llm_query/rlm_query calls, or internal reasoning."
            ),
            user_message=_build_finalizer_prompt(
                payload,
                root_prompt=root_prompt,
                raw_response=normalized,
                trajectory=trajectory or {},
            ),
        )
    except Exception as exc:
        _log_agent_event(
            payload.agent,
            heading="RLM final answer cleanup skipped",
            content=f"{type(exc).__name__}: {exc}",
            event_type="error",
        )
        return normalized, None

    finalized_text = _normalize_text_response(finalized).strip()
    extracted = _extract_wrapped_final_answer(finalized_text)
    if extracted is not None:
        finalized_text = extracted
    if not finalized_text:
        return normalized, None

    _log_agent_event(
        payload.agent,
        heading="RLM final answer cleaned up",
        content="Agent Zero converted an internal RLM draft into a final assistant response.",
    )
    return finalized_text, "finalizer_model"


def _extract_wrapped_final_answer(text: str) -> str | None:
    if not text:
        return None

    final_match = re.search(r"^\s*FINAL\((.*)\)\s*$", text, re.MULTILINE | re.DOTALL)
    if final_match:
        return final_match.group(1).strip()

    final_var_match = re.search(r"^\s*FINAL_VAR\((.*)\)\s*$", text, re.MULTILINE | re.DOTALL)
    if final_var_match:
        return None

    return None


def _response_needs_finalization(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _is_exact_json_object(stripped):
        return False

    lowered = stripped.lower()
    if "```repl" in lowered:
        return True

    internal_markers = (
        "llm_query(",
        "llm_query_batched(",
        "rlm_query(",
        "rlm_query_batched(",
        "final(",
        "final_var(",
        "show_vars(",
    )
    return any(marker in lowered for marker in internal_markers)


def _is_exact_json_object(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    return isinstance(parsed, dict)


def _unwrap_finalizer_model(model: Any | None) -> Any | None:
    if model is None:
        return None
    base_model = getattr(model, "base_model", None)
    return base_model or model


def _build_finalizer_prompt(
    payload: RoutePayload,
    *,
    root_prompt: str,
    raw_response: str,
    trajectory: dict[str, Any],
) -> str:
    prompt_data = {
        "root_prompt": root_prompt,
        "visible_messages": payload.packed.visible_messages[-6:],
        "offloaded_block_ids": [
            str(block.get("id") or "")
            for block in payload.packed.offloaded_blocks[:20]
            if isinstance(block, dict)
        ],
        "offloaded_block_count": len(payload.packed.offloaded_blocks),
        "raw_rlm_response": raw_response,
        "trajectory_digest": _build_trajectory_digest(trajectory),
    }
    return (
        "The RLM run returned internal working output instead of a final answer.\n\n"
        "Use the information below to produce the single next assistant message Agent Zero "
        "should send. If the correct output is an exact JSON tool call object, return only "
        "that JSON object. Otherwise return concise assistant text only.\n\n"
        f"{json.dumps(prompt_data, ensure_ascii=False, indent=2)}"
    )


def _build_trajectory_digest(trajectory: dict[str, Any], max_entries: int = 18) -> list[dict[str, Any]]:
    iterations = list((trajectory or {}).get("iterations") or [])
    digest: list[dict[str, Any]] = []
    for iteration_index, iteration in enumerate(iterations[-4:], start=max(len(iterations) - 3, 1)):
        if len(digest) >= max_entries:
            break
        response = _truncate_digest_text(iteration.get("response"))
        if response:
            digest.append(
                {
                    "iteration": iteration.get("iteration", iteration_index),
                    "type": "response",
                    "content": response,
                }
            )

        for code_block in iteration.get("code_blocks") or []:
            if len(digest) >= max_entries:
                break
            result = dict(code_block.get("result") or {})
            stdout = _truncate_digest_text(result.get("stdout"))
            stderr = _truncate_digest_text(result.get("stderr"))
            if stdout:
                digest.append(
                    {
                        "iteration": iteration.get("iteration", iteration_index),
                        "type": "stdout",
                        "content": stdout,
                    }
                )
            if stderr and len(digest) < max_entries:
                digest.append(
                    {
                        "iteration": iteration.get("iteration", iteration_index),
                        "type": "stderr",
                        "content": stderr,
                    }
                )
            for call in result.get("rlm_calls") or []:
                if len(digest) >= max_entries:
                    break
                call_response = _truncate_digest_text(call.get("response"))
                if call_response:
                    digest.append(
                        {
                            "iteration": iteration.get("iteration", iteration_index),
                            "type": "subcall_response",
                            "content": call_response,
                        }
                    )
    return digest


def _truncate_digest_text(value: Any, limit: int = 1200) -> str:
    text = _normalize_text_response(value).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} chars truncated]"


def _recover_limit_result(
    payload: RoutePayload,
    *,
    root_kwargs: dict[str, Any],
    logger: Any,
    rlm: Any,
    exc: Exception,
    allow_limit_recovery: bool,
    emit_limit_message: bool,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not allow_limit_recovery:
        return None

    status = _classify_rlm_limit_exception(exc)
    if status is None:
        return None

    partial_answer = _normalize_text_response(getattr(exc, "partial_answer", None))
    if not partial_answer and rlm is not None:
        partial_answer = _normalize_text_response(getattr(rlm, "_best_partial_answer", None))

    response_text = partial_answer
    if not response_text and emit_limit_message:
        response_text = _format_limit_message(status, exc)
    if not response_text:
        return None

    run_record = _build_run_record(
        payload,
        root_kwargs=root_kwargs,
        response_text=response_text,
        trajectory=logger.get_trajectory() if logger is not None else {},
        summary_extra={
            "status": status,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "partial_response": bool(partial_answer),
        },
    )
    return {"response": response_text}, run_record


def _build_run_record(
    payload: RoutePayload,
    *,
    root_kwargs: dict[str, Any],
    response_text: str,
    trajectory: dict[str, Any] | None,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = str(uuid4())
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    summary = {
        "trigger_reason": payload.packed.trigger_reason,
        "root_backend": payload.root_mapping.backend,
        "root_model": root_kwargs.get("model_name"),
        "subcall_backend": payload.subcall_mapping.backend if payload.subcall_mapping else None,
        "environment": payload.environment.environment,
        "response_preview": response_text[:500],
        "offloaded_block_count": len(payload.packed.offloaded_blocks),
        "approx_tokens_before": payload.packed.approx_tokens_before,
        "approx_tokens_after": payload.packed.approx_tokens_after,
    }
    if summary_extra:
        summary.update(summary_extra)
    return {
        "run_id": run_id,
        "started_at": started_at,
        "summary": summary,
        "trajectory": trajectory or {},
    }


def _classify_rlm_limit_exception(exc: Exception) -> str | None:
    match type(exc).__name__:
        case "TimeoutExceededError":
            return "timeout_exceeded"
        case "TokenLimitExceededError":
            return "token_limit_exceeded"
        case "BudgetExceededError":
            return "budget_exceeded"
        case "ErrorThresholdExceededError":
            return "error_threshold_exceeded"
        case "CancellationError":
            return "cancelled"
        case _:
            return None


def _format_limit_message(status: str, exc: Exception) -> str:
    labels = {
        "timeout_exceeded": "RLM stopped because it hit the configured timeout.",
        "token_limit_exceeded": "RLM stopped because it hit the configured token limit.",
        "budget_exceeded": "RLM stopped because it hit the configured budget limit.",
        "error_threshold_exceeded": "RLM stopped because it hit the configured error limit.",
        "cancelled": "RLM was cancelled before it completed.",
    }
    prefix = labels.get(status, "RLM stopped before it completed.")
    detail = str(exc).strip()
    return f"{prefix} {detail}".strip()


def _should_retry_with_local_environment(payload: RoutePayload, exc: Exception) -> bool:
    if payload.environment.environment != "docker":
        return False
    if str(payload.plugin_config.get("environment_mode", "auto") or "auto").lower() != "auto":
        return False
    message = str(exc or "").lower()
    if not message:
        return False
    docker_failure_markers = (
        "failed to start container",
        "docker",
        "cpuset",
        "read-only file system",
        "permission denied",
        "cannot connect to the docker daemon",
    )
    return any(marker in message for marker in docker_failure_markers)


def _attach_callbacks(payload: RoutePayload, rlm_kwargs: dict[str, Any]) -> None:
    agent = payload.agent
    if agent is None or getattr(agent, "context", None) is None:
        return
    logger = getattr(agent.context, "log", None)
    if logger is None or not hasattr(logger, "log"):
        return

    def on_iteration_start(depth: int, iteration_num: int) -> None:
        logger.log(
            type="util",
            heading=f"RLM iteration {iteration_num} started",
            content=f"Depth {depth}",
        )

    def on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
        logger.log(
            type="util",
            heading=f"RLM subcall depth {depth}",
            content=f"{model}: {prompt_preview}",
        )

    rlm_kwargs["on_iteration_start"] = on_iteration_start
    rlm_kwargs["on_subcall_start"] = on_subcall_start


def _default_root_prompt() -> str:
    return (
        "Generate the next Agent Zero assistant message from the visible conversation and "
        "the offloaded external context blocks. Output only the final assistant text or the "
        "exact JSON tool call object that Agent Zero should consume next."
    )


def _filter_constructor_kwargs(rlm_class: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(rlm_class)
    except (TypeError, ValueError):
        return kwargs

    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs

    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in kwargs.items() if key in accepted}
