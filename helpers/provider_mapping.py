from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderMapping:
    supported: bool
    backend: str | None = None
    backend_kwargs: dict[str, Any] | None = None
    reason: str = ""


_NATIVE_PROVIDER_MAP = {
    "openai": "openai",
    "openrouter": "openrouter",
    "anthropic": "anthropic",
    "azure": "azure_openai",
    "google": "gemini",
    "gemini": "gemini",
    "portkey": "portkey",
}

_PROVIDER_CANONICAL_ALIASES = {
    "anthropic_oauth": "anthropic",
}

_PROVIDER_API_KEY_ALIASES = {
    "anthropic": ("anthropic_oauth",),
    "anthropic_oauth": ("anthropic",),
    "azure": ("azure_openai",),
    "azure_openai": ("azure",),
    "gemini": ("google",),
    "google": ("gemini",),
}

_PLACEHOLDER_SECRET_VALUES = {"", "na", "none", "null"}
_OPENAI_STYLE_BACKENDS = {"openai", "openrouter", "vercel", "vllm"}
_OPENAI_STYLE_CLIENT_KEYS = {
    "timeout",
    "default_headers",
    "default_query",
    "max_retries",
    "organization",
    "project",
    "websocket_base_url",
    "http_client",
}
_ANTHROPIC_CLIENT_KEYS = {"timeout", "max_tokens"}
_AZURE_CLIENT_KEYS = {
    "timeout",
    "max_retries",
    "default_headers",
    "default_query",
    "http_client",
}
_GEMINI_CLIENT_KEYS = {"timeout"}
_PORTKEY_CLIENT_KEYS = {"timeout"}


def map_agent_zero_config_to_rlm(
    config: dict[str, Any] | None,
    *,
    runtime_model: Any | None = None,
) -> ProviderMapping:
    if not config and runtime_model is None:
        return ProviderMapping(False, reason="Missing model configuration.")

    config = dict(config or {})
    runtime = _extract_runtime_model_details(runtime_model)

    provider = _canonicalize_provider(
        str(
        config.get("provider")
        or runtime.get("config_provider")
        or runtime.get("provider")
        or ""
        ).strip().lower()
    )
    provider_meta = _get_chat_provider_metadata(provider)
    litellm_provider = _canonicalize_provider(
        str(
            (provider_meta or {}).get("litellm_provider")
            or runtime.get("provider")
            or provider
            or ""
        ).strip().lower()
    )
    model_name = str(
        config.get("name")
        or runtime.get("model_name")
        or ""
    ).strip()
    kwargs = {
        **dict(runtime.get("kwargs", {}) or {}),
        **dict(config.get("kwargs", {}) or {}),
    }
    api_key = _clean_secret_value(
        config.get("api_key")
        or kwargs.get("api_key")
        or _resolve_provider_api_key(provider)
    )
    api_base = str(
        config.get("api_base")
        or kwargs.get("api_base")
        or kwargs.get("base_url")
        or runtime.get("api_base")
        or ""
    ).strip()

    if not provider or not model_name:
        return ProviderMapping(False, reason="Model configuration is incomplete.")

    backend = _resolve_backend(provider, litellm_provider, api_base)
    if backend == "azure_openai":
        azure_endpoint = api_base or kwargs.get("azure_endpoint") or kwargs.get("api_base")
        if not azure_endpoint:
            return ProviderMapping(
                False,
                reason="Unsupported Azure configuration without an endpoint.",
            )
        backend_kwargs = {
            "model_name": model_name,
            "api_key": api_key or None,
            "azure_endpoint": azure_endpoint,
        }
        if api_version := kwargs.get("api_version"):
            backend_kwargs["api_version"] = api_version
        if deployment := kwargs.get("azure_deployment"):
            backend_kwargs["azure_deployment"] = deployment
        backend_kwargs.update(_normalize_backend_kwargs(backend, kwargs))
        return ProviderMapping(True, backend=backend, backend_kwargs=backend_kwargs)

    if backend is not None:
        backend_kwargs = _build_backend_kwargs(
            backend=backend,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            kwargs=kwargs,
        )
        return ProviderMapping(True, backend=backend, backend_kwargs=backend_kwargs)

    compat_base = api_base or kwargs.get("api_base")
    if compat_base:
        return ProviderMapping(
            True,
            backend="openai",
            backend_kwargs=_build_backend_kwargs(
                backend="openai",
                model_name=model_name,
                api_key=api_key,
                api_base=compat_base,
                kwargs=kwargs,
            ),
            reason="Using OpenAI-compatible fallback mapping.",
        )

    return ProviderMapping(
        False,
        reason=f"Unsupported provider '{provider}' without an OpenAI-compatible api_base.",
    )


def _extract_runtime_model_details(runtime_model: Any | None) -> dict[str, Any]:
    if runtime_model is None:
        return {}

    runtime_kwargs = dict(getattr(runtime_model, "kwargs", {}) or {})
    runtime_provider = _canonicalize_provider(
        str(getattr(runtime_model, "provider", "") or "").strip().lower()
    )
    runtime_model_name = _normalize_runtime_model_name(
        runtime_provider,
        str(getattr(runtime_model, "model_name", "") or "").strip(),
    )

    model_conf = getattr(runtime_model, "a0_model_conf", None)
    config_provider = ""
    if model_conf is not None:
        config_provider = _canonicalize_provider(
            str(getattr(model_conf, "provider", "") or "").strip().lower()
        )
        runtime_provider = runtime_provider or _canonicalize_provider(
            config_provider
        )
        runtime_model_name = runtime_model_name or str(getattr(model_conf, "name", "") or "").strip()
        runtime_kwargs = {
            **dict(getattr(model_conf, "kwargs", {}) or {}),
            **runtime_kwargs,
        }
        if api_key := _clean_secret_value(getattr(model_conf, "api_key", "") or ""):
            runtime_kwargs.setdefault("api_key", api_key)
        if api_base := str(getattr(model_conf, "api_base", "") or "").strip():
            runtime_kwargs.setdefault("api_base", api_base)

    return {
        "provider": runtime_provider,
        "config_provider": config_provider,
        "model_name": runtime_model_name,
        "kwargs": runtime_kwargs,
        "api_base": runtime_kwargs.get("api_base") or runtime_kwargs.get("base_url") or "",
    }


def _normalize_runtime_model_name(provider: str, model_name: str) -> str:
    if not provider or not model_name:
        return model_name
    prefix = f"{provider}/"
    if model_name.startswith(prefix):
        return model_name[len(prefix):]
    return model_name


def _resolve_provider_api_key(provider: str) -> str:
    if not provider:
        return ""
    try:
        import models

        for candidate in _provider_key_candidates(provider):
            key = _clean_secret_value(models.get_api_key(candidate))
            if key:
                return key
    except Exception:
        return ""
    return ""


def _canonicalize_provider(provider: str) -> str:
    return _PROVIDER_CANONICAL_ALIASES.get(provider, provider)


def _resolve_backend(provider: str, litellm_provider: str, api_base: str) -> str | None:
    direct = _NATIVE_PROVIDER_MAP.get(provider)
    if direct == "openai" and provider != "openai" and not api_base:
        direct = None
    if direct is not None:
        return direct

    translated = _NATIVE_PROVIDER_MAP.get(litellm_provider)
    if translated == "openai" and provider != "openai" and not api_base:
        return None
    return translated


def _build_backend_kwargs(
    *,
    backend: str,
    model_name: str,
    api_key: str,
    api_base: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    backend_kwargs: dict[str, Any] = {"model_name": model_name}
    if api_key:
        backend_kwargs["api_key"] = api_key
    if api_base and backend != "azure_openai":
        backend_kwargs["base_url"] = api_base
    backend_kwargs.update(_normalize_backend_kwargs(backend, kwargs))
    return backend_kwargs


def _normalize_backend_kwargs(backend: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    allowed = _allowed_backend_client_keys(backend)
    for key, value in kwargs.items():
        normalized_key = _normalize_backend_key(backend, key)
        if normalized_key in {"api_key", "api_base", "azure_endpoint", "azure_deployment", "model"}:
            continue
        if normalized_key not in allowed:
            continue
        if normalized_key == "default_headers":
            merged = {
                **dict(normalized.get("default_headers", {}) or {}),
                **dict(value or {}),
            }
            if merged:
                normalized["default_headers"] = merged
            continue
        normalized[normalized_key] = value
    return normalized


def _allowed_backend_client_keys(backend: str) -> set[str]:
    if backend in _OPENAI_STYLE_BACKENDS:
        return _OPENAI_STYLE_CLIENT_KEYS
    if backend == "anthropic":
        return _ANTHROPIC_CLIENT_KEYS
    if backend == "azure_openai":
        return _AZURE_CLIENT_KEYS
    if backend == "gemini":
        return _GEMINI_CLIENT_KEYS
    if backend == "portkey":
        return _PORTKEY_CLIENT_KEYS
    return set()


def _normalize_backend_key(backend: str, key: str) -> str:
    if key == "extra_headers" and backend in _OPENAI_STYLE_BACKENDS | {"azure_openai"}:
        return "default_headers"
    return key


def _get_chat_provider_metadata(provider: str) -> dict[str, Any]:
    if not provider:
        return {}
    try:
        from helpers.providers import get_provider_config

        return dict(get_provider_config("chat", provider) or {})
    except Exception:
        return {}


def _provider_key_candidates(provider: str) -> tuple[str, ...]:
    canonical = _canonicalize_provider(provider)
    aliases = _PROVIDER_API_KEY_ALIASES.get(canonical, ())
    ordered = [canonical, *aliases]
    seen: list[str] = []
    for candidate in ordered:
        if candidate and candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def _clean_secret_value(value: Any) -> str:
    cleaned = str(value or "").strip()
    if cleaned.lower() in _PLACEHOLDER_SECRET_VALUES:
        return ""
    return cleaned
