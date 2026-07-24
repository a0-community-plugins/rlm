from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_UPLOAD_ROOT = PROJECT_ROOT / "usr" / "uploads"


@dataclass
class PackedContext:
    should_route: bool
    trigger_reason: str
    visible_messages: list[dict[str, Any]]
    offloaded_blocks: list[dict[str, Any]]
    approx_tokens_before: int
    approx_tokens_after: int
    threshold_tokens: int


def pack_messages_for_rlm(
    messages: list[Any],
    config: dict[str, Any] | None,
    *,
    attachment_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> PackedContext:
    config = config or {}
    min_block_chars = int(config.get("min_block_chars", 4000) or 4000)
    trigger_threshold_pct = float(config.get("trigger_threshold_pct", 0.8) or 0.8)
    ctx_length = int(config.get("ctx_length", 128000) or 128000)
    attachment_max_chars = int(config.get("attachment_max_chars", 2_000_000) or 2_000_000)
    threshold_tokens = int(ctx_length * trigger_threshold_pct)
    allowed_attachment_roots = _normalize_attachment_roots(
        attachment_roots if attachment_roots is not None else [DEFAULT_UPLOAD_ROOT]
    )

    serialized = [_serialize_message(message) for message in messages]
    approx_tokens_before = _estimate_tokens(serialized)

    offloaded_blocks: list[dict[str, Any]] = []
    visible_messages: list[dict[str, Any]] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"block-{counter}"

    for index, message in enumerate(serialized):
        content, blocks = _rewrite_content(
            message.get("content"),
            next_id=next_id,
            min_block_chars=min_block_chars,
            attachment_max_chars=attachment_max_chars,
            message_index=index,
            role=message.get("role", "user"),
            attachment_roots=allowed_attachment_roots,
        )
        visible_messages.append({**message, "content": content})
        offloaded_blocks.extend(blocks)

    attachment_tokens = sum(
        _estimate_tokens(block.get("content", ""))
        for block in offloaded_blocks
        if block.get("source", {}).get("kind") == "attachment"
    )
    approx_tokens_before += attachment_tokens
    approx_tokens_after = _estimate_tokens(visible_messages)
    threshold_reached = approx_tokens_before >= threshold_tokens
    reduction_succeeded = approx_tokens_after < threshold_tokens
    should_route = threshold_reached and bool(offloaded_blocks) and reduction_succeeded

    if should_route:
        trigger_reason = "oversized_external_context"
    elif not threshold_reached:
        trigger_reason = "below_threshold"
    elif not offloaded_blocks:
        trigger_reason = "no_eligible_blocks"
    else:
        trigger_reason = "insufficient_reduction"

    return PackedContext(
        should_route=should_route,
        trigger_reason=trigger_reason,
        visible_messages=visible_messages,
        offloaded_blocks=offloaded_blocks,
        approx_tokens_before=approx_tokens_before,
        approx_tokens_after=approx_tokens_after,
        threshold_tokens=threshold_tokens,
    )


def _serialize_message(message: Any) -> dict[str, Any]:
    role = "user"
    message_type = getattr(message, "type", "")
    if message_type == "system":
        role = "system"
    elif message_type == "ai":
        role = "assistant"

    return {
        "role": role,
        "content": _decode_structured_content(getattr(message, "content", "")),
    }


def _rewrite_content(
    content: Any,
    *,
    next_id,
    min_block_chars: int,
    attachment_max_chars: int,
    message_index: int,
    role: str,
    attachment_roots: tuple[Path, ...],
    path: list[str] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    path = path or []
    blocks: list[dict[str, Any]] = []

    if isinstance(content, dict):
        rewritten: dict[str, Any] = {}
        for key, value in content.items():
            if key == "attachments" and isinstance(value, list):
                rewritten[key] = list(value)
                for attachment in value:
                    block = _attachment_block(
                        attachment,
                        next_id=next_id,
                        min_block_chars=min_block_chars,
                        attachment_max_chars=attachment_max_chars,
                        message_index=message_index,
                        attachment_roots=attachment_roots,
                    )
                    if block is not None:
                        blocks.append(block)
                continue

            if role != "assistant" and _is_large_candidate(value, min_block_chars):
                block_id = next_id()
                blocks.append(
                    {
                        "id": block_id,
                        "source": {
                            "kind": "message_field",
                            "message_index": message_index,
                            "field": key,
                            "path": path + [key],
                        },
                        "content": _stringify(value),
                    }
                )
                rewritten[key] = _placeholder(block_id, "message_field", key)
                continue

            rewritten_value, nested_blocks = _rewrite_content(
                value,
                next_id=next_id,
                min_block_chars=min_block_chars,
                attachment_max_chars=attachment_max_chars,
                message_index=message_index,
                role=role,
                attachment_roots=attachment_roots,
                path=path + [key],
            )
            rewritten[key] = rewritten_value
            blocks.extend(nested_blocks)
        return rewritten, blocks

    if isinstance(content, list):
        rewritten_list = []
        for idx, item in enumerate(content):
            rewritten_item, nested_blocks = _rewrite_content(
                item,
                next_id=next_id,
                min_block_chars=min_block_chars,
                attachment_max_chars=attachment_max_chars,
                message_index=message_index,
                role=role,
                attachment_roots=attachment_roots,
                path=path + [str(idx)],
            )
            rewritten_list.append(rewritten_item)
            blocks.extend(nested_blocks)
        return rewritten_list, blocks

    return content, blocks


def _attachment_block(
    attachment: Any,
    *,
    next_id,
    min_block_chars: int,
    attachment_max_chars: int,
    message_index: int,
    attachment_roots: tuple[Path, ...],
) -> dict[str, Any] | None:
    if not isinstance(attachment, str) or not attachment:
        return None

    path = _resolve_allowed_attachment(attachment, attachment_roots)
    if path is None:
        return None

    try:
        raw = path.read_bytes()
    except Exception:
        return None

    if b"\x00" in raw:
        return None

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:
            return None

    text = text[:attachment_max_chars]
    if len(text) < min_block_chars:
        return None

    block_id = next_id()
    return {
        "id": block_id,
        "source": {
            "kind": "attachment",
            "message_index": message_index,
            "path": str(path),
        },
        "content": text,
    }


def _normalize_attachment_roots(
    roots: list[str | Path] | tuple[str | Path, ...],
) -> tuple[Path, ...]:
    normalized: list[Path] = []
    for root in roots:
        try:
            normalized.append(Path(root).expanduser().resolve())
        except (OSError, RuntimeError):
            continue
    return tuple(dict.fromkeys(normalized))


def _resolve_allowed_attachment(
    attachment: str,
    roots: tuple[Path, ...],
) -> Path | None:
    raw_path = Path(attachment).expanduser()
    candidates = [raw_path]
    if attachment.startswith("/a0/"):
        candidates.append(PROJECT_ROOT / attachment.removeprefix("/a0/"))
    elif not raw_path.is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)

    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            continue
        if not resolved.is_file():
            continue
        if any(resolved.is_relative_to(root) for root in roots):
            return resolved
    return None


def _is_large_candidate(value: Any, min_block_chars: int) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list)):
        return len(_stringify(value)) >= min_block_chars
    if isinstance(value, str):
        return len(value) >= min_block_chars
    return False


def _placeholder(block_id: str, kind: str, label: str) -> str:
    return f"[RLM_OFFLOADED:{block_id} kind={kind} field={label}]"


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _estimate_tokens(value: Any) -> int:
    text = _stringify(value)
    return max(1, len(text) // 2)


def _decode_structured_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content

    stripped = content.strip()
    if not stripped or stripped[0] not in "[{":
        return content

    try:
        return json.loads(stripped)
    except Exception:
        return content
