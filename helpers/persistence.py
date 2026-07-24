from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID


class RunStore:
    def __init__(self, root: str | Path, retention_count: int = 25):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_count = retention_count

    def save_run(self, record: dict[str, Any]) -> dict[str, Any]:
        run_id = _canonical_run_id(record["run_id"])
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_payload = {
            "run_id": run_id,
            "started_at": record.get("started_at"),
            **dict(record.get("summary", {}) or {}),
        }

        (run_dir / "summary.json").write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "trajectory.json").write_text(
            json.dumps(record.get("trajectory", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self.prune_runs()
        return record

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for summary_path in self.root.glob("*/summary.json"):
            try:
                run_id = _canonical_run_id(summary_path.parent.name)
            except (TypeError, ValueError):
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(summary, dict):
                continue
            runs.append({**summary, "run_id": run_id})
        runs.sort(key=lambda item: item.get("started_at", ""), reverse=True)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            record_path = self._run_dir(run_id) / "record.json"
        except (TypeError, ValueError):
            return None
        if not record_path.exists():
            return None
        try:
            return json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def prune_runs(self, keep: int | None = None) -> int:
        keep_count = self.retention_count if keep is None else max(0, int(keep))
        runs = self.list_runs()
        removed = 0
        for stale in runs[keep_count:]:
            try:
                run_dir = self._run_dir(stale["run_id"])
            except (TypeError, ValueError):
                continue
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
                removed += 1
        return removed

    def _run_dir(self, run_id: Any) -> Path:
        canonical = _canonical_run_id(run_id)
        run_dir = (self.root / canonical).resolve()
        if not run_dir.is_relative_to(self.root):
            raise ValueError("Run ID resolves outside the run store.")
        return run_dir


def _canonical_run_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Run ID is required.")
    try:
        parsed = UUID(text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Run ID must be a UUID.") from exc
    canonical = str(parsed)
    if text.lower() != canonical:
        raise ValueError("Run ID must use canonical UUID form.")
    return canonical
