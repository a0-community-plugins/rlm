from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, root: str | Path, retention_count: int = 25):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_count = retention_count

    def save_run(self, record: dict[str, Any]) -> dict[str, Any]:
        run_id = str(record["run_id"])
        run_dir = self.root / run_id
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
                runs.append(json.loads(summary_path.read_text(encoding="utf-8")))
            except Exception:
                continue
        runs.sort(key=lambda item: item.get("started_at", ""), reverse=True)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        record_path = self.root / run_id / "record.json"
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
            run_dir = self.root / stale["run_id"]
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
                removed += 1
        return removed

