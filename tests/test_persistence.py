from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import support  # noqa: F401

from usr.plugins.rlm.helpers.persistence import RunStore


def record(run_id: str, offset_seconds: int = 0) -> dict:
    started = datetime.now(UTC) + timedelta(seconds=offset_seconds)
    return {
        "run_id": run_id,
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "summary": {"status": "completed"},
        "trajectory": {"iterations": []},
    }


class PersistenceTests(unittest.TestCase):
    def test_round_trip_and_prune_use_uuid_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RunStore(temp_dir, retention_count=1)
            older = str(uuid4())
            newer = str(uuid4())
            store.save_run(record(older, offset_seconds=-10))
            store.save_run(record(newer))

            self.assertIsNone(store.get_run(older))
            self.assertEqual(store.get_run(newer)["run_id"], newer)
            self.assertEqual([item["run_id"] for item in store.list_runs()], [newer])

    def test_get_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RunStore(temp_dir)
            self.assertIsNone(store.get_run("../../outside"))
            self.assertIsNone(store.get_run(""))

    def test_save_rejects_noncanonical_run_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RunStore(temp_dir)
            with self.assertRaises(ValueError):
                store.save_run(record("../../outside"))

    def test_list_ignores_non_uuid_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            invalid = root / "not-a-run"
            invalid.mkdir()
            (invalid / "summary.json").write_text(
                '{"run_id":"../../outside","started_at":"9999"}',
                encoding="utf-8",
            )
            store = RunStore(root)

            self.assertEqual(store.list_runs(), [])


if __name__ == "__main__":
    unittest.main()
