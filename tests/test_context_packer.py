from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import support  # noqa: F401

from usr.plugins.rlm.helpers.context_packer import pack_messages_for_rlm


class Message:
    def __init__(self, content, message_type: str = "human"):
        self.content = content
        self.type = message_type


class ContextPackerTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ctx_length": 200,
            "trigger_threshold_pct": 0.5,
            "min_block_chars": 10,
            "attachment_max_chars": 1000,
        }

    def test_reads_large_text_attachment_inside_allowed_upload_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_root = Path(temp_dir) / "uploads"
            upload_root.mkdir()
            attachment = upload_root / "notes.txt"
            attachment.write_text("important evidence " * 20, encoding="utf-8")

            packed = pack_messages_for_rlm(
                [Message({"user_message": "review", "attachments": [str(attachment)]})],
                self.config,
                attachment_roots=[upload_root],
            )

        self.assertTrue(packed.should_route)
        self.assertEqual(len(packed.offloaded_blocks), 1)
        self.assertEqual(packed.offloaded_blocks[0]["source"]["kind"], "attachment")

    def test_rejects_attachment_outside_allowed_upload_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            upload_root = base / "uploads"
            upload_root.mkdir()
            outside = base / "private.txt"
            outside.write_text("private material " * 20, encoding="utf-8")
            content = json.dumps(
                {"user_message": "review", "attachments": [str(outside)]}
            )

            packed = pack_messages_for_rlm(
                [Message(content)],
                self.config,
                attachment_roots=[upload_root],
            )

        self.assertFalse(packed.should_route)
        self.assertEqual(packed.offloaded_blocks, [])

    def test_rejects_symlink_escape_from_upload_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            upload_root = base / "uploads"
            upload_root.mkdir()
            outside = base / "private.txt"
            outside.write_text("private material " * 20, encoding="utf-8")
            link = upload_root / "linked.txt"
            link.symlink_to(outside)

            packed = pack_messages_for_rlm(
                [Message({"user_message": "review", "attachments": [str(link)]})],
                self.config,
                attachment_roots=[upload_root],
            )

        self.assertFalse(packed.should_route)
        self.assertEqual(packed.offloaded_blocks, [])

    def test_offloads_large_non_assistant_message_field(self):
        packed = pack_messages_for_rlm(
            [Message({"tool_result": "large result " * 30})],
            self.config,
            attachment_roots=[],
        )

        self.assertTrue(packed.should_route)
        self.assertEqual(
            packed.offloaded_blocks[0]["source"]["kind"],
            "message_field",
        )
        self.assertIn("RLM_OFFLOADED", packed.visible_messages[0]["content"]["tool_result"])


if __name__ == "__main__":
    unittest.main()
