from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from checkpoint_core import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    COMPLETE_MARKER,
    METADATA_FILE,
    checkpoint_signature,
    find_best_checkpoint,
    find_latest_checkpoint,
    remaining_epoch_items,
    shuffled_epoch_items,
)


class CheckpointCoreTests(unittest.TestCase):
    def test_signature_is_stable_and_detects_changes(self) -> None:
        first = checkpoint_signature({"seed": 42, "batch": 4})
        reordered = checkpoint_signature({"batch": 4, "seed": 42})
        changed = checkpoint_signature({"seed": 42, "batch": 8})
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)

    def test_epoch_order_and_resume_slice_are_deterministic(self) -> None:
        first = shuffled_epoch_items(list(range(12)), seed=99)
        second = shuffled_epoch_items(list(range(12)), seed=99)
        self.assertEqual(first, second)
        self.assertEqual(remaining_epoch_items(first, 4, 2), first[8:])

    def test_latest_ignores_partial_and_prefers_epoch_boundary(self) -> None:
        signature = checkpoint_signature({"run": 1})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root / "checkpoint-step-10", signature, 10, 0, 80)
            self._write(
                root / "checkpoint-step-11-partial",
                signature,
                11,
                0,
                88,
                complete=False,
            )
            expected = root / "checkpoint-step-10-boundary"
            self._write(expected, signature, 10, 1, 0)
            self.assertEqual(find_latest_checkpoint(root, signature), expected)

    def test_incompatible_latest_checkpoint_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root / "checkpoint-other", "other", 3, 0, 24)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                find_latest_checkpoint(root, "expected")

    def test_best_checkpoint_ignores_partial_and_prefers_metric(self) -> None:
        signature = checkpoint_signature({"run": 2})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lower = root / "best-checkpoint-10"
            higher = root / "best-checkpoint-20"
            partial = root / "best-checkpoint-30-partial"
            self._write_best(lower, signature, 10, 0.81)
            self._write_best(higher, signature, 20, 0.84)
            self._write_best(partial, signature, 30, 0.90, complete=False)
            self.assertEqual(find_best_checkpoint(root, signature), higher)

    def test_incompatible_best_checkpoint_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_best(root / "best-checkpoint-other", "other", 3, 0.9)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                find_best_checkpoint(root, "expected")

    @staticmethod
    def _write(
        path: Path,
        signature: str,
        step: int,
        epoch: int,
        batch: int,
        complete: bool = True,
    ) -> None:
        path.mkdir()
        (path / METADATA_FILE).write_text(
            json.dumps(
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "signature": signature,
                    "global_step": step,
                    "epoch_index": epoch,
                    "next_batch_index": batch,
                    "created_at_utc": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        if complete:
            (path / COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")

    @staticmethod
    def _write_best(
        path: Path,
        signature: str,
        step: int,
        metric: float,
        complete: bool = True,
    ) -> None:
        path.mkdir()
        (path / METADATA_FILE).write_text(
            json.dumps(
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "signature": signature,
                    "global_step": step,
                    "selection_metric_value": metric,
                    "created_at_utc": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        if complete:
            (path / COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
