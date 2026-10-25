"""Lightweight checkpoint metadata and deterministic epoch helpers.

This module deliberately has no PyTorch, Transformers, or PEFT imports so its
failure-handling rules can be unit-tested without initializing the ML stack.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


CHECKPOINT_FORMAT_VERSION = 1
COMPLETE_MARKER = "COMPLETE"
METADATA_FILE = "metadata.json"


def checkpoint_candidate_sort_key(
    candidate: tuple[tuple[int, int, int, str], Path, dict[str, Any]],
) -> tuple[int, int, int, str]:
    return candidate[0]


def best_checkpoint_candidate_sort_key(
    candidate: tuple[tuple[float, int, str], Path, dict[str, Any]],
) -> tuple[float, int, str]:
    return candidate[0]


@dataclass
class CheckpointProgress:
    """The exact optimizer boundary from which training can continue."""

    epoch_index: int = 0
    next_batch_index: int = 0
    global_step: int = 0
    training_history: list[dict[str, Any]] = field(default_factory=list)
    accumulated_loss: float = 0.0
    correct_pairs: int = 0
    observed_pairs: int = 0
    pair_type_counts: dict[str, int] = field(default_factory=dict)
    selection_history: list[dict[str, Any]] = field(default_factory=list)


def checkpoint_signature(configuration: dict[str, Any]) -> str:
    """Return a stable hash used to reject incompatible checkpoints."""
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shuffled_epoch_items(items: Sequence[Any], seed: int) -> list[Any]:
    """Shuffle without consuming process-global RNG state.

    A retry reconstructs exactly the same list, then slices off the batches
    recorded by the checkpoint. This avoids both re-training and re-tokenizing
    all already-completed batches.
    """
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def remaining_epoch_items(
    items: Sequence[Any], batch_size: int, next_batch_index: int
) -> list[Any]:
    """Return items after the completed, zero-based batch boundary."""
    if batch_size <= 0 or next_batch_index < 0:
        raise ValueError("batch_size must be positive and batch index non-negative")
    offset = min(next_batch_index * batch_size, len(items))
    return list(items[offset:])


def read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    return json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))


def find_latest_checkpoint(
    checkpoint_root: Path, expected_signature: str
) -> Path | None:
    """Find the newest complete compatible checkpoint.

    A directory without COMPLETE may have been interrupted during torch.save
    and is intentionally ignored. If the newest complete checkpoint belongs
    to another training configuration, fail instead of silently starting over
    or loading state into an incompatible optimizer/model.
    """
    if not checkpoint_root.is_dir():
        return None

    candidates: list[tuple[tuple[int, int, int, str], Path, dict[str, Any]]] = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if not path.is_dir() or not (path / COMPLETE_MARKER).is_file():
            continue
        try:
            metadata = read_checkpoint_metadata(path)
            sort_key = (
                int(metadata["global_step"]),
                int(metadata["epoch_index"]),
                int(metadata["next_batch_index"]),
                str(metadata.get("created_at_utc", "")),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((sort_key, path, metadata))

    if not candidates:
        return None
    _, latest_path, latest_metadata = max(
        candidates, key=checkpoint_candidate_sort_key
    )
    if latest_metadata.get("signature") != expected_signature:
        raise RuntimeError(
            "Latest checkpoint is incompatible with this run's model, data, "
            f"or hyperparameters: {latest_path}"
        )
    if latest_metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint format in {latest_path}: "
            f"{latest_metadata.get('format_version')!r}"
        )
    return latest_path


def find_best_checkpoint(
    checkpoint_root: Path, expected_signature: str
) -> Path | None:
    """Find the complete model-selection checkpoint with the best metric.

    Recovery checkpoints answer where training should resume. Best checkpoints
    answer which already-observed adapter should be exported. They deliberately
    live in separate directory name spaces so recovery pruning cannot delete an
    earlier model whose development NDCG remains best.
    """
    if not checkpoint_root.is_dir():
        return None

    candidates: list[tuple[tuple[float, int, str], Path, dict[str, Any]]] = []
    for path in checkpoint_root.glob("best-checkpoint-*"):
        if not path.is_dir() or not (path / COMPLETE_MARKER).is_file():
            continue
        try:
            metadata = read_checkpoint_metadata(path)
            sort_key = (
                float(metadata["selection_metric_value"]),
                int(metadata["global_step"]),
                str(metadata.get("created_at_utc", "")),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((sort_key, path, metadata))

    if not candidates:
        return None
    _, best_path, best_metadata = max(
        candidates, key=best_checkpoint_candidate_sort_key
    )
    if best_metadata.get("signature") != expected_signature:
        raise RuntimeError(
            "Best checkpoint is incompatible with this run's model, data, "
            f"or hyperparameters: {best_path}"
        )
    if best_metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint format in {best_path}: "
            f"{best_metadata.get('format_version')!r}"
        )
    return best_path
