"""Serialize and restore PEFT training state at optimizer boundaries."""

from __future__ import annotations

import json
import os
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from checkpoint_core import (
    CHECKPOINT_FORMAT_VERSION,
    COMPLETE_MARKER,
    METADATA_FILE,
    CheckpointProgress,
    find_best_checkpoint,
    find_latest_checkpoint,
    read_checkpoint_metadata,
)


ADAPTER_STATE_FILE = "adapter_state.pt"
TRAINING_STATE_FILE = "training_state.pt"


def write_checkpoint_state(
    destination: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    progress: CheckpointProgress,
) -> None:
    """Write the adapter and every state needed to branch or resume training."""
    adapter_state = get_peft_model_state_dict(model)
    torch.save(adapter_state, destination / ADAPTER_STATE_FILE)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "progress": progress,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        },
        destination / TRAINING_STATE_FILE,
    )


def completed_checkpoint_sort_key(
    checkpoint: tuple[tuple[int, int, int, str], Path],
) -> tuple[int, int, int, str]:
    return checkpoint[0]


def save_training_checkpoint(
    checkpoint_root: Path,
    signature: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    progress: CheckpointProgress,
    checkpoints_to_keep: int,
) -> Path:
    """Write a versioned checkpoint and mark it complete only at the end.

    The caller invokes this only after optimizer.step(), scheduler.step(), and
    zero_grad(). Therefore no partially accumulated gradients need to be saved.
    If the VM is evicted during either torch.save call, COMPLETE is absent and
    the next attempt ignores that directory.
    """
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_name = (
        f"checkpoint-{progress.global_step:08d}"
        f"-e{progress.epoch_index:04d}"
        f"-b{progress.next_batch_index:08d}"
        f"-{uuid.uuid4().hex[:8]}"
    )
    destination = checkpoint_root / checkpoint_name
    destination.mkdir()

    write_checkpoint_state(
        destination, model, optimizer, scheduler, scaler, progress
    )

    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "signature": signature,
        "global_step": progress.global_step,
        "epoch_index": progress.epoch_index,
        "next_batch_index": progress.next_batch_index,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # This is the commit record. A checkpoint is never loadable before it.
    (destination / COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")

    # Recovery scans COMPLETE directories, so this human-friendly pointer is
    # never a single point of failure.
    pointer = checkpoint_root / "latest.json"
    temporary_pointer = checkpoint_root / f".latest-{uuid.uuid4().hex}.json"
    temporary_pointer.write_text(
        json.dumps({"checkpoint": checkpoint_name}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.replace(temporary_pointer, pointer)
    except OSError as exc:
        # Recovery scans checkpoint directories and does not depend on this
        # convenience pointer. Some remote mount drivers restrict renames.
        print(f"warning: could not update {pointer}: {exc}")
    prune_completed_checkpoints(checkpoint_root, checkpoints_to_keep)
    print(
        "Saved training checkpoint "
        f"step={progress.global_step} epoch_index={progress.epoch_index} "
        f"next_batch={progress.next_batch_index} to {destination}",
        flush=True,
    )
    return destination


def save_best_training_checkpoint(
    checkpoint_root: Path,
    signature: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    progress: CheckpointProgress,
    selection_metric: str,
    selection_metric_value: float,
    development_metrics: dict[str, Any],
) -> Path:
    """Persist a complete best-model state outside recovery rotation."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_name = (
        f"best-checkpoint-{progress.global_step:08d}-{uuid.uuid4().hex[:8]}"
    )
    destination = checkpoint_root / checkpoint_name
    destination.mkdir()
    write_checkpoint_state(
        destination, model, optimizer, scheduler, scaler, progress
    )
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": "model_selection",
        "signature": signature,
        "global_step": progress.global_step,
        "epoch_index": progress.epoch_index,
        "next_batch_index": progress.next_batch_index,
        "selection_metric": selection_metric,
        "selection_metric_value": float(selection_metric_value),
        "development_metrics": development_metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")

    # Only after the new candidate is complete is the previous best removed.
    # Partial writes remain visible for diagnosis but are ignored by loading.
    for obsolete in checkpoint_root.glob("best-checkpoint-*"):
        if obsolete == destination or not (obsolete / COMPLETE_MARKER).is_file():
            continue
        try:
            obsolete_metadata = read_checkpoint_metadata(obsolete)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if obsolete_metadata.get("signature") != signature:
            continue
        try:
            shutil.rmtree(obsolete)
        except OSError as exc:
            print(f"warning: could not prune old best checkpoint {obsolete}: {exc}")
    print(
        "Saved best model checkpoint "
        f"step={progress.global_step} {selection_metric}="
        f"{selection_metric_value:.6f} to {destination}",
        flush=True,
    )
    return destination


def best_checkpoint_metadata(
    checkpoint_root: Path, signature: str
) -> tuple[Path, dict[str, Any]] | None:
    """Return the compatible best checkpoint and its small JSON metadata."""
    source = find_best_checkpoint(checkpoint_root, signature)
    if source is None:
        return None
    return source, read_checkpoint_metadata(source)


def load_best_model_checkpoint(
    checkpoint_root: Path, signature: str, model: Any
) -> tuple[Path, dict[str, Any]]:
    """Load only the best adapter into the current PEFT model for export."""
    resolved = best_checkpoint_metadata(checkpoint_root, signature)
    if resolved is None:
        raise RuntimeError(f"No complete best checkpoint found in {checkpoint_root}")
    source, metadata = resolved
    adapter_state = torch.load(
        source / ADAPTER_STATE_FILE,
        map_location="cpu",
        weights_only=True,
    )
    set_peft_model_state_dict(model, adapter_state)
    print(
        "Loaded best model checkpoint "
        f"step={metadata['global_step']} "
        f"{metadata['selection_metric']}="
        f"{float(metadata['selection_metric_value']):.6f}: {source}",
        flush=True,
    )
    return source, metadata


def prune_completed_checkpoints(checkpoint_root: Path, keep: int) -> None:
    """Keep the newest complete checkpoints; never touch partial writes."""
    completed: list[tuple[tuple[int, int, int, str], Path]] = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if not path.is_dir() or not (path / COMPLETE_MARKER).is_file():
            continue
        try:
            metadata = read_checkpoint_metadata(path)
            key = (
                int(metadata["global_step"]),
                int(metadata["epoch_index"]),
                int(metadata["next_batch_index"]),
                str(metadata.get("created_at_utc", "")),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        completed.append((key, path))
    for _, obsolete in sorted(completed, key=completed_checkpoint_sort_key)[:-keep]:
        try:
            shutil.rmtree(obsolete)
        except OSError as exc:
            print(f"warning: could not prune old checkpoint {obsolete}: {exc}")


def load_training_checkpoint(
    checkpoint_root: Path,
    signature: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
) -> CheckpointProgress | None:
    """Restore the most advanced complete checkpoint, if one exists."""
    source = find_latest_checkpoint(checkpoint_root, signature)
    if source is None:
        print(f"No complete checkpoint found in {checkpoint_root}; starting fresh")
        return None

    adapter_state = torch.load(
        source / ADAPTER_STATE_FILE,
        map_location="cpu",
        weights_only=True,
    )
    set_peft_model_state_dict(model, adapter_state)
    state = torch.load(
        source / TRAINING_STATE_FILE,
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    progress = state["progress"]
    if not isinstance(progress, CheckpointProgress):
        raise RuntimeError(f"Invalid progress state in checkpoint: {source}")
    print(
        "Resuming from checkpoint "
        f"step={progress.global_step} epoch_index={progress.epoch_index} "
        f"next_batch={progress.next_batch_index}: {source}",
        flush=True,
    )
    return progress
