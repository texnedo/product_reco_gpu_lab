"""Validated, versioned experiment configuration for ESCI AML jobs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from esci_core import TEXT_CLEANING_MODES


SAMPLING_STRATEGIES = frozenset(("query_round_robin", "mixed_type_balanced"))
PAIR_WEIGHTINGS = frozenset(("none", "grade_gap"))


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved values used to construct one reproducible training job."""

    source_path: Path
    source_sha256: str
    run_label: str
    aml_experiment_name: str
    data_asset: str
    model_asset: str
    prompt_file: str
    text_cleaning: str
    train_queries: int
    eval_queries: int
    small_version: bool
    max_length: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    epochs: int
    train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    pair_weighting: str
    sampling_strategy: str
    pairs_per_query: int
    balance_fraction: float
    balanced_pairs_per_type_per_query: int
    eval_batch_size: int
    ndcg_k: int
    selection_queries: int
    selection_every_steps: int
    selection_min_delta: float
    checkpoint_every_steps: int
    checkpoints_to_keep: int
    resume: bool
    num_workers: int
    seed: int


SECTION_KEYS = {
    "run": frozenset(("label", "aml_experiment_name")),
    "assets": frozenset(("data", "model")),
    "prompt": frozenset(("file",)),
    "preprocessing": frozenset(("text_cleaning",)),
    "data": frozenset(("train_queries", "eval_queries", "small_version")),
    "model": frozenset(("max_length",)),
    "lora": frozenset(("rank", "alpha", "dropout")),
    "training": frozenset(
        (
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_ratio",
            "weight_decay",
            "pair_weighting",
        )
    ),
    "sampling": frozenset(
        (
            "strategy",
            "pairs_per_query",
            "balance_fraction",
            "balanced_pairs_per_type_per_query",
        )
    ),
    "evaluation": frozenset(
        (
            "batch_size",
            "ndcg_k",
            "selection_queries",
            "selection_every_steps",
            "selection_min_delta",
        )
    ),
    "checkpoint": frozenset(("every_steps", "keep", "resume")),
    "runtime": frozenset(("num_workers", "seed")),
}


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Experiment config section {name!r} must be a mapping")
    unknown = set(value) - SECTION_KEYS[name]
    missing = SECTION_KEYS[name] - set(value)
    if unknown:
        raise ValueError(f"Unknown keys in {name}: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing keys in {name}: {sorted(missing)}")
    return value


def _exact(value: Any, expected_type: type, path: str) -> Any:
    if type(value) is not expected_type:
        raise ValueError(
            f"{path} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def validate_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    """Validate both YAML-loaded configs and programmatic CLI overrides."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", config.run_label):
        raise ValueError("run.label may contain lowercase letters, digits, and dashes")
    if config.sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(
            f"sampling.strategy must be one of {sorted(SAMPLING_STRATEGIES)}"
        )
    if config.pair_weighting not in PAIR_WEIGHTINGS:
        raise ValueError(
            f"training.pair_weighting must be one of {sorted(PAIR_WEIGHTINGS)}"
        )
    if config.text_cleaning not in TEXT_CLEANING_MODES:
        raise ValueError(
            f"preprocessing.text_cleaning must be one of "
            f"{sorted(TEXT_CLEANING_MODES)}"
        )
    positive = {
        "data.train_queries": config.train_queries,
        "data.eval_queries": config.eval_queries,
        "model.max_length": config.max_length,
        "lora.rank": config.lora_rank,
        "lora.alpha": config.lora_alpha,
        "training.epochs": config.epochs,
        "training.batch_size": config.train_batch_size,
        "training.gradient_accumulation_steps": config.gradient_accumulation_steps,
        "training.learning_rate": config.learning_rate,
        "sampling.pairs_per_query": config.pairs_per_query,
        "sampling.balanced_pairs_per_type_per_query": (
            config.balanced_pairs_per_type_per_query
        ),
        "evaluation.batch_size": config.eval_batch_size,
        "evaluation.ndcg_k": config.ndcg_k,
        "checkpoint.every_steps": config.checkpoint_every_steps,
        "checkpoint.keep": config.checkpoints_to_keep,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Experiment values must be positive: {invalid}")
    if config.num_workers < 0:
        raise ValueError("runtime.num_workers cannot be negative")
    if not 0.0 <= config.lora_dropout < 1.0:
        raise ValueError("lora.dropout must be in [0, 1)")
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    if config.weight_decay < 0.0:
        raise ValueError("training.weight_decay cannot be negative")
    if config.selection_queries < 0 or config.selection_every_steps < 0:
        raise ValueError("evaluation selection values cannot be negative")
    if config.selection_min_delta < 0.0:
        raise ValueError("evaluation.selection_min_delta cannot be negative")
    selection_enabled = (
        config.selection_queries > 0 or config.selection_every_steps > 0
    )
    if selection_enabled and (
        config.selection_queries <= 0 or config.selection_every_steps <= 0
    ):
        raise ValueError(
            "evaluation.selection_queries and selection_every_steps must both "
            "be positive or both be zero"
        )
    if not selection_enabled and config.selection_min_delta != 0.0:
        raise ValueError(
            "evaluation.selection_min_delta requires model selection to be enabled"
        )
    if config.selection_queries >= config.eval_queries:
        raise ValueError(
            "evaluation.selection_queries must leave at least one final "
            "holdout query"
        )
    if not 0.0 <= config.balance_fraction <= 1.0:
        raise ValueError("sampling.balance_fraction must be in [0, 1]")
    if (
        config.sampling_strategy == "mixed_type_balanced"
        and config.balance_fraction <= 0.0
    ):
        raise ValueError(
            "mixed_type_balanced requires a positive sampling.balance_fraction"
        )
    if (
        config.sampling_strategy == "query_round_robin"
        and config.balance_fraction != 0.0
    ):
        raise ValueError(
            "query_round_robin requires sampling.balance_fraction: 0.0"
        )
    return config


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load a strict YAML schema so misspelled settings fail before submission."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    expected_top_level = {"schema_version", *SECTION_KEYS}
    unknown = set(document) - expected_top_level
    missing = expected_top_level - set(document)
    if unknown:
        raise ValueError(f"Unknown experiment config sections: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing experiment config sections: {sorted(missing)}")
    if document["schema_version"] != 1:
        raise ValueError("Only experiment config schema_version: 1 is supported")

    run = _section(document, "run")
    assets = _section(document, "assets")
    prompt = _section(document, "prompt")
    preprocessing = _section(document, "preprocessing")
    data = _section(document, "data")
    model = _section(document, "model")
    lora = _section(document, "lora")
    training = _section(document, "training")
    sampling = _section(document, "sampling")
    evaluation = _section(document, "evaluation")
    checkpoint = _section(document, "checkpoint")
    runtime = _section(document, "runtime")

    config = ExperimentConfig(
        source_path=path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        run_label=_exact(run["label"], str, "run.label"),
        aml_experiment_name=_exact(
            run["aml_experiment_name"], str, "run.aml_experiment_name"
        ),
        data_asset=_exact(assets["data"], str, "assets.data"),
        model_asset=_exact(assets["model"], str, "assets.model"),
        prompt_file=_exact(prompt["file"], str, "prompt.file"),
        text_cleaning=_exact(
            preprocessing["text_cleaning"],
            str,
            "preprocessing.text_cleaning",
        ),
        train_queries=_exact(data["train_queries"], int, "data.train_queries"),
        eval_queries=_exact(data["eval_queries"], int, "data.eval_queries"),
        small_version=_exact(data["small_version"], bool, "data.small_version"),
        max_length=_exact(model["max_length"], int, "model.max_length"),
        lora_rank=_exact(lora["rank"], int, "lora.rank"),
        lora_alpha=_exact(lora["alpha"], int, "lora.alpha"),
        lora_dropout=_number(lora["dropout"], "lora.dropout"),
        epochs=_exact(training["epochs"], int, "training.epochs"),
        train_batch_size=_exact(
            training["batch_size"], int, "training.batch_size"
        ),
        gradient_accumulation_steps=_exact(
            training["gradient_accumulation_steps"],
            int,
            "training.gradient_accumulation_steps",
        ),
        learning_rate=_number(training["learning_rate"], "training.learning_rate"),
        warmup_ratio=_number(training["warmup_ratio"], "training.warmup_ratio"),
        weight_decay=_number(training["weight_decay"], "training.weight_decay"),
        pair_weighting=_exact(
            training["pair_weighting"], str, "training.pair_weighting"
        ),
        sampling_strategy=_exact(
            sampling["strategy"], str, "sampling.strategy"
        ),
        pairs_per_query=_exact(
            sampling["pairs_per_query"], int, "sampling.pairs_per_query"
        ),
        balance_fraction=_number(
            sampling["balance_fraction"], "sampling.balance_fraction"
        ),
        balanced_pairs_per_type_per_query=_exact(
            sampling["balanced_pairs_per_type_per_query"],
            int,
            "sampling.balanced_pairs_per_type_per_query",
        ),
        eval_batch_size=_exact(
            evaluation["batch_size"], int, "evaluation.batch_size"
        ),
        ndcg_k=_exact(evaluation["ndcg_k"], int, "evaluation.ndcg_k"),
        selection_queries=_exact(
            evaluation["selection_queries"],
            int,
            "evaluation.selection_queries",
        ),
        selection_every_steps=_exact(
            evaluation["selection_every_steps"],
            int,
            "evaluation.selection_every_steps",
        ),
        selection_min_delta=_number(
            evaluation["selection_min_delta"],
            "evaluation.selection_min_delta",
        ),
        checkpoint_every_steps=_exact(
            checkpoint["every_steps"], int, "checkpoint.every_steps"
        ),
        checkpoints_to_keep=_exact(checkpoint["keep"], int, "checkpoint.keep"),
        resume=_exact(checkpoint["resume"], bool, "checkpoint.resume"),
        num_workers=_exact(runtime["num_workers"], int, "runtime.num_workers"),
        seed=_exact(runtime["seed"], int, "runtime.seed"),
    )
    return validate_experiment_config(config)


def apply_overrides(
    config: ExperimentConfig, overrides: dict[str, Any]
) -> ExperimentConfig:
    """Apply explicit CLI values after YAML while retaining validation."""
    supplied = {name: value for name, value in overrides.items() if value is not None}
    return validate_experiment_config(replace(config, **supplied))


def _float_slug(value: float) -> str:
    text = f"{value:.0e}"
    return text.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def generated_display_name(config: ExperimentConfig, eval_only: bool) -> str:
    """Build an AML-safe name whose important experimental variables are visible."""
    if eval_only:
        final_queries = config.eval_queries - config.selection_queries
        evaluation_slug = f"eval{final_queries}"
        if config.selection_queries:
            evaluation_slug += f"-dev{config.selection_queries}"
        name = (
            f"qwen-esci-{config.run_label}-base-{evaluation_slug}"
            f"-ml{config.max_length}-s{config.seed}"
        )
    else:
        sampler = (
            "rr"
            if config.sampling_strategy == "query_round_robin"
            else f"mix{round(config.balance_fraction * 100)}"
        )
        name = (
            f"qwen-esci-{config.run_label}-{sampler}-p{config.pairs_per_query}"
            f"-e{config.epochs}-b{config.train_batch_size}"
            f"-ga{config.gradient_accumulation_steps}"
            f"-lr{_float_slug(config.learning_rate)}-r{config.lora_rank}"
            f"-s{config.seed}"
        )
    if config.text_cleaning != "none":
        cleaning_slug = config.text_cleaning.replace("basic_v", "clean")
        name = f"{name}-{cleaning_slug}"
    if not eval_only and config.selection_queries:
        final_queries = config.eval_queries - config.selection_queries
        name = (
            f"{name}-dev{config.selection_queries}-h{final_queries}"
        )
    if len(name) > 255:
        raise ValueError("Generated AML display name exceeds 255 characters")
    return name
