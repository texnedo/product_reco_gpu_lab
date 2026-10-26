#!/usr/bin/env python3
"""Submit a YAML-configured ESCI LoRA training or evaluation job to Azure ML."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import sys
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import BuildContext, Environment
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_CONTEXT = PROJECT_ROOT / "environments" / "train"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_config import (  # noqa: E402
    SAMPLING_STRATEGIES,
    TEXT_CLEANING_MODES,
    apply_overrides,
    generated_display_name,
    load_experiment_config,
)


REQUIRED_ENV = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
    "AZURE_ML_COMPUTE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Versioned experiment YAML; required to prevent accidental runs.",
    )
    parser.add_argument("--data-asset")
    parser.add_argument("--model-asset")
    parser.add_argument("--environment-name", default="qwen-esci-training")
    parser.add_argument("--environment-version", default="1")
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--display-name",
        help="AML display name (defaults to a mode-specific name).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the unmodified base Qwen model; do not train LoRA.",
    )
    # Explicit CLI values override YAML. Defaults remain None so merely invoking
    # the submitter cannot silently replace a value from the selected config.
    parser.add_argument("--train-queries", type=int)
    parser.add_argument("--eval-queries", type=int)
    parser.add_argument(
        "--small-version", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--text-cleaning", choices=sorted(TEXT_CLEANING_MODES))
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--pairs-per-query", type=int)
    parser.add_argument("--sampling-strategy", choices=sorted(SAMPLING_STRATEGIES))
    parser.add_argument("--balance-fraction", type=float)
    parser.add_argument("--balanced-pairs-per-type-per-query", type=int)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        help="Persist state after this many optimizer updates.",
    )
    parser.add_argument("--checkpoints-to-keep", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--pair-weighting",
        choices=("none", "grade_gap"),
    )
    parser.add_argument("--ndcg-k", type=int)
    parser.add_argument("--selection-queries", type=int)
    parser.add_argument("--selection-every-steps", type=int)
    parser.add_argument("--selection-min-delta", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved job without contacting Azure ML.",
    )
    parser.add_argument("--no-stream", action="store_true")
    return parser.parse_args()


def resolve_experiment_config(args: argparse.Namespace):
    config = load_experiment_config(args.config)
    return apply_overrides(
        config,
        {
            "data_asset": args.data_asset,
            "model_asset": args.model_asset,
            "train_queries": args.train_queries,
            "eval_queries": args.eval_queries,
            "small_version": args.small_version,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "text_cleaning": args.text_cleaning,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "pairs_per_query": args.pairs_per_query,
            "sampling_strategy": args.sampling_strategy,
            "balance_fraction": args.balance_fraction,
            "balanced_pairs_per_type_per_query": (
                args.balanced_pairs_per_type_per_query
            ),
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "checkpoints_to_keep": args.checkpoints_to_keep,
            "resume": args.resume,
            "pair_weighting": args.pair_weighting,
            "ndcg_k": args.ndcg_k,
            "selection_queries": args.selection_queries,
            "selection_every_steps": args.selection_every_steps,
            "selection_min_delta": args.selection_min_delta,
            "num_workers": args.num_workers,
            "seed": args.seed,
        },
    )


def project_relative_file(path: Path, description: str) -> str:
    """Require files referenced by AML commands to be inside the code snapshot."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"{description} must be inside {PROJECT_ROOT}") from exc


def validate_environment() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def context_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_environment(
    ml_client: MLClient, name: str, version: str
) -> str:
    uri = f"azureml:{name}:{version}"
    fingerprint = context_fingerprint(ENVIRONMENT_CONTEXT)
    try:
        existing = ml_client.environments.get(name=name, version=version)
    except ResourceNotFoundError:
        existing = None

    if existing:
        registered = (existing.tags or {}).get("source_sha256")
        if registered and registered != fingerprint:
            raise RuntimeError(
                f"{uri} exists but the Docker context has changed; bump its version."
            )
        print(f"Using existing environment {uri}")
        return uri

    print(
        f"Registering environment {uri}; its first image build can take several minutes ..."
    )
    ml_client.environments.create_or_update(
        Environment(
            name=name,
            version=version,
            description="Qwen3 ESCI LoRA training environment on Azure ACPT.",
            build=BuildContext(path=str(ENVIRONMENT_CONTEXT)),
            tags={"source_sha256": fingerprint},
        )
    )
    return uri


def main() -> int:
    args = parse_args()
    config = resolve_experiment_config(args)
    config_file = project_relative_file(config.source_path, "Experiment config")
    prompt_file = project_relative_file(
        PROJECT_ROOT / config.prompt_file, "Prompt file"
    )
    display_name = args.display_name or generated_display_name(
        config, args.eval_only
    )
    trainer_command = (
        "python src/train_esci.py "
        "--data-dir ${{inputs.esci_data}} "
        "--model-dir ${{inputs.base_model}} "
        f"--prompt-file {shlex.quote(prompt_file)} "
        f"--experiment-config-file {shlex.quote(config_file)} "
        f"--run-display-name {shlex.quote(display_name)} "
        "--metrics-output ${{outputs.evaluation}} "
        f"--eval-queries {config.eval_queries} "
        f"--max-length {config.max_length} "
        f"--text-cleaning {config.text_cleaning} "
        f"--eval-batch-size {config.eval_batch_size} "
        f"--ndcg-k {config.ndcg_k} "
        f"--selection-queries {config.selection_queries} "
        f"--selection-every-steps {config.selection_every_steps} "
        f"--selection-min-delta {config.selection_min_delta} "
        f"--num-workers {config.num_workers} "
        f"--seed {config.seed} "
        + ("--small-version" if config.small_version else "--no-small-version")
    )
    outputs = {"evaluation": Output(type=AssetTypes.URI_FOLDER)}
    if args.eval_only:
        trainer_command += " --eval-only"
    else:
        trainer_command += (
            " --model-output ${{outputs.trained_model}}"
            " --checkpoint-dir ${{outputs.checkpoints}}"
            f" --train-queries {config.train_queries}"
            f" --epochs {config.epochs}"
            f" --train-batch-size {config.train_batch_size}"
            f" --gradient-accumulation-steps {config.gradient_accumulation_steps}"
            f" --learning-rate {config.learning_rate}"
            f" --warmup-ratio {config.warmup_ratio}"
            f" --weight-decay {config.weight_decay}"
            f" --lora-rank {config.lora_rank}"
            f" --lora-alpha {config.lora_alpha}"
            f" --lora-dropout {config.lora_dropout}"
            f" --pairs-per-query {config.pairs_per_query}"
            f" --sampling-strategy {config.sampling_strategy}"
            f" --balance-fraction {config.balance_fraction}"
            " --balanced-pairs-per-type-per-query"
            f" {config.balanced_pairs_per_type_per_query}"
            f" --pair-weighting {config.pair_weighting}"
            f" --checkpoint-every-steps {config.checkpoint_every_steps}"
            f" --checkpoints-to-keep {config.checkpoints_to_keep}"
            + (" --resume" if config.resume else " --no-resume")
        )
        outputs.update(
            {
                "trained_model": Output(type=AssetTypes.CUSTOM_MODEL),
                # Unlike upload mode, rw_mount writes checkpoints to durable
                # backing storage while the job is still running. AML retries
                # therefore see the same completed checkpoint folders.
                "checkpoints": Output(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.RW_MOUNT,
                ),
            }
        )

    if args.dry_run:
        print(f"config:       {config_file} ({config.source_sha256[:12]})")
        print(f"display name: {display_name}")
        print(f"experiment:   {args.experiment_name or config.aml_experiment_name}")
        print(f"data asset:   {config.data_asset}")
        print(f"model asset:  {config.model_asset}")
        print(f"command:      {trainer_command}")
        return 0

    validate_environment()
    ml_client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )
    environment = ensure_environment(
        ml_client, args.environment_name, args.environment_version
    )

    job = command(
        code=str(PROJECT_ROOT),
        command=trainer_command,
        inputs={
            "esci_data": Input(
                type=AssetTypes.URI_FOLDER,
                path=config.data_asset,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "base_model": Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=config.model_asset,
                mode=InputOutputModes.DOWNLOAD,
            ),
        },
        outputs=outputs,
        environment=environment,
        compute=os.environ["AZURE_ML_COMPUTE"],
        experiment_name=args.experiment_name or config.aml_experiment_name,
        display_name=display_name,
        environment_variables={"TOKENIZERS_PARALLELISM": "false"},
    )
    submitted = ml_client.jobs.create_or_update(job)
    print(
        "Submitted AML evaluation job:"
        if args.eval_only
        else "Submitted AML training job:"
    )
    print(f"  name:   {submitted.name}")
    print(f"  config: {config_file} ({config.source_sha256[:12]})")
    print(f"  display: {display_name}")
    print(f"  status: {submitted.status}")
    if submitted.services and "Studio" in submitted.services:
        print(f"  studio: {submitted.services['Studio'].endpoint}")
    if not args.no_stream:
        ml_client.jobs.stream(submitted.name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
