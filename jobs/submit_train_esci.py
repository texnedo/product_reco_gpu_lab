#!/usr/bin/env python3
"""Submit the v1 ESCI LoRA training and evaluation job to Azure ML."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import BuildContext, Environment
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_CONTEXT = PROJECT_ROOT / "environments" / "train"
REQUIRED_ENV = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
    "AZURE_ML_COMPUTE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-asset", default="azureml:esci-ranking:1")
    parser.add_argument("--model-asset", default="azureml:qwen3-reranker-0-6b:1")
    parser.add_argument("--environment-name", default="qwen-esci-training")
    parser.add_argument("--environment-version", default="1")
    parser.add_argument("--experiment-name", default="qwen-esci-reranker")
    parser.add_argument("--display-name", default="qwen-esci-v1-lora")
    parser.add_argument("--train-queries", type=int, default=5_000)
    parser.add_argument("--eval-queries", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--pairs-per-query", type=int, default=12)
    parser.add_argument(
        "--pair-weighting",
        choices=("none", "grade_gap"),
        default="none",
    )
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--no-stream", action="store_true")
    return parser.parse_args()


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
        command=(
            "python src/train_esci.py "
            "--data-dir ${{inputs.esci_data}} "
            "--model-dir ${{inputs.base_model}} "
            "--prompt-file config/prompts/esci_v1.txt "
            "--model-output ${{outputs.trained_model}} "
            "--metrics-output ${{outputs.evaluation}} "
            f"--train-queries {args.train_queries} "
            f"--eval-queries {args.eval_queries} "
            f"--epochs {args.epochs} "
            f"--max-length {args.max_length} "
            f"--train-batch-size {args.train_batch_size} "
            f"--eval-batch-size {args.eval_batch_size} "
            f"--gradient-accumulation-steps {args.gradient_accumulation_steps} "
            f"--learning-rate {args.learning_rate} "
            f"--pairs-per-query {args.pairs_per_query} "
            f"--pair-weighting {args.pair_weighting} "
            f"--ndcg-k {args.ndcg_k}"
        ),
        inputs={
            "esci_data": Input(
                type=AssetTypes.URI_FOLDER,
                path=args.data_asset,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "base_model": Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=args.model_asset,
                mode=InputOutputModes.DOWNLOAD,
            ),
        },
        outputs={
            "trained_model": Output(type=AssetTypes.CUSTOM_MODEL),
            "evaluation": Output(type=AssetTypes.URI_FOLDER),
        },
        environment=environment,
        compute=os.environ["AZURE_ML_COMPUTE"],
        experiment_name=args.experiment_name,
        display_name=args.display_name,
        environment_variables={"TOKENIZERS_PARALLELISM": "false"},
    )
    submitted = ml_client.jobs.create_or_update(job)
    print("Submitted AML training job:")
    print(f"  name:   {submitted.name}")
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
