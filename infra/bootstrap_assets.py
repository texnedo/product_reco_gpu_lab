#!/usr/bin/env python3
"""Prepare and idempotently register the lab's versioned AML input assets."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from prepare_esci import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    configured_source,
    load_config,
    prepare_esci,
    project_path,
)


REQUIRED_AML_ENV = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
)
AML_ASSET_NAME = re.compile(r"[A-Za-z0-9_-]{1,255}\Z")


@dataclass(frozen=True)
class Fingerprint:
    sha256: str
    file_count: int
    size_bytes: int

    def tags(self) -> dict[str, str]:
        return {
            "source_sha256": self.sha256,
            "source_file_count": str(self.file_count),
            "source_size_bytes": str(self.size_bytes),
            "managed_by": "infra/bootstrap_assets.py",
        }


def validate_asset_name(name: str) -> None:
    if not AML_ASSET_NAME.fullmatch(name):
        raise ValueError(
            f"Invalid AML asset name {name!r}; use 1-255 letters, digits, "
            "dashes, or underscores only."
        )


def is_excluded(relative: PurePosixPath, exclusions: tuple[PurePosixPath, ...]) -> bool:
    return any(relative == item or item in relative.parents for item in exclusions)


def asset_files(root: Path, exclude: list[str] | None = None) -> list[tuple[Path, PurePosixPath]]:
    exclusions = tuple(PurePosixPath(item.strip("/")) for item in (exclude or []))
    files: list[tuple[Path, PurePosixPath]] = []
    for directory, child_dirs, names in os.walk(root):
        directory_path = Path(directory)
        child_dirs[:] = [
            name
            for name in child_dirs
            if not is_excluded(
                PurePosixPath((directory_path / name).relative_to(root).as_posix()),
                exclusions,
            )
        ]
        for name in names:
            path = directory_path / name
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if not is_excluded(relative, exclusions) and path.is_file():
                files.append((path, relative))
    return sorted(files, key=lambda item: str(item[1]))


def fingerprint_folder(root: Path, exclude: list[str] | None = None) -> Fingerprint:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path, relative in asset_files(root, exclude):
        stat = path.stat()
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
        count += 1
        size += stat.st_size
    if count == 0:
        raise ValueError(f"Asset folder contains no files: {root}")
    return Fingerprint(digest.hexdigest(), count, size)


def require_files(root: Path, names: list[str]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root} is missing required file(s): " + ", ".join(missing)
        )


@contextlib.contextmanager
def upload_path(
    source: Path, exclude: list[str] | None = None
) -> Iterator[Path]:
    """Stage an upload tree with excluded VCS files, using hard links when possible."""
    if not exclude:
        yield source
        return

    staging_root = PROJECT_ROOT / ".aml-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="asset-", dir=staging_root))
    try:
        for path, relative in asset_files(source, exclude):
            target = temporary / Path(str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def existing_asset(collection: Any, name: str, version: str) -> Any | None:
    from azure.core.exceptions import ResourceNotFoundError

    try:
        return collection.get(name=name, version=version)
    except ResourceNotFoundError:
        return None


def verify_existing(asset: Any, fingerprint: Fingerprint, uri: str) -> None:
    tags = asset.tags or {}
    registered_hash = tags.get("source_sha256")
    if registered_hash and registered_hash != fingerprint.sha256:
        raise RuntimeError(
            f"{uri} already exists but its source fingerprint differs. "
            "Do not overwrite an immutable version; bump the version in config/assets.yaml."
        )
    if registered_hash:
        print(f"✓ {uri} already exists and matches the local source")
    else:
        print(f"✓ {uri} already exists (no source fingerprint tag to verify)")


def connect_ml_client() -> Any:
    missing = [name for name in REQUIRED_AML_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    try:
        from azure.ai.ml import MLClient
        from azure.identity import AzureCliCredential
    except ImportError as exc:
        raise RuntimeError(
            "Azure ML dependencies are missing; install requirements-bootstrap.txt"
        ) from exc

    return MLClient(
        credential=AzureCliCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )


def register_data(ml_client: Any, asset: dict[str, Any], path: Path) -> str:
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data

    name, version = str(asset["name"]), str(asset["version"])
    uri = f"azureml:{name}:{version}"
    fingerprint = fingerprint_folder(path)
    existing = existing_asset(ml_client.data, name, version)
    if existing:
        verify_existing(existing, fingerprint, uri)
        return uri

    tags = fingerprint.tags()
    tags.update(
        {
            "locale": str(asset["locale"]),
            "source_revision": str(asset.get("source_revision", "unknown")),
        }
    )
    print(f"Uploading and registering {uri} ({fingerprint.size_bytes:,} bytes) ...")
    ml_client.data.create_or_update(
        Data(
            name=name,
            version=version,
            description=str(asset.get("description", "")),
            path=str(path),
            type=AssetTypes.URI_FOLDER,
            tags=tags,
        )
    )
    print(f"✓ registered {uri}")
    return uri


def register_model(ml_client: Any, asset: dict[str, Any], source: Path) -> str:
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    exclude = [str(item) for item in asset.get("exclude", [])]
    require_files(source, [str(item) for item in asset.get("required_files", [])])
    fingerprint = fingerprint_folder(source, exclude)
    name, version = str(asset["name"]), str(asset["version"])
    uri = f"azureml:{name}:{version}"
    existing = existing_asset(ml_client.models, name, version)
    if existing:
        verify_existing(existing, fingerprint, uri)
        return uri

    tags = fingerprint.tags()
    tags.update(
        {
            "hf_repo": str(asset.get("hf_repo", "unknown")),
            "hf_revision": str(asset.get("hf_revision", "unknown")),
        }
    )
    print(f"Uploading and registering {uri} ({fingerprint.size_bytes:,} bytes) ...")
    with upload_path(source, exclude) as staged:
        ml_client.models.create_or_update(
            Model(
                name=name,
                version=version,
                description=str(asset.get("description", "")),
                path=str(staged),
                type=AssetTypes.CUSTOM_MODEL,
                tags=tags,
            )
        )
    print(f"✓ registered {uri}")
    return uri


def show_plan(
    data_asset: dict[str, Any],
    data_path: Path,
    model_asset: dict[str, Any],
    model_path: Path,
) -> None:
    print("Dry run; no Azure resources will be changed.\n")
    for kind, asset, path, exclusions in (
        ("data", data_asset, data_path, []),
        ("model", model_asset, model_path, asset_exclusions(model_asset)),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{kind} path does not exist: {path}")
        fingerprint = fingerprint_folder(path, exclusions)
        print(f"{kind}:  azureml:{asset['name']}:{asset['version']}")
        print(f"  source: {path}")
        print(f"  files:  {fingerprint.file_count:,}")
        print(f"  bytes:  {fingerprint.size_bytes:,}")
        print(f"  sha256: {fingerprint.sha256}")


def asset_exclusions(asset: dict[str, Any]) -> list[str]:
    return [str(item) for item in asset.get("exclude", [])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-source", help="Override the raw ESCI source folder")
    parser.add_argument("--model-source", help="Override the Qwen model folder")
    parser.add_argument(
        "--skip-data-preparation",
        action="store_true",
        help="Require the prepared data folder to exist instead of creating it",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and fingerprint without Azure"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    data_asset = config["data"]["esci"]
    model_asset = config["models"]["qwen"]
    validate_asset_name(str(data_asset["name"]))
    validate_asset_name(str(model_asset["name"]))

    raw_data = configured_source(data_asset, args.data_source)
    prepared_data = project_path(str(data_asset["prepared_path"]))
    model_source = configured_source(model_asset, args.model_source)

    if not args.skip_data_preparation:
        files = data_asset["files"]
        prepare_esci(
            source=raw_data,
            destination=prepared_data,
            locale=str(data_asset["locale"]),
            examples_name=str(files["examples"]),
            products_name=str(files["products"]),
            source_revision=str(data_asset.get("source_revision", "unknown")),
        )
    elif not prepared_data.is_dir():
        raise FileNotFoundError(
            f"Prepared data not found: {prepared_data}. Remove --skip-data-preparation."
        )

    if args.dry_run:
        show_plan(data_asset, prepared_data, model_asset, model_source)
        return 0

    ml_client = connect_ml_client()
    print(
        "AML workspace: "
        f"{os.environ['AZURE_RESOURCE_GROUP']} / {os.environ['AZURE_ML_WORKSPACE']}\n"
    )
    uris = (
        register_data(ml_client, data_asset, prepared_data),
        register_model(ml_client, model_asset, model_source),
    )
    print("\nWorkspace input assets:")
    for uri in uris:
        print(f"  {uri}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
