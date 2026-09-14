#!/usr/bin/env python3
"""Prepare an immutable, English-only ESCI folder for AML registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "assets.yaml"
METADATA_FILE = "dataset_info.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def configured_source(asset: dict[str, Any], override: str | None = None) -> Path:
    value = override
    env_name = asset.get("source_path_env")
    if not value and env_name:
        value = os.getenv(str(env_name))
    return project_path(value or str(asset["source_path"]))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def filter_parquet(source: Path, destination: Path, locale: str) -> int:
    """Filter one parquet file by product_locale without loading it all in RAM."""
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            "pyarrow is required to prepare ESCI; install requirements-bootstrap.txt"
        ) from exc

    parquet = pq.ParquetFile(source)
    if "product_locale" not in parquet.schema_arrow.names:
        raise ValueError(f"{source} has no product_locale column")

    writer = pq.ParquetWriter(
        destination,
        parquet.schema_arrow,
        compression="zstd",
        use_dictionary=True,
    )
    rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=65_536):
            table = pa.Table.from_batches([batch])
            filtered = table.filter(pc.equal(table["product_locale"], locale))
            if filtered.num_rows:
                writer.write_table(filtered)
                rows += filtered.num_rows
    finally:
        writer.close()
    return rows


def prepared_is_complete(path: Path, examples_name: str, products_name: str) -> bool:
    return all(
        (path / name).is_file()
        for name in (examples_name, products_name, METADATA_FILE)
    )


def prepare_esci(
    source: Path,
    destination: Path,
    locale: str,
    examples_name: str,
    products_name: str,
    source_revision: str,
) -> dict[str, Any]:
    """Create a versioned dataset folder, or safely reuse a complete one."""
    source_files = {
        "examples": source / examples_name,
        "products": source / products_name,
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ESCI source file(s): " + ", ".join(missing))

    if destination.exists():
        if not prepared_is_complete(destination, examples_name, products_name):
            raise RuntimeError(
                f"Prepared path exists but is incomplete: {destination}. "
                "Remove it or choose a new asset version/path."
            )
        with (destination / METADATA_FILE).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("locale") != locale:
            raise RuntimeError(
                f"Prepared path {destination} uses locale {metadata.get('locale')!r}, "
                f"not {locale!r}; create a new version."
            )
        if metadata.get("source_revision") != source_revision:
            raise RuntimeError(
                f"Prepared path {destination} uses source revision "
                f"{metadata.get('source_revision')!r}, not {source_revision!r}; "
                "create a new version."
            )
        print(f"Prepared ESCI data already exists: {destination}")
        return metadata

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        output_rows = {}
        for key, source_file in source_files.items():
            print(f"Filtering {source_file.name} to product_locale={locale!r} ...")
            output_rows[key] = filter_parquet(
                source_file, temporary / source_file.name, locale
            )

        metadata = {
            "format_version": 1,
            "locale": locale,
            "source_revision": source_revision,
            "source_files": {
                key: {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for key, path in source_files.items()
            },
            "rows": output_rows,
        }
        with (temporary / METADATA_FILE).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

        temporary.rename(destination)
        print(f"Prepared ESCI data: {destination}")
        print(f"  examples: {output_rows['examples']:,}")
        print(f"  products: {output_rows['products']:,}")
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", help="Override the ESCI source folder")
    parser.add_argument("--output", help="Override the prepared output folder")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    asset = load_config(config_path)["data"]["esci"]
    source = configured_source(asset, args.source)
    destination = project_path(args.output or str(asset["prepared_path"]))
    files = asset["files"]
    prepare_esci(
        source=source,
        destination=destination,
        locale=str(asset["locale"]),
        examples_name=str(files["examples"]),
        products_name=str(files["products"]),
        source_revision=str(asset.get("source_revision", "unknown")),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
