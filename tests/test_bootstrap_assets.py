from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "infra"))

from bootstrap_assets import (  # noqa: E402
    asset_files,
    fingerprint_folder,
    upload_path,
    validate_asset_name,
)
from prepare_esci import prepare_esci  # noqa: E402


class PrepareEsciTests(unittest.TestCase):
    def test_filters_both_tables_and_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()

            pq.write_table(
                pa.table(
                    {
                        "query": ["shoes", "zapatos", "hat"],
                        "product_locale": ["us", "es", "us"],
                    }
                ),
                source / "examples.parquet",
            )
            pq.write_table(
                pa.table(
                    {
                        "product_id": ["a", "b", "c"],
                        "product_locale": ["us", "jp", "us"],
                    }
                ),
                source / "products.parquet",
            )

            metadata = prepare_esci(
                source,
                output,
                "us",
                "examples.parquet",
                "products.parquet",
                "test-revision",
            )

            self.assertEqual(metadata["rows"], {"examples": 2, "products": 2})
            self.assertEqual(
                pq.read_table(output / "examples.parquet")["product_locale"].to_pylist(),
                ["us", "us"],
            )
            written = json.loads((output / "dataset_info.json").read_text())
            self.assertEqual(written["source_revision"], "test-revision")


class UploadStagingTests(unittest.TestCase):
    def test_validates_aml_asset_names_before_upload(self) -> None:
        validate_asset_name("qwen3-reranker-0-6b")
        with self.assertRaisesRegex(ValueError, "Invalid AML asset name"):
            validate_asset_name("qwen3-reranker-0.6b")

    def test_fingerprint_and_staging_exclude_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            (model / ".git").mkdir(parents=True)
            (model / ".git" / "large-object").write_bytes(b"x" * 100)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "weights").write_bytes(b"abc")

            fingerprint = fingerprint_folder(model, [".git"])
            self.assertEqual(fingerprint.file_count, 2)
            self.assertEqual(fingerprint.size_bytes, 5)

            with upload_path(model, [".git"]) as staged:
                included = [str(relative) for _, relative in asset_files(staged)]
                self.assertEqual(included, ["config.json", "weights"])


if __name__ == "__main__":
    unittest.main()
