from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_config import (  # noqa: E402
    apply_overrides,
    generated_display_name,
    load_experiment_config,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_all_checked_in_configs_are_valid(self) -> None:
        configs = sorted((PROJECT_ROOT / "config/experiments").glob("*.yaml"))
        self.assertEqual(len(configs), 4)
        for path in configs:
            with self.subTest(path=path.name):
                self.assertEqual(load_experiment_config(path).source_path, path)

    def test_loads_balanced_config_and_generates_explanatory_name(self) -> None:
        config = load_experiment_config(
            PROJECT_ROOT / "config/experiments/esci_v3_balanced.yaml"
        )

        self.assertEqual(config.sampling_strategy, "mixed_type_balanced")
        self.assertEqual(config.balance_fraction, 0.5)
        self.assertEqual(config.pairs_per_query, 12)
        self.assertEqual(
            generated_display_name(config, eval_only=False),
            "qwen-esci-v3-balanced-mix50-p12-e1-b4-ga8-lr2e-4-r16-s42",
        )

    def test_explicit_overrides_are_validated(self) -> None:
        config = load_experiment_config(
            PROJECT_ROOT / "config/experiments/esci_v1_12_pairs.yaml"
        )
        balanced = apply_overrides(
            config,
            {
                "sampling_strategy": "mixed_type_balanced",
                "balance_fraction": 0.5,
            },
        )
        self.assertEqual(balanced.balance_fraction, 0.5)

        with self.assertRaisesRegex(ValueError, "requires.*balance_fraction"):
            apply_overrides(
                config,
                {"sampling_strategy": "mixed_type_balanced"},
            )

    def test_v4_enables_periodic_best_model_selection(self) -> None:
        config = load_experiment_config(
            PROJECT_ROOT / "config/experiments/esci_v4_q10k_p12.yaml"
        )
        self.assertEqual(config.train_queries, 10_000)
        self.assertEqual(config.pairs_per_query, 12)
        self.assertEqual(config.selection_queries, 250)
        self.assertEqual(config.selection_every_steps, 625)
        self.assertEqual(config.text_cleaning, "basic_v1")
        self.assertEqual(
            generated_display_name(config, eval_only=False),
            "qwen-esci-v4-q10k-rr-p12-e1-b4-ga8-lr2e-4-r16-s42-clean1",
        )

        with self.assertRaisesRegex(ValueError, "must both be positive"):
            apply_overrides(config, {"selection_every_steps": 0})

    def test_rejects_incomplete_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("schema_version: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing experiment"):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
