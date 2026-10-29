from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import asdict
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
        self.assertEqual(len(configs), 6)
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
            "qwen-esci-v4-q10k-rr-p12-e1-b4-ga8-lr2e-4-r16-s42-clean1-dev250-h750",
        )

        with self.assertRaisesRegex(ValueError, "must both be positive"):
            apply_overrides(config, {"selection_every_steps": 0})

    def test_v5_changes_only_the_intended_learning_rate(self) -> None:
        v4 = load_experiment_config(
            PROJECT_ROOT / "config/experiments/esci_v4_q10k_p12.yaml"
        )
        v5 = load_experiment_config(
            PROJECT_ROOT / "config/experiments/esci_v5_lr1e4.yaml"
        )

        self.assertEqual(v5.learning_rate, 1e-4)
        self.assertEqual(v5.pairs_per_query, 12)
        self.assertEqual(v5.train_queries, 10_000)
        self.assertEqual(v5.selection_queries, 250)
        self.assertEqual(v5.eval_queries - v5.selection_queries, 750)
        self.assertEqual(v5.text_cleaning, v4.text_cleaning)
        v4_comparable = asdict(v4)
        v5_comparable = asdict(v5)
        for key in ("source_path", "source_sha256", "run_label", "learning_rate"):
            v4_comparable.pop(key)
            v5_comparable.pop(key)
        self.assertEqual(v5_comparable, v4_comparable)
        self.assertEqual(
            generated_display_name(v5, eval_only=False),
            "qwen-esci-v5-q10k-rr-p12-e1-b4-ga8-lr1e-4-r16-s42-clean1-dev250-h750",
        )
        self.assertEqual(
            generated_display_name(v5, eval_only=True),
            "qwen-esci-v5-q10k-base-eval750-dev250-ml512-s42-clean1",
        )

    def test_v6_trades_pair_depth_for_query_breadth_at_fixed_budget(self) -> None:
        v5 = load_experiment_config(
            PROJECT_ROOT / "config/experiments/esci_v5_lr1e4.yaml"
        )
        v6 = load_experiment_config(
            PROJECT_ROOT / "config/experiments/esci_v6_q20k_p6.yaml"
        )

        self.assertEqual(v6.train_queries, 20_000)
        self.assertEqual(v6.pairs_per_query, 6)
        self.assertEqual(
            v6.train_queries * v6.pairs_per_query,
            v5.train_queries * v5.pairs_per_query,
        )
        v5_comparable = asdict(v5)
        v6_comparable = asdict(v6)
        for key in (
            "source_path",
            "source_sha256",
            "run_label",
            "train_queries",
            "pairs_per_query",
        ):
            v5_comparable.pop(key)
            v6_comparable.pop(key)
        self.assertEqual(v6_comparable, v5_comparable)
        self.assertEqual(
            generated_display_name(v6, eval_only=False),
            "qwen-esci-v6-q20k-rr-p6-e1-b4-ga8-lr1e-4-r16-s42-clean1-dev250-h750",
        )

    def test_rejects_incomplete_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("schema_version: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing experiment"):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
