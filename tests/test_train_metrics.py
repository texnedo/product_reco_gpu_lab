from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from esci_core import (  # noqa: E402
    GRADE,
    clean_product_text,
    development_query_ids,
    product_document,
    ranking_metrics,
    sample_training_pairs,
)


class ProductTextCleaningTests(unittest.TestCase):
    def test_basic_v1_removes_html_entities_scripts_and_normalizes_space(self) -> None:
        raw = (
            "<p>Strong&nbsp;steel   &amp; aluminum.</p>"
            "<script>ignore()</script><br>Fits model X."
        )

        self.assertEqual(
            clean_product_text(raw, "basic_v1"),
            "Strong steel & aluminum. Fits model X.",
        )

    def test_basic_v1_retains_repeated_values_inside_a_distinct_segment(self) -> None:
        raw = "Includes 2 x 2-inch filters, size < 5 > 3, and fits A, A1, and A2."

        self.assertEqual(clean_product_text(raw, "basic_v1"), raw)
        self.assertEqual(clean_product_text(None, "basic_v1"), "[MISSING]")

    def test_basic_v1_does_not_remove_repeated_sentences(self) -> None:
        raw = "<p>Fast.</p><p>Fast.</p>"

        self.assertEqual(clean_product_text(raw, "basic_v1"), "Fast. Fast.")

    def test_product_document_preserves_legacy_mode_and_cleans_when_enabled(self) -> None:
        row = {
            "product_title": "<b>Widget</b>",
            "product_brand": "Acme",
            "product_color": None,
            "product_bullet_point": "<p>Fast.</p><p>Fast.</p>",
            "product_description": "Useful &amp; sturdy.",
        }

        self.assertIn("<b>Widget</b>", product_document(row, "none"))
        cleaned = product_document(row, "basic_v1")
        self.assertIn("Title: Widget", cleaned)
        self.assertIn("Bullet points: Fast. Fast.", cleaned)
        self.assertIn("Description: Useful & sturdy.", cleaned)

    def test_unknown_cleaning_mode_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown text cleaning mode"):
            clean_product_text("text", "future_mode")


class EvaluationSplitTests(unittest.TestCase):
    def test_development_ids_are_stable_disjoint_and_order_independent(self) -> None:
        query_ids = list(range(1000))
        first = development_query_ids(query_ids, 250, seed=42)
        repeated = development_query_ids(reversed(query_ids), 250, seed=42)
        holdout = set(query_ids) - set(first)

        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 250)
        self.assertEqual(len(holdout), 750)
        self.assertFalse(set(first) & holdout)

    def test_development_ids_must_leave_a_final_holdout(self) -> None:
        with self.assertRaisesRegex(ValueError, "holdout"):
            development_query_ids(range(10), 10, seed=42)


class PairSamplingTests(unittest.TestCase):
    def test_pairs_follow_esci_order_and_never_compare_equal_labels(self) -> None:
        query_ids = [1, 1, 1, 1, 1]
        labels = ["E", "E", "S", "C", "I"]
        pairs = sample_training_pairs(query_ids, labels, 20, seed=7)

        self.assertEqual(len(pairs), 9)
        for pair in pairs:
            self.assertGreater(GRADE[pair.higher_label], GRADE[pair.lower_label])
            self.assertNotEqual(pair.higher_label, pair.lower_label)
            self.assertEqual(query_ids[pair.higher_index], query_ids[pair.lower_index])

    def test_pair_budget_is_per_query_and_sampling_is_reproducible(self) -> None:
        query_ids = [1, 1, 1, 1, 2, 2, 2, 2]
        labels = ["E", "S", "C", "I"] * 2
        first = sample_training_pairs(query_ids, labels, 3, seed=42)
        second = sample_training_pairs(query_ids, labels, 3, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        counts = {1: 0, 2: 0}
        for pair in first:
            counts[query_ids[pair.higher_index]] += 1
        self.assertEqual(counts, {1: 3, 2: 3})

    def test_query_with_one_label_produces_no_pairs(self) -> None:
        pairs = sample_training_pairs([1, 1, 2, 2], ["E", "E", "I", "I"], 4, 1)
        self.assertEqual(pairs, [])

    def test_mixed_sampler_preserves_budget_and_reduces_global_imbalance(self) -> None:
        query_ids: list[int] = []
        labels: list[str] = []
        # Most queries only support E>S, reproducing the structural imbalance
        # in ESCI. A smaller group supports all six pair types.
        for query_id in range(10):
            query_ids.extend([query_id] * 4)
            labels.extend(["E", "E", "S", "S"])
        for query_id in range(10, 14):
            query_ids.extend([query_id] * 8)
            labels.extend(["E", "E", "S", "S", "C", "C", "I", "I"])

        natural = sample_training_pairs(query_ids, labels, 6, seed=42)
        mixed = sample_training_pairs(
            query_ids,
            labels,
            6,
            seed=42,
            sampling_strategy="mixed_type_balanced",
            balance_fraction=0.5,
            balanced_pairs_per_type_per_query=4,
        )
        repeated = sample_training_pairs(
            query_ids,
            labels,
            6,
            seed=42,
            sampling_strategy="mixed_type_balanced",
            balance_fraction=0.5,
            balanced_pairs_per_type_per_query=4,
        )

        self.assertEqual(mixed, repeated)
        self.assertEqual(len(mixed), len(natural))
        self.assertEqual(
            len({(pair.higher_index, pair.lower_index) for pair in mixed}),
            len(mixed),
        )
        for pair in mixed:
            self.assertEqual(
                query_ids[pair.higher_index], query_ids[pair.lower_index]
            )

        natural_counts = Counter(pair.pair_type for pair in natural)
        mixed_counts = Counter(pair.pair_type for pair in mixed)
        natural_spread = max(natural_counts.values()) - min(natural_counts.values())
        mixed_spread = max(mixed_counts.values()) - min(mixed_counts.values())
        self.assertLess(mixed_spread, natural_spread)


class RankingMetricsTests(unittest.TestCase):
    def test_perfect_order_and_direct_broad_definitions(self) -> None:
        summary, per_query = ranking_metrics(
            [1, 1, 1, 1],
            ["E", "S", "C", "I"],
            [4.0, 3.0, 2.0, 1.0],
            k=2,
        )
        self.assertAlmostEqual(summary["ndcg@2"], 1.0)
        self.assertAlmostEqual(summary["direct_precision@2"], 1.0)
        self.assertAlmostEqual(summary["direct_recall@2"], 1.0)
        self.assertAlmostEqual(summary["broad_precision@2"], 1.0)
        self.assertAlmostEqual(summary["broad_recall@2"], 2 / 3)
        self.assertAlmostEqual(per_query[0]["ndcg@2"], 1.0)

    def test_metrics_are_averaged_per_query(self) -> None:
        summary, per_query = ranking_metrics(
            [1, 1, 2, 2],
            ["E", "I", "E", "I"],
            [1.0, 0.0, 0.0, 1.0],
            k=2,
        )
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(len(per_query), 2)
        self.assertGreater(summary["ndcg@2"], 0.5)
        self.assertLess(summary["ndcg@2"], 1.0)
        self.assertEqual(summary["direct_precision@2"], 0.5)
        self.assertEqual(summary["direct_recall@2"], 1.0)

    def test_undefined_recall_is_excluded_from_mean(self) -> None:
        summary, per_query = ranking_metrics(
            [1, 1, 2, 2],
            ["I", "I", "E", "I"],
            [2.0, 1.0, 2.0, 1.0],
            k=1,
        )
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["direct_recall_eligible_query_count"], 1)
        self.assertEqual(summary["ndcg_eligible_query_count"], 1)
        self.assertIsNone(per_query[0]["direct_recall@1"])
        self.assertIsNone(per_query[0]["ndcg@1"])


if __name__ == "__main__":
    unittest.main()
