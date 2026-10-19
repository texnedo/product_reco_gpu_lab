from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from esci_core import GRADE, ranking_metrics, sample_training_pairs  # noqa: E402


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
