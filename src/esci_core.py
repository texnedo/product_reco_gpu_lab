"""Framework-free ESCI ranking rules, pair sampling, and evaluation metrics.

This module intentionally depends only on the Python standard library. Tests
and exploratory tools can import it without initializing PyTorch,
Transformers, PEFT, CUDA discovery, or Azure ML integrations.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


# These grades express the only ordering assumption made by training.
# Equal-label pairs are deliberately omitted: ESCI tells us that E beats S,
# for example, but gives no evidence that one E product beats another E product.
LABEL_ORDER = ("E", "S", "C", "I")
GRADE = {"E": 3, "S": 2, "C": 1, "I": 0}
DIRECT_RELEVANT = frozenset(("E", "S"))
BROAD_RELEVANT = frozenset(("E", "S", "C"))


@dataclass(frozen=True)
class TrainingPair:
    """Indices of two rows for the same query, ordered higher then lower."""

    higher_index: int
    lower_index: int
    higher_label: str
    lower_label: str

    @property
    def grade_gap(self) -> int:
        return GRADE[self.higher_label] - GRADE[self.lower_label]

    @property
    def pair_type(self) -> str:
        return f"{self.higher_label}>{self.lower_label}"


def safe_text(value: Any) -> str:
    """Turn nullable dataframe values into stable prompt text."""
    if value is None:
        return "[MISSING]"
    try:
        if value != value:  # NaN is the unusual value not equal to itself.
            return "[MISSING]"
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else "[MISSING]"


def product_document(row: dict[str, Any]) -> str:
    """Serialize all available product fields into Qwen's Document slot."""
    return (
        f"Title: {safe_text(row['product_title'])}\n"
        f"Brand: {safe_text(row['product_brand'])}\n"
        f"Color: {safe_text(row['product_color'])}\n"
        f"Bullet points: {safe_text(row['product_bullet_point'])}\n"
        f"Description: {safe_text(row['product_description'])}"
    )


def sample_training_pairs(
    query_ids: Sequence[Any],
    labels: Sequence[str],
    max_pairs_per_query: int,
    seed: int,
) -> list[TrainingPair]:
    """Sample same-query, different-label comparisons.

    A query containing E/S/C/I has six pair types. We shuffle candidates
    within each type, then draw round-robin across types. This keeps numerous
    easy E>I combinations from crowding subtle adjacent comparisons out of a
    small pair budget. A different epoch seed exposes different products.
    """
    if len(query_ids) != len(labels):
        raise ValueError("query_ids and labels lengths differ")
    if max_pairs_per_query <= 0:
        raise ValueError("max_pairs_per_query must be positive")

    grouped: dict[Any, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (query_id, label) in enumerate(zip(query_ids, labels)):
        if label not in GRADE:
            raise ValueError(f"Unexpected ESCI label: {label!r}")
        grouped[query_id][label].append(index)

    rng = random.Random(seed)
    sampled: list[TrainingPair] = []
    for by_label in grouped.values():
        candidates_by_type: list[list[TrainingPair]] = []
        for higher_position, higher_label in enumerate(LABEL_ORDER):
            for lower_label in LABEL_ORDER[higher_position + 1 :]:
                candidates = [
                    TrainingPair(higher_index, lower_index, higher_label, lower_label)
                    for higher_index in by_label.get(higher_label, ())
                    for lower_index in by_label.get(lower_label, ())
                ]
                if candidates:
                    rng.shuffle(candidates)
                    candidates_by_type.append(candidates)

        rng.shuffle(candidates_by_type)
        query_pairs: list[TrainingPair] = []
        while candidates_by_type and len(query_pairs) < max_pairs_per_query:
            made_progress = False
            for candidates in candidates_by_type:
                if candidates and len(query_pairs) < max_pairs_per_query:
                    query_pairs.append(candidates.pop())
                    made_progress = True
            if not made_progress:
                break
        sampled.extend(query_pairs)
    return sampled


def dcg(relevances: Iterable[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(position + 2)
        for position, relevance in enumerate(relevances)
    )


def mean_or_zero(values: Sequence[float]) -> float:
    """Average a metric, returning zero when no query has a denominator."""
    return sum(values) / len(values) if values else 0.0


def ranking_metrics(
    query_ids: Sequence[Any],
    gold: Sequence[str],
    scores: Sequence[float],
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute query-level NDCG and direct/broad precision and recall.

    direct: E and S are relevant because either can fulfill the query.
    broad: E, S, and C are relevant, including complementary products.
    NDCG keeps all four grades, rewarding E>S>C>I within the top results.
    """
    if not (len(query_ids) == len(gold) == len(scores)):
        raise ValueError("ranking input lengths differ")
    if k <= 0:
        raise ValueError("k must be positive")

    grouped: dict[Any, list[tuple[str, float]]] = defaultdict(list)
    for query_id, label, score in zip(query_ids, gold, scores):
        grouped[query_id].append((label, float(score)))

    per_query: list[dict[str, Any]] = []
    aggregates: dict[str, list[float]] = defaultdict(list)
    for query_id, candidates in grouped.items():
        predicted = sorted(candidates, key=lambda item: item[1], reverse=True)
        top = predicted[:k]
        row: dict[str, Any] = {
            "query_id": int(query_id),
            "candidate_count": len(candidates),
        }
        for name, relevant_labels in (
            ("direct", DIRECT_RELEVANT),
            ("broad", BROAD_RELEVANT),
        ):
            retrieved = sum(label in relevant_labels for label, _ in top)
            total = sum(label in relevant_labels for label, _ in candidates)
            precision = retrieved / len(top) if top else 0.0
            recall = retrieved / total if total else None
            row[f"{name}_precision@{k}"] = precision
            row[f"{name}_recall@{k}"] = recall
            row[f"{name}_relevant_candidate_count"] = total
            aggregates[f"{name}_precision@{k}"].append(precision)
            if recall is not None:
                aggregates[f"{name}_recall@{k}"].append(recall)

        predicted_grades = [GRADE[label] for label, _ in top]
        ideal_grades = sorted(
            (GRADE[label] for label, _ in candidates), reverse=True
        )[:k]
        ideal_dcg = dcg(ideal_grades)
        ndcg = dcg(predicted_grades) / ideal_dcg if ideal_dcg else None
        row[f"ndcg@{k}"] = ndcg
        if ndcg is not None:
            aggregates[f"ndcg@{k}"].append(ndcg)
        per_query.append(row)

    summary = {
        f"ndcg@{k}": mean_or_zero(aggregates[f"ndcg@{k}"]),
        f"direct_precision@{k}": mean_or_zero(
            aggregates[f"direct_precision@{k}"]
        ),
        f"direct_recall@{k}": mean_or_zero(aggregates[f"direct_recall@{k}"]),
        f"broad_precision@{k}": mean_or_zero(
            aggregates[f"broad_precision@{k}"]
        ),
        f"broad_recall@{k}": mean_or_zero(aggregates[f"broad_recall@{k}"]),
        "query_count": len(grouped),
        "ndcg_eligible_query_count": len(aggregates[f"ndcg@{k}"]),
        "direct_recall_eligible_query_count": len(
            aggregates[f"direct_recall@{k}"]
        ),
        "broad_recall_eligible_query_count": len(
            aggregates[f"broad_recall@{k}"]
        ),
    }
    return summary, per_query


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Turn nested scalar dictionaries into MLflow-compatible metric names."""
    flattened = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)):
            flattened[name] = float(value)
    return flattened
