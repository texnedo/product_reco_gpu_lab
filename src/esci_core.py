"""Framework-free ESCI ranking rules, pair sampling, and evaluation metrics.

This module intentionally depends only on the Python standard library. Tests
and exploratory tools can import it without initializing PyTorch,
Transformers, PEFT, CUDA discovery, or Azure ML integrations.
"""

from __future__ import annotations

import math
import random
import re
from html import unescape
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
PAIR_TYPES = tuple(
    (higher_label, lower_label)
    for higher_position, higher_label in enumerate(LABEL_ORDER)
    for lower_label in LABEL_ORDER[higher_position + 1 :]
)
TEXT_CLEANING_MODES = frozenset(("none", "basic_v1"))

# Amazon descriptions commonly contain simple HTML fragments rather than
# meaningful markup. Block boundaries are converted to spaces before tags are
# removed so adjacent list items do not run together. This deliberately avoids
# aggressive language-specific normalization or fuzzy deduplication.
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_BLOCK_TAG_RE = re.compile(
    r"<(?:br|hr)\s*/?>|</(?:p|div|li|ul|ol|h[1-6]|tr|td|th)\s*>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<!--.*?-->|</?[A-Za-z][^>]*>", re.DOTALL)
_SEGMENT_BOUNDARY_RE = re.compile(r"\s*\n+\s*|(?<=[.!?])\s+(?=[A-Z0-9])")
_WHITESPACE_RE = re.compile(r"\s+")


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


def basic_clean_product_segments(value: Any) -> tuple[str, ...]:
    """Return HTML-free, whitespace-normalized segments without deduplication."""
    text = safe_text(value)
    if text == "[MISSING]":
        return (text,)

    # Decode twice because product feeds sometimes contain nested entities
    # such as &amp;nbsp;. A fixed two passes remains deterministic.
    text = unescape(unescape(text))
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)

    segments = tuple(
        segment
        for raw_segment in _SEGMENT_BOUNDARY_RE.split(text)
        if (segment := _WHITESPACE_RE.sub(" ", raw_segment).strip())
    )
    return segments if segments else ("[MISSING]",)


def clean_product_text(value: Any, mode: str = "none") -> str:
    """Apply a versioned, conservative cleanup to one product text field.

    ``basic_v1`` decodes HTML entities, drops markup and script/style blocks,
    and normalizes whitespace. It does not deduplicate sentences, cap fields,
    reorder them, stem words, or remove frequent terms. Those stronger choices
    should be evaluated as separate experiments.
    """
    if mode not in TEXT_CLEANING_MODES:
        raise ValueError(
            f"Unknown text cleaning mode {mode!r}; "
            f"expected one of {sorted(TEXT_CLEANING_MODES)}"
        )
    if mode == "none":
        return safe_text(value)
    return " ".join(basic_clean_product_segments(value))


def product_document(
    row: dict[str, Any], text_cleaning: str = "none"
) -> str:
    """Serialize all available product fields into Qwen's Document slot."""
    return (
        f"Title: {clean_product_text(row['product_title'], text_cleaning)}\n"
        f"Brand: {clean_product_text(row['product_brand'], text_cleaning)}\n"
        f"Color: {clean_product_text(row['product_color'], text_cleaning)}\n"
        "Bullet points: "
        f"{clean_product_text(row['product_bullet_point'], text_cleaning)}\n"
        "Description: "
        f"{clean_product_text(row['product_description'], text_cleaning)}"
    )


def _pairs_for_type(
    by_label: dict[str, list[int]],
    higher_label: str,
    lower_label: str,
) -> list[TrainingPair]:
    return [
        TrainingPair(higher_index, lower_index, higher_label, lower_label)
        for higher_index in by_label.get(higher_label, ())
        for lower_index in by_label.get(lower_label, ())
    ]


def _sample_query_round_robin(
    grouped: dict[Any, dict[str, list[int]]],
    max_pairs_per_query: int,
    rng: random.Random,
) -> list[TrainingPair]:
    """Apply the original type-round-robin budget independently per query."""
    sampled: list[TrainingPair] = []
    for by_label in grouped.values():
        candidates_by_type: list[list[TrainingPair]] = []
        for higher_label, lower_label in PAIR_TYPES:
            candidates = _pairs_for_type(by_label, higher_label, lower_label)
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


def _globally_balanced_pairs(
    grouped: dict[Any, dict[str, list[int]]],
    excluded_pairs: set[tuple[int, int]],
    requested_count: int,
    per_query_type_cap: int,
    rng: random.Random,
) -> list[TrainingPair]:
    """Draw an approximately uniform number of pairs from all six types.

    Each query can contribute at most ``per_query_type_cap`` candidates to one
    type's global pool. Without this cap, a query containing many E and I
    products could dominate that pool through its Cartesian product.
    """
    pools: dict[str, list[TrainingPair]] = {
        f"{higher}>{lower}": [] for higher, lower in PAIR_TYPES
    }
    for by_label in grouped.values():
        for higher_label, lower_label in PAIR_TYPES:
            candidates = [
                pair
                for pair in _pairs_for_type(
                    by_label, higher_label, lower_label
                )
                if (pair.higher_index, pair.lower_index) not in excluded_pairs
            ]
            rng.shuffle(candidates)
            pair_type = f"{higher_label}>{lower_label}"
            pools[pair_type].extend(candidates[:per_query_type_cap])

    pair_type_names = list(pools)
    for candidates in pools.values():
        rng.shuffle(candidates)

    quotient, remainder = divmod(requested_count, len(pair_type_names))
    selected: list[TrainingPair] = []
    offsets: dict[str, int] = {}
    for type_index, pair_type in enumerate(pair_type_names):
        quota = quotient + int(type_index < remainder)
        chosen = pools[pair_type][:quota]
        selected.extend(chosen)
        offsets[pair_type] = len(chosen)

    # A type can have fewer candidates than its equal quota. Redistribute that
    # shortfall across other types rather than silently shrinking the training
    # and optimizer-step budgets.
    while len(selected) < requested_count:
        made_progress = False
        for pair_type in pair_type_names:
            offset = offsets[pair_type]
            if offset < len(pools[pair_type]):
                selected.append(pools[pair_type][offset])
                offsets[pair_type] = offset + 1
                made_progress = True
                if len(selected) == requested_count:
                    break
        if not made_progress:
            available = len(selected)
            raise ValueError(
                "Balanced candidate pools contain only "
                f"{available:,} of {requested_count:,} requested pairs; "
                "increase balanced_pairs_per_type_per_query"
            )
    rng.shuffle(selected)
    return selected


def sample_training_pairs(
    query_ids: Sequence[Any],
    labels: Sequence[str],
    max_pairs_per_query: int,
    seed: int,
    sampling_strategy: str = "query_round_robin",
    balance_fraction: float = 0.0,
    balanced_pairs_per_type_per_query: int = 12,
) -> list[TrainingPair]:
    """Sample same-query, different-label comparisons.

    ``query_round_robin`` is the original sampler. For each query it shuffles
    candidate products within each available label-pair type, then cycles
    across those types. Its global distribution remains imbalanced because
    far more ESCI queries contain E/S than C/I.

    ``mixed_type_balanced`` retains a fraction of that per-query sample and
    fills the original total pair budget from global pair-type pools. The
    global component is divided equally among E>S, E>C, E>I, S>C, S>I, and
    C>I, subject to a per-query/type cap. A different epoch seed exposes
    different product combinations in both components.
    """
    if len(query_ids) != len(labels):
        raise ValueError("query_ids and labels lengths differ")
    if max_pairs_per_query <= 0:
        raise ValueError("max_pairs_per_query must be positive")
    if sampling_strategy not in ("query_round_robin", "mixed_type_balanced"):
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy!r}")
    if not 0.0 <= balance_fraction <= 1.0:
        raise ValueError("balance_fraction must be in [0, 1]")
    if balanced_pairs_per_type_per_query <= 0:
        raise ValueError("balanced_pairs_per_type_per_query must be positive")
    if sampling_strategy == "query_round_robin" and balance_fraction != 0.0:
        raise ValueError("query_round_robin requires balance_fraction=0")
    if sampling_strategy == "mixed_type_balanced" and balance_fraction <= 0.0:
        raise ValueError("mixed_type_balanced requires balance_fraction > 0")

    grouped: dict[Any, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (query_id, label) in enumerate(zip(query_ids, labels)):
        if label not in GRADE:
            raise ValueError(f"Unexpected ESCI label: {label!r}")
        grouped[query_id][label].append(index)

    full_sample = _sample_query_round_robin(
        grouped, max_pairs_per_query, random.Random(seed)
    )
    if sampling_strategy == "query_round_robin":
        return full_sample

    natural_budget = round(max_pairs_per_query * (1.0 - balance_fraction))
    natural_sample = _sample_query_round_robin(
        grouped, natural_budget, random.Random(seed)
    )
    balanced_count = len(full_sample) - len(natural_sample)
    excluded = {
        (pair.higher_index, pair.lower_index) for pair in natural_sample
    }
    balanced_sample = _globally_balanced_pairs(
        grouped,
        excluded,
        balanced_count,
        balanced_pairs_per_type_per_query,
        random.Random(seed + 104_729),
    )
    return [*natural_sample, *balanced_sample]


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
