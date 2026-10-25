# ESCI reranker experiment report

This report compares the unmodified Qwen3-Reranker baseline with the first
three LoRA experiments. All four archives are stored in `run_results/`.

## Executive summary

- Pairwise LoRA training works: both v1 and v3 improve NDCG@10 by about 0.023
  and direct Recall@10 by about 0.021 over the base model.
- v1 remains the best run for the primary NDCG@10 and broad-relevance
  metrics. v3 is marginally better on direct Precision@10 and Recall@10.
- The v3 balanced sampler is effectively tied with v1 at the query level.
  Every paired 95% bootstrap confidence interval for v3 minus v1 contains
  zero.
- The sampler did change what the model learned. Pair-weighted ordering
  accuracy improved by 2.74 percentage points for `S>C` and 3.46 points for
  `C>I`, the two boundaries most underrepresented in v1. These improvements
  are directional rather than statistically conclusive when queries are used
  as the sampling unit.
- Doubling the old sampler from 12 to 24 pairs per query reduced all ranking
  metrics relative to v1. Its lower training loss and higher training pair
  accuracy therefore did not generalize.

For now, retain v1 as the NDCG-oriented incumbent. Retain v3 when improved
coverage of the `C` boundaries is important. Do not change the learning rate
based on this comparison alone.

## Comparability

The four result archives contain exactly the same 1,000 evaluation query IDs,
20,255 `(query, product)` candidates, and ESCI labels. They also use the same
prompt hash, maximum sequence length, scoring function, graded relevance, and
seed. This makes the per-query comparisons paired rather than comparisons of
different evaluation samples.

The most controlled comparison is v1 versus v3. Both use 59,982 training
pairs, 1,875 optimizer steps, one epoch, learning rate `2e-4`, batch size 4,
gradient accumulation 8, and LoRA rank 16. Their intended difference is the
sampling distribution. V2 uses almost twice as many pairs and optimizer
steps, so it tests additional exposure rather than sampling alone.

## Ranking results

| Run | NDCG@10 | Direct P@10 | Direct R@10 | Broad P@10 | Broad R@10 |
|---|---:|---:|---:|---:|---:|
| Base model | 0.828314 | 0.840239 | 0.613639 | 0.885161 | 0.604202 |
| v1: round robin, 12 pairs | **0.852150** | 0.859339 | 0.634114 | **0.895361** | **0.614750** |
| v2: round robin, 24 pairs | 0.843832 | 0.853039 | 0.626727 | 0.888961 | 0.608139 |
| v3: 50% balanced mix, 12 pairs | 0.851626 | **0.860539** | **0.634416** | 0.893261 | 0.612270 |

Absolute changes relative to the base model:

| Run | NDCG@10 | Direct P@10 | Direct R@10 | Broad P@10 | Broad R@10 |
|---|---:|---:|---:|---:|---:|
| v1 | +0.023835 | +0.019100 | +0.020475 | +0.010200 | +0.010548 |
| v2 | +0.015518 | +0.012800 | +0.013089 | +0.003800 | +0.003937 |
| v3 | +0.023311 | +0.020300 | +0.020777 | +0.008100 | +0.008068 |

V3 changes relative to v1 are small: NDCG@10 is -0.000524, direct
Precision@10 is +0.001200, direct Recall@10 is +0.000302, broad Precision@10
is -0.002100, and broad Recall@10 is -0.002480.

## Paired v3 versus v1 uncertainty

The following intervals come from 20,000 paired bootstrap resamples of the
1,000 evaluation queries. A positive value favors v3.

| Metric | Mean delta | Paired 95% interval | Queries better / worse / tied |
|---|---:|---:|---:|
| NDCG@10 | -0.000524 | [-0.005843, +0.004742] | 428 / 444 / 128 |
| Direct P@10 | +0.001200 | [-0.002900, +0.005300] | 124 / 105 / 771 |
| Direct R@10 | +0.000302 | [-0.004548, +0.005246] | 124 / 105 / 771 |
| Broad P@10 | -0.002100 | [-0.005500, +0.001300] | 83 / 92 / 825 |
| Broad R@10 | -0.002480 | [-0.006698, +0.001817] | 83 / 92 / 825 |

These intervals measure uncertainty across the fixed evaluation queries. They
do not measure variation from GPU kernels, initialization, or a different
training sample. Therefore the safest conclusion is equivalence at the
current resolution, not proof that both samplers are identical.

## What balancing changed

V3 kept the total pair and optimizer-step budgets fixed while redistributing
comparisons toward the rare ESCI boundaries.

| Pair type | v1 count | v1 share | v3 count | v3 share | Count change |
|---|---:|---:|---:|---:|---:|
| `E>S` | 29,066 | 48.46% | 19,478 | 32.47% | -33.0% |
| `E>C` | 4,693 | 7.82% | 7,348 | 12.25% | +56.6% |
| `E>I` | 13,852 | 23.09% | 11,919 | 19.87% | -14.0% |
| `S>C` | 3,061 | 5.10% | 6,546 | 10.91% | +113.9% |
| `S>I` | 7,578 | 12.63% | 8,795 | 14.66% | +16.1% |
| `C>I` | 1,732 | 2.89% | 5,896 | 9.83% | +240.4% |

For the diagnostic below, every cross-label product pair within an evaluation
query is checked for the desired score order. The result is pair-weighted, so
queries with more applicable candidates contribute more comparisons.

| Boundary | Base | v1 | v2 | v3 | v3 - v1 | Eval pairs |
|---|---:|---:|---:|---:|---:|---:|
| `E>S` | 0.6625 | **0.6949** | 0.6901 | 0.6920 | -0.0028 | 58,401 |
| `E>C` | 0.6637 | 0.7760 | 0.7627 | **0.7763** | +0.0003 | 8,910 |
| `E>I` | 0.7191 | 0.7568 | 0.7374 | **0.7597** | +0.0029 | 31,014 |
| `S>C` | 0.5128 | 0.6940 | 0.6991 | **0.7215** | +0.0274 | 4,337 |
| `S>I` | 0.6295 | 0.6462 | 0.6247 | **0.6508** | +0.0046 | 17,619 |
| `C>I` | **0.6439** | 0.5572 | 0.5548 | 0.5918 | +0.0346 | 2,918 |

The intended `S>C` and `C>I` improvements are visible, but neither
query-clustered 95% interval excludes zero (`S>C`: [-0.0041, +0.0598],
`C>I`: [-0.0003, +0.0725]). More importantly, v3 still does not recover the
base model's `C>I` ordering accuracy. This boundary needs more investigation
before increasing its sampling weight again.

V3 also returns, on average, 0.033 fewer `C` products and 0.021 more `I`
products in the top ten than v1. That is consistent with its small broad
metric decline, although the decline itself is inside the paired uncertainty
interval.

## Training diagnostics

| Run | Sampled pairs | Optimizer steps | Mean RankNet loss | Training pair accuracy |
|---|---:|---:|---:|---:|
| v1 | 59,982 | 1,875 | 0.423737 | 0.798740 |
| v2 | 117,691 | 3,678 | 0.345132 | 0.843191 |
| v3 | 59,982 | 1,875 | 0.448906 | 0.784669 |

V1 and v3 training loss and accuracy are not directly comparable because v3
contains many more scarce and difficult pair types. V2 is the useful warning:
better optimization metrics accompanied worse held-out ranking metrics. More
updates under the original sampling distribution are not the next lever to
pull.

## Decision and next experiment

1. Use v1 as the current checkpoint when the primary objective is NDCG@10 or
   broad top-10 retrieval.
2. Treat v3 as evidence that sampler balance can improve underrepresented
   boundaries without destroying overall ranking, but not as a new overall
   winner.
3. The next run, v4, returns to v1 round-robin sampling and expands from 5,000
   to 10,000 training queries while retaining 12 pairs per query. It therefore
   tests whether additional query diversity makes better use of an update
   budget comparable to v2. It also applies the conservative `basic_v1` text
   cleanup before truncation: HTML/entities are removed and whitespace is
   normalized, while repeated sentences, field order, per-field allocation,
   and `max_length=512` remain unchanged.
4. Evaluate a stable 250-query development subset during v4 and export the
   checkpoint with the highest development NDCG@10 rather than assuming the
   final optimizer step is best. This protects against the late degradation
   suggested by v2.
5. Keep learning rate, LoRA rank, batch size, gradient accumulation, epoch
   count, and pair weighting unchanged. Use `S>C`/`C>I` ordering as secondary
   diagnostics, not model-selection targets.

A 25% balanced mix remains a useful later dose-response experiment, but it
should not be combined with the query-count change. Changing the learning rate
or hard-mining policy at the same time would likewise confound the v4 result.
Because v4 deliberately combines broader query coverage with conservative text
cleaning, it should be treated as the candidate run rather than a clean causal
ablation of either change. If it improves, evaluate `basic_v1` separately on a
matched 5,000-query run before attributing the gain.

## Reproducibility notes

- Aggregate values come from each archive's `evaluation/metrics.json`.
- Paired intervals come from `evaluation/per_query_metrics.jsonl` with a fixed
  bootstrap seed.
- Boundary accuracy and top-ten label counts come from
  `evaluation/predictions.parquet`.
- The archived raw scores are sufficient to recompute every table without
  rerunning model inference.
