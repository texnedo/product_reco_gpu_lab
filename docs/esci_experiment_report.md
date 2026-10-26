# ESCI reranker experiment report

This report compares the unmodified Qwen3-Reranker baseline with the first
four LoRA experiments. All five archives are stored in `run_results/`.

## Executive summary

- Pairwise LoRA training works: all four trained runs improve every reported
  ranking metric over the archived base model.
- V4 has the highest recorded 1,000-query NDCG@10 at 0.854258, but 250 of
  those queries were used to select its checkpoint. On the untouched 750
  queries, v4 NDCG is 0.850545 versus 0.850373 for v1: +0.000172 with a paired
  95% interval of [-0.006370, +0.006651]. V4 is therefore tied with v1, not a
  demonstrated NDCG winner.
- V1 remains best on broad Precision@10 and Recall@10. V3 remains best on
  direct Precision@10 and Recall@10. V4 therefore gains ordering quality
  while giving back a small amount of top-ten recall.
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
- V4 development NDCG peaked at optimizer step 1,875 and declined afterward.
  Best-checkpoint selection prevented the second half of the 3,749-step run
  from degrading the exported model.

Retain v1 for broad top-ten retrieval and v3 when direct metrics or
`S>C`/`C>I` separation are important; v4 is not a new general incumbent on the
untouched holdout. V5 is prepared with the leakage-free 250-development /
750-final split and lowers v4's peak learning rate from `2e-4` to `1e-4`.
It deliberately keeps the p12 pair budget unchanged.

## Comparability

The five result archives contain exactly the same 1,000 evaluation query IDs,
20,255 `(query, product)` candidates, and ESCI labels. They also use the same
prompt hash, maximum sequence length, scoring function, graded relevance, and
seed. This makes the per-query comparisons paired rather than comparisons of
different evaluation samples.

The most controlled comparison is v1 versus v3. Both use 59,982 training
pairs, 1,875 optimizer steps, one epoch, learning rate `2e-4`, batch size 4,
gradient accumulation 8, and LoRA rank 16. Their intended difference is the
sampling distribution. V2 uses almost twice as many pairs and optimizer
steps, so it tests additional exposure rather than sampling alone.

V4 is comparable at the evaluation-query level but is not a single-variable
ablation. It expands training from 5,000 to 10,000 queries, applies `basic_v1`
HTML/whitespace cleaning to product text, and exports the best of seven
development checkpoints. Consequently, v4-versus-v1 intervals quantify the
observed system difference, but cannot attribute it separately to data breadth,
cleaning, or checkpoint selection. In addition, the reported 1,000-query v4
metric includes the 250 queries used for checkpoint selection. The untouched
750-query slice is the less biased comparison and should be treated as primary.
The archived base run also lacks v4 text cleaning; v4's step-zero development
measurement supplies a cleaned-input base reference only for the 250-query
development subset.

## Ranking results

| Run | NDCG@10 | Direct P@10 | Direct R@10 | Broad P@10 | Broad R@10 |
|---|---:|---:|---:|---:|---:|
| Base model | 0.828314 | 0.840239 | 0.613639 | 0.885161 | 0.604202 |
| v1: round robin, 12 pairs | 0.852150 | 0.859339 | 0.634114 | **0.895361** | **0.614750** |
| v2: round robin, 24 pairs | 0.843832 | 0.853039 | 0.626727 | 0.888961 | 0.608139 |
| v3: 50% balanced mix, 12 pairs | 0.851626 | **0.860539** | **0.634416** | 0.893261 | 0.612270 |
| v4: 10k queries, round robin, 12 pairs, cleaned, selected | **0.854258** | 0.858239 | 0.631054 | 0.894361 | 0.612414 |

Absolute changes relative to the base model:

| Run | NDCG@10 | Direct P@10 | Direct R@10 | Broad P@10 | Broad R@10 |
|---|---:|---:|---:|---:|---:|
| v1 | +0.023835 | +0.019100 | +0.020475 | +0.010200 | +0.010548 |
| v2 | +0.015518 | +0.012800 | +0.013089 | +0.003800 | +0.003937 |
| v3 | +0.023311 | +0.020300 | +0.020777 | +0.008100 | +0.008068 |
| v4 | **+0.025944** | +0.018000 | +0.017415 | +0.009200 | +0.008212 |

V3 changes relative to v1 are small: NDCG@10 is -0.000524, direct
Precision@10 is +0.001200, direct Recall@10 is +0.000302, broad Precision@10
is -0.002100, and broad Recall@10 is -0.002480.

V4 changes relative to v1 are also small: NDCG@10 is +0.002109, direct
Precision@10 is -0.001100, direct Recall@10 is -0.003060, broad Precision@10
is -0.001000, and broad Recall@10 is -0.002336.

### Untouched 750-query holdout

The table above reproduces each archive's official 1,000-query metric. For an
unbiased v4 comparison, remove the 250 query IDs used for development
selection and average the remaining 750 queries:

| Run | NDCG@10 | Direct P@10 | Direct R@10 | Broad P@10 | Broad R@10 |
|---|---:|---:|---:|---:|---:|
| Base model | 0.825072 | 0.839619 | 0.612030 | 0.887115 | 0.604226 |
| v1 | 0.850373 | 0.857219 | 0.630857 | **0.895648** | **0.612677** |
| v3 | **0.851807** | **0.858552** | **0.632132** | 0.893381 | 0.611026 |
| v4 | 0.850545 | 0.854819 | 0.626608 | 0.893381 | 0.609372 |

On this untouched slice, v4 is only +0.000172 NDCG over v1 and is lower on
all four precision/recall measures. The apparent full-set NDCG lead is
concentrated in the development subset, where v4 is +0.007918 over v1. This is
expected after selecting the checkpoint on that subset and is why future runs
need a disjoint final metric. V5 implements that protocol directly.

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

## Paired v4 versus v1 uncertainty

The same 20,000-resample paired bootstrap procedure was applied to v4 minus
v1. A positive value favors v4. The first table uses all 1,000 queries to match
the archived headline metric.

| Metric | Mean delta | Paired 95% interval | Queries better / worse / tied |
|---|---:|---:|---:|
| NDCG@10 | +0.002109 | [-0.003256, +0.007594] | 459 / 418 / 123 |
| Direct P@10 | -0.001100 | [-0.005400, +0.003200] | 123 / 138 / 739 |
| Direct R@10 | -0.003060 | [-0.008428, +0.002098] | 123 / 138 / 739 |
| Broad P@10 | -0.001000 | [-0.004600, +0.002700] | 94 / 107 / 799 |
| Broad R@10 | -0.002336 | [-0.006503, +0.001673] | 94 / 107 / 799 |

Every interval contains zero. More importantly, the disjoint 750-query holdout
removes the subset used for checkpoint selection:

| Metric | Mean delta | Paired 95% interval | Queries better / worse / tied |
|---|---:|---:|---:|
| NDCG@10 | +0.000172 | [-0.006370, +0.006651] | 333 / 326 / 91 |
| Direct P@10 | -0.002400 | [-0.007467, +0.002400] | 92 / 111 / 547 |
| Direct R@10 | -0.004249 | [-0.010737, +0.001893] | 92 / 111 / 547 |
| Broad P@10 | -0.002267 | [-0.006400, +0.001870] | 67 / 85 / 598 |
| Broad R@10 | -0.003305 | [-0.008398, +0.001280] | 67 / 85 / 598 |

V4 is clearly better than the base model on the full set—the v4-minus-base
NDCG interval is [+0.018777, +0.033105]—but it is not distinguishable from v1.
The untouched slice supports equivalence in NDCG with a small, uncertain
recall tradeoff.

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

| Boundary | Base | v1 | v2 | v3 | v4 | v4 - v1 | Eval pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E>S` | 0.6625 | 0.6949 | 0.6901 | 0.6920 | **0.7023** | +0.0074 | 58,401 |
| `E>C` | 0.6637 | 0.7760 | 0.7627 | 0.7763 | **0.7789** | +0.0029 | 8,910 |
| `E>I` | 0.7191 | 0.7568 | 0.7374 | 0.7597 | **0.7695** | +0.0127 | 31,014 |
| `S>C` | 0.5128 | 0.6940 | 0.6991 | **0.7215** | 0.6880 | -0.0060 | 4,337 |
| `S>I` | 0.6295 | 0.6462 | 0.6247 | **0.6508** | 0.6488 | +0.0026 | 17,619 |
| `C>I` | **0.6439** | 0.5572 | 0.5548 | 0.5918 | 0.5781 | +0.0209 | 2,918 |

The intended `S>C` and `C>I` improvements are visible, but neither
query-clustered 95% interval excludes zero (`S>C`: [-0.0041, +0.0598],
`C>I`: [-0.0003, +0.0725]). More importantly, v3 still does not recover the
base model's `C>I` ordering accuracy. This boundary needs more investigation
before increasing its sampling weight again.

V3 also returns, on average, 0.033 fewer `C` products and 0.021 more `I`
products in the top ten than v1. That is consistent with its small broad
metric decline, although the decline itself is inside the paired uncertainty
interval.

V4 strengthens all three `E` boundaries relative to v1, especially `E>I`,
which is consistent with its higher NDCG. It weakens `S>C` and remains well
below the base model on `C>I`. Its average top ten contains 5.590 `E`, 2.988
`S`, 0.361 `C`, and 1.052 `I` products, versus 5.557, 3.032, 0.360, and 1.042
for v1. In other words, v4 promotes slightly more exact products, not more
broadly relevant products overall; that explains the NDCG/recall tradeoff.

## Training diagnostics

| Run | Sampled pairs | Optimizer steps | Selected step | Mean RankNet loss | Training pair accuracy |
|---|---:|---:|---:|---:|---:|
| v1 | 59,982 | 1,875 | final | 0.423737 | 0.798740 |
| v2 | 117,691 | 3,678 | final | 0.345132 | 0.843191 |
| v3 | 59,982 | 1,875 | final | 0.448906 | 0.784669 |
| v4 | 119,959 | 3,749 | **1,875** | 0.428475 | 0.795905 |

V1 and v3 training loss and accuracy are not directly comparable because v3
contains many more scarce and difficult pair types. V2 is the useful warning:
better optimization metrics accompanied worse held-out ranking metrics. V4
reinforces that warning: its full-epoch training summary does not describe the
exported adapter, because development selection restored the step-1,875 state.

## V4 development trajectory

The same fixed 250-query development subset was evaluated throughout v4.

| Optimizer step | Development NDCG@10 | Direct R@10 | Broad R@10 | New best? |
|---:|---:|---:|---:|:---:|
| 0 | 0.837277 | 0.621038 | 0.606626 | yes |
| 625 | 0.856346 | 0.636226 | 0.613334 | yes |
| 1,250 | 0.859091 | 0.641178 | 0.617437 | yes |
| **1,875** | **0.864821** | 0.644393 | **0.621540** | **yes** |
| 2,500 | 0.854542 | **0.645039** | 0.620530 | no |
| 3,125 | 0.857729 | 0.641513 | 0.619039 | no |
| 3,749 | 0.855679 | 0.644100 | 0.619476 | no |

The peak occurs halfway through the epoch. By step 1,875 the shuffled prefix
had already touched 9,998 of 10,000 training queries, but only 82,302 of
201,110 products (40.9%). The rest of the epoch increased product exposure to
127,379 (63.3%) without improving development NDCG. This does not prove that
additional products are unhelpful; it shows that the current optimizer and
random-pair curriculum do not convert that added exposure into a gain.

## Pair budget and product coverage

The current setting is a **pair** budget, not an item budget. Products may be
reused in several pairs. The following deterministic simulation uses the same
10,000 v4 queries, seed, and round-robin sampler.

| Pairs/query cap | Realized pairs | Optimizer steps | Unique products touched | Product coverage | Mean items/query touched |
|---:|---:|---:|---:|---:|---:|
| 6 | 60,000 | 1,875 | 83,453 | 41.5% | 8.35 |
| 12 | 119,959 | 3,749 | 127,379 | 63.3% | 12.74 |
| 18 | 178,393 | 5,575 | 153,846 | 76.5% | 15.39 |
| 24 | 235,374 | 7,356 | 169,157 | 84.1% | 16.92 |

There is genuine unused product coverage at p12, but simply raising the cap
mostly adds the same dominant `E>S` and `E>I` distribution and greatly extends
training. V2 already showed that doubling random pairs can improve training
loss while harming held-out ranking. A better follow-up is a coverage-first or
hard-pair sampler that uses more distinct products before reusing one, while
holding the total pair/update budget fixed.

## Decision and next experiment

1. Do not promote v4 over v1/v3 based on the selected 1,000-query aggregate.
   On the untouched 750 queries it is effectively tied in NDCG and lower in
   recall. Continue to use v1 for broad retrieval and v3 when direct metrics or
   the `C` boundaries are the priority.
2. V5 now enforces a deterministic, disjoint evaluation protocol: 250 queries
   are used only for checkpoint selection and the remaining 750 are used only
   for final reporting. The job writes `evaluation_split.json` with both query
   ID sets and records zero overlap in `metrics.json`. `--eval-only` applies
   the identical split so its baseline remains directly comparable.
3. Run the prepared controlled learning-rate experiment next:
   `config/experiments/esci_v5_lr1e4.yaml` copies v4 and changes only peak
   learning rate from `2e-4` to `1e-4`. It retains p12, the same 10,000
   queries, cleaning, seed, schedule type, 250-query development subset, and
   checkpoint interval. This tests whether slower updates can benefit from the
   second half of product exposure instead of peaking at step 1,875.
4. If budget permits a small rate sweep rather than one run, prioritize
   `1e-4`, then `5e-5`; keep `2e-4` as the existing control. Do not combine the
   rate sweep with p18/p24, sampler balancing, or hard mining.
5. Do not raise random pairs per query yet. P18 would cost 5,575 optimizer
   steps and p24 7,356, while the current run already declined after 1,875.
   Best-checkpoint selection limits damage but does not make the extra compute
   informative.
6. After selecting a learning rate, test **coverage-first p12**: maximize
   distinct products per query before reusing products, while preserving
   cross-label validity and the same total pair count. If that helps, then
   consider p18. Hard-pair sampling based on close scores or current ordering
   errors is more promising than adding arbitrary pairs.

The learning-rate interpretation needs one caveat. V4 uses a linear schedule
over 3,749 steps; at the selected step 1,875, its instantaneous rate was about
`1.05e-4`. That observation motivates a `1e-4` peak-rate run, but does not prove
that `1e-4` is optimal—the selected checkpoint also reflects the amount and
order of data already consumed. Intermediate evaluation remains essential.

## Reproducibility notes

- Aggregate values come from each archive's `evaluation/metrics.json`.
- Paired intervals come from `evaluation/per_query_metrics.jsonl` with a fixed
  bootstrap seed.
- Boundary accuracy and top-ten label counts come from
  `evaluation/predictions.parquet`.
- The archived raw scores are sufficient to recompute every table without
  rerunning model inference.
