# ESCI reranker experiment report

This report compares the unmodified Qwen3-Reranker baseline with the first six
LoRA experiments. All seven archives are stored in `run_results/`.

## Executive summary

- Pairwise LoRA training works: every trained run improves NDCG@10 over the
  unmodified base model on the common holdout.
- V6 is the new incumbent on all five 750-query holdout metrics. Its NDCG@10
  is 0.861641, versus 0.854425 for v5 and 0.850373 for v1.
- V6 minus v5 NDCG@10 is +0.007216 with a paired 95% interval of
  [+0.001716, +0.012530]. Direct Precision@10 and Recall@10 also improve with
  intervals entirely above zero. The smaller broad-metric gains remain
  uncertain.
- V6 holds the pair/update budget fixed: 20,000 queries x 6 pairs produces
  120,000 pairs and 3,750 optimizer steps, versus v5's 10,000 x 12 setup with
  119,959 pairs and 3,749 steps. The gain therefore supports query breadth and
  candidate diversity rather than simply more optimization.
- All six ESCI boundary accuracies improve from v5 to v6. `E>C` rises by 3.81
  percentage points and `S>C` by 5.17 points; their query-clustered intervals
  exclude zero.
- V6 has the worst full-epoch training loss and pair accuracy among the recent
  runs, yet the best holdout ranking. Training-pair optimization remains an
  unreliable model-selection metric.
- Best-checkpoint selection chose step 2,500. Four AML retries resumed from
  durable checkpoints and preserved the best-model state, validating recovery
  under repeated preemption.

Promote v6 as the current general ranking incumbent. Its advantage over v5 is
supported on NDCG and direct relevance, while broad relevance is directionally
better but not statistically resolved at 750 queries.

## Comparability

All seven result archives draw from the same 1,000 evaluation query IDs,
20,255 `(query, product)` candidates, and ESCI labels. V5 and v6 natively use
the same deterministic 250-query development set and disjoint 750-query final
holdout, with zero overlap. Older archives can be restricted to those exact
750 IDs for paired comparison. All runs also use the same prompt hash, maximum
sequence length, scoring function, graded relevance, and seed.

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

V5 versus v4 primarily tests peak learning rate `1e-4` versus `2e-4`; both use
10,000 queries, p12, cleaning, and the same development IDs. V6 versus v5 is
the cleanest recent data-breadth comparison: v6 doubles selected training
queries from 10,000 to 20,000 and halves pairs per query from 12 to 6. The
realized pair and update budgets, model settings, learning rate, pair-type
distribution, evaluation IDs, and checkpoint cadence remain effectively
fixed.

## Ranking results

The first table preserves the historical archives' official 1,000-query
metrics. V5 and v6 are intentionally absent because their official final
metrics use only the disjoint 750-query holdout.

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

For an unbiased comparison, remove the 250 development query IDs from the
older archives and average the remaining 750 queries. V5 and v6 report this
holdout directly:

| Run | NDCG@10 | Direct P@10 | Direct R@10 | Broad P@10 | Broad R@10 |
|---|---:|---:|---:|---:|---:|
| Base model | 0.825072 | 0.839619 | 0.612030 | 0.887115 | 0.604226 |
| v1 | 0.850373 | 0.857219 | 0.630857 | 0.895648 | 0.612677 |
| v3 | 0.851807 | 0.858552 | 0.632132 | 0.893381 | 0.611026 |
| v4 | 0.850545 | 0.854819 | 0.626608 | 0.893381 | 0.609372 |
| v5: 10k queries, p12, LR `1e-4` | 0.854425 | 0.856952 | 0.629724 | 0.895381 | 0.612037 |
| **v6: 20k queries, p6, LR `1e-4`** | **0.861641** | **0.862285** | **0.635127** | **0.897781** | **0.613674** |

V5 improves NDCG over v4 but is otherwise close to v1/v3. V6 is the first run
to lead this common holdout on every aggregate metric. Relative to v5 it gains
+0.007216 NDCG, +0.005333 direct precision, +0.005403 direct recall,
+0.002400 broad precision, and +0.001638 broad recall.

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

## Paired v6 versus v5 uncertainty

V6 and v5 use exactly the same 750 final query IDs. The following intervals
come from 20,000 paired query bootstrap resamples; positive values favor v6.

| Metric | Mean delta | Paired 95% interval | Queries better / worse / tied |
|---|---:|---:|---:|
| NDCG@10 | **+0.007216** | **[+0.001716, +0.012530]** | 336 / 293 / 121 |
| Direct P@10 | **+0.005333** | **[+0.001067, +0.009600]** | 91 / 63 / 596 |
| Direct R@10 | **+0.005403** | **[+0.000970, +0.009885]** | 91 / 63 / 596 |
| Broad P@10 | +0.002400 | [-0.001333, +0.006000] | 66 / 54 / 630 |
| Broad R@10 | +0.001638 | [-0.002083, +0.005316] | 66 / 54 / 630 |

The NDCG and direct-relevance gains are resolved at this query sample; the
broad gains are not. V6 also beats v1 by +0.011269 NDCG with a paired interval
of [+0.005265, +0.017148]. Direct Precision@10 is +0.005067 over v1 with a
barely positive interval of [+0.000133, +0.009867], while v6-versus-v1 direct
recall and broad metrics remain directionally positive but uncertain.

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

### V6 boundary behavior on the 750-query holdout

The same diagnostic restricted to the leakage-free holdout shows that v6
improves every ESCI boundary relative to v5:

| Boundary | Base | v1 | v3 | v4 | v5 | **v6** | v6 - v5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E>S` | 0.6640 | 0.6996 | 0.6994 | 0.7051 | 0.7108 | **0.7135** | +0.0027 |
| `E>C` | 0.6449 | 0.7656 | 0.7574 | 0.7485 | 0.7444 | **0.7825** | **+0.0381** |
| `E>I` | 0.7185 | 0.7546 | 0.7561 | 0.7675 | 0.7700 | **0.7750** | +0.0049 |
| `S>C` | 0.4815 | 0.6639 | 0.6949 | 0.6583 | 0.6476 | **0.6993** | **+0.0517** |
| `S>I` | 0.6229 | 0.6467 | 0.6431 | 0.6398 | 0.6460 | **0.6528** | +0.0068 |
| `C>I` | **0.6286** | 0.5420 | 0.5811 | 0.5826 | 0.5656 | 0.5728 | +0.0071 |

Query-clustered bootstrap intervals exclude zero for the two largest gains:
`E>C` is [+0.0121, +0.0683] and `S>C` is [+0.0227, +0.0834]. The other four
boundary changes are positive but uncertain. V6 still does not recover the
base model's unusual `C>I` ordering, although it improves that boundary over
v5.

V6's average top ten contains 5.684 `E`, 2.933 `S`, 0.355 `C`, and 1.019 `I`
products, versus v5's 5.627, 2.937, 0.384, and 1.043. It promotes more exact
products and removes both complement and irrelevant products; unlike v4/v5,
that shift accompanies improvements in direct precision/recall rather than an
NDCG-only tradeoff.

## Training diagnostics

| Run | Sampled pairs | Optimizer steps | Selected step | Mean RankNet loss | Training pair accuracy |
|---|---:|---:|---:|---:|---:|
| v1 | 59,982 | 1,875 | final | 0.423737 | 0.798740 |
| v2 | 117,691 | 3,678 | final | 0.345132 | 0.843191 |
| v3 | 59,982 | 1,875 | final | 0.448906 | 0.784669 |
| v4 | 119,959 | 3,749 | **1,875** | 0.428475 | 0.795905 |
| v5 | 119,959 | 3,749 | **1,250** | 0.443419 | 0.784793 |
| v6 | 120,000 | 3,750 | **2,500** | 0.494420 | 0.751958 |

V1 and v3 training loss and accuracy are not directly comparable because v3
contains many more scarce and difficult pair types. V2 is the useful warning:
better optimization metrics accompanied worse held-out ranking metrics. V4
reinforces that warning: its full-epoch training summary does not describe the
exported adapter, because development selection restored the step-1,875 state.
V5 and v6 strengthen the conclusion: v6 has a substantially higher full-epoch
loss and lower pair accuracy than v5, but clearly better holdout ranking. Its
broader query set contains harder or less repeatedly exposed comparisons, so
loss values across these data distributions should not be interpreted as a
quality leaderboard.

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

## V5 and V6 development trajectories

V5 and v6 use the same fixed 250 development query IDs. Bold values are the
checkpoint selected by NDCG@10.

| Optimizer step | v5 NDCG@10 | v6 NDCG@10 |
|---:|---:|---:|
| 0 | 0.837277 | 0.837277 |
| 625 | 0.855570 | 0.867532 |
| 1,250 | **0.866981** | 0.864418 |
| 1,875 | 0.862995 | 0.867245 |
| 2,500 | 0.858689 | **0.870519** |
| 3,125 | 0.855011 | 0.870191 |
| final (3,749 / 3,750) | 0.856386 | 0.866708 |

V5 peaked after one third of its updates and declined. V6 reached a higher
peak after two thirds of an effectively identical update budget, and step
3,125 was nearly tied. Broader query exposure therefore extended useful
learning; it did not merely create a lucky early checkpoint. Development
direct Recall@10 actually peaks at step 3,125 (0.654740), but NDCG selection
correctly retains step 2,500 for the declared primary objective.

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

V6 tests breadth while holding that budget fixed:

| Run | Queries | Pairs/query | Realized pairs | Optimizer steps | Candidate rows touched |
|---|---:|---:|---:|---:|---:|
| v5 | 10,000 | 12 | 119,959 | 3,749 | 127,379 |
| v6 | 20,000 | 6 | 120,000 | 3,750 | **166,729** |

V6 exposes 39,350 more query-product rows (+30.9%) at essentially identical
compute. Its pair-type counts also remain very close to v5: the largest change
is only 194 additional `E>S` pairs out of 120,000. The improvement is therefore
better explained by query/candidate diversity than by label-boundary
reweighting.

There is genuine unused product coverage at p12, but simply raising the cap
mostly adds the same dominant `E>S` and `E>I` distribution and greatly extends
training. V2 already showed that doubling random pairs can improve training
loss while harming held-out ranking. V6 confirms the better alternative:
spend the fixed budget across more queries. A coverage-first or hard-pair
sampler can now build on v6 without increasing total updates.

## Decision and next experiment

1. Promote v6 as the current incumbent. It leads the untouched holdout on all
   five aggregate metrics and has a resolved NDCG/direct-relevance advantage
   over v5 at this query sample.
2. Retain the v6 budget for the next optimization: 20,000 queries, p6,
   120,000 pairs, LR `1e-4`, and the same 250/750 split. This prevents a new
   sampler or loss experiment from being confounded with more compute.
3. The next model-improvement experiment should use **within-query hard-pair
   mining**: combine approximately 50% low-margin or misordered v6 training
   pairs with 50% round-robin coverage pairs. Never treat products from other
   queries as negatives because they are unjudged, not known irrelevant.
4. A coverage-first p6 sampler is a lower-cost alternative: maximize distinct
   products and available pair types within each query before reusing a
   product. Hold the realized pair count at 120,000.
5. Only after the sampling experiment, test a more metric-aligned loss. Start
   with grade-gap-weighted RankNet, then consider hardness-plus-grade-gap or
   LambdaRank/Delta-NDCG weighting. Avoid changing sampler and loss together.
6. If the goal is a scientific claim rather than the best single checkpoint,
   repeat v6 with another training seed. The paired intervals quantify query
   uncertainty but not initialization or training-sample variance.
7. Do not increase random pairs per query or run a broad LoRA/LR sweep yet.
   V6 demonstrates that redistributing the existing budget is more productive
   than adding arbitrary comparisons, and rank 16 shows no sign of a capacity
   ceiling.

The strongest causal conclusion is narrow but useful: at the same learning
rate, pair distribution, number of sampled pairs, optimizer updates, model,
and evaluation split, moving from 10k/p12 to 20k/p6 materially improves held-
out ranking. It does not by itself prove that 20k/p6 is globally optimal.

## Reproducibility notes

- Aggregate values come from each archive's `evaluation/metrics.json`.
- Paired intervals come from `evaluation/per_query_metrics.jsonl` with a fixed
  bootstrap seed of 42 and 20,000 resamples.
- Boundary accuracy and top-ten label counts come from
  `evaluation/predictions.parquet`.
- V5 and v6 `evaluation_split.json` files contain exactly equal development
  and final query-ID lists; both record zero overlap.
- V6 was interrupted four times. The retries resumed at optimizer steps 1,250,
  1,325, 2,225, and 2,450, preserved model-selection state, completed step
  3,750, restored the selected step-2,500 adapter, and exported the merged
  model.
- The archived raw scores are sufficient to recompute every table without
  rerunning model inference.
