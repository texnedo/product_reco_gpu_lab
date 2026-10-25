# Product reranker GPU lab

An end-to-end Azure ML lab for fine-tuning, evaluating, serving, and optimizing
Qwen3-Reranker-0.6B on the Amazon ESCI shopping-query dataset. The detailed
scope is in [docs/project_spec.md](docs/project_spec.md).

## Bootstrap the training inputs

The bootstrap registers immutable, versioned AML assets for the prepared ESCI
dataset and the Qwen checkpoint. The default local layout is:

```text
parent/
├── product_reco_gpu_lab/
└── ext_repos/
    ├── esci-data/
    └── Qwen3-Reranker-0.6B/
```

No machine-specific path is committed. For another layout, set
`ESCI_DATA_PATH` and `QWEN_MODEL_PATH` or pass the corresponding CLI options.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-bootstrap.txt

cp .aml.env.example local.aml.env
# Edit local.aml.env, then:
source local.aml.env

# Prepare US-English parquet files and inspect the exact upload set.
python infra/bootstrap_assets.py --dry-run

# Prepare (if needed), upload, and register both assets.
python infra/bootstrap_assets.py
```

The command is idempotent. If an asset version already exists, its recorded
content fingerprint is checked and the upload is skipped. If the same name and
version points to different local content, the command stops and requires a new
version in `config/assets.yaml`.

The resulting training input references are:

```text
azureml:esci-ranking:1
azureml:qwen3-reranker-0-6b:1
```

`infra/prepare_esci.py` can also be run independently. It filters both source
parquet files by `product_locale == "us"`, writes them to
`artifacts/esci/v1`, and records source hashes and row counts in
`dataset_info.json`. Generated artifacts and local AML environment files are
ignored by Git.

To create a new immutable dataset revision, change both `version` and
`prepared_path` in `config/assets.yaml`; never reuse a registered version for
different content.

## Run ESCI reranker experiments

The versioned instruction is stored in `config/prompts/esci_v1.txt`. The job
keeps Qwen3-Reranker's native `yes`/`no` output and defines the product ranking
score as `yes_logit - no_logit`. It applies LoRA to the transformer and trains
same-query product pairs with RankNet under the order `E > S > C > I`.
Same-label pairs are not constrained because ESCI provides no ordering within
a label.

Experiment and learning settings live in versioned YAML files under
`config/experiments/`. The first three configs reproduce completed runs; v4 is
the next broader-data experiment:

| Config | Sampler | Purpose |
|---|---|---|
| `esci_v1_12_pairs.yaml` | per-query round robin, 12 pairs | best NDCG and broad metrics |
| `esci_v2_24_pairs.yaml` | per-query round robin, 24 pairs | more updates; worse held-out ranking |
| `esci_v3_balanced.yaml` | 50% round robin + 50% globally type-balanced | tied overall; stronger rare boundaries |
| `esci_v4_q10k_p12.yaml` | per-query round robin, 12 pairs, 10,000 queries | broader run, basic text cleaning, development model selection |

See [`docs/esci_experiment_report.md`](docs/esci_experiment_report.md) for the
aggregate, paired-query, and label-boundary comparison of the base model and
all three LoRA runs.

Submit v4 after both input assets are registered. It is the default config, so
the two commands below are equivalent:

```bash
source local.aml.env
python jobs/submit_train_esci.py
python jobs/submit_train_esci.py \
    --config config/experiments/esci_v4_q10k_p12.yaml
```

The generated AML display name exposes the important resolved values:

```text
qwen-esci-v4-q10k-rr-p12-e1-b4-ga8-lr2e-4-r16-s42-clean1
```

Here `q10k` records query breadth, `rr` is round-robin sampling, `p12` is the
per-query pair budget, `b4/ga8` are batch and gradient accumulation, and the
remaining segments record epochs, learning rate, LoRA rank, and seed. Azure
ML's experiment name remains the stable `qwen-esci-reranker` group. `clean1`
means the versioned `basic_v1` product-text cleanup is enabled. It decodes HTML
entities, removes tags and script/style blocks, normalizes whitespace, and
preserves repeated sentences before token truncation. It does not remove
stopwords, cap or reorder fields, so the 512-token budget and existing
serializer priority remain unchanged. Override
only the run display with `--display-name` when necessary.

CLI learning options remain available as explicit overrides, but their
defaults are `None`, so choosing a YAML never silently inherits old CLI
defaults. Prefer copying a YAML for durable experiments:

```bash
python jobs/submit_train_esci.py \
    --config config/experiments/esci_v4_q10k_p12.yaml \
    --learning-rate 0.0001
```

Validate the resolved name and trainer command without contacting Azure:

```bash
python jobs/submit_train_esci.py \
    --config config/experiments/esci_v4_q10k_p12.yaml \
    --dry-run
```

Before comparing training variants, record the untouched Qwen baseline on the
same test-query sample and with the same prompt and scoring code:

```bash
python jobs/submit_train_esci.py \
    --config config/experiments/esci_v4_q10k_p12.yaml \
    --eval-only
```

This mode does not load the train split, create LoRA adapters or optimizer
state, write checkpoints, or export a model. Its only AML output is
`evaluation`, and its generated display name identifies the selected config as
a base-model evaluation.
`metrics.json` identifies it with `run_mode: base_model_evaluation` and
`model_variant: unmodified_base_model`, so it cannot be mistaken for a
zero-epoch training run.

The balanced config keeps the successful v1 budget: 59,982 unique pairs and
1,875 optimizer steps on the current 5,000-query selection. Half of the sample
uses the existing within-query round robin. The other half is divided globally
across all six label boundaries, with a per-query/type cap to prevent a few
queries from dominating. On the current data this changes the pair mix from
48.5% `E>S` and 2.9% `C>I` to approximately 32.5% and 9.8%, respectively,
without changing the total training or optimizer-step counts.

V4 selects 119,959 pairs from 10,000 queries and performs 3,749 optimizer
steps. A stable 250-query subset of the 1,000 evaluation queries is scored at
step zero, every 625 optimizer steps, and the final step. Development NDCG@10
is logged to MLflow. Whenever it improves, the job writes a complete best
checkpoint containing LoRA, optimizer, scheduler, GradScaler, RNG, and exact
progress state. At the end, the trainer reloads this best adapter before the
full evaluation and merged-model export.

The submission script creates the immutable `qwen-esci-training:1` environment
from `environments/train/` the first time it runs. The job consumes:

```text
azureml:esci-ranking:1
azureml:qwen3-reranker-0-6b:1
```

The training mode emits three AML outputs:

- `trained_model`: the LoRA adapter and a merged, self-contained model;
- `evaluation`: aggregate metrics, per-query metrics, and product scores;
- `checkpoints`: durable LoRA, optimizer, scheduler, GradScaler, RNG, epoch,
  and batch state used both for preemption recovery and best-model selection.

`checkpoints` is an AML `rw_mount`, so completed checkpoints reach datastore
storage while the process is still running. The trainer saves every 25
optimizer updates by default and at every epoch boundary, retaining the newest
three complete checkpoints. An automatic retry
of the same AML job scans this output, ignores any checkpoint interrupted
before its `COMPLETE` marker, and resumes from the most advanced valid one.
Change the recovery window with `--checkpoint-every-steps`; smaller values
redo less work after eviction but write the adapter and optimizer state more
often. A separately submitted new AML job receives a new output location and
does not automatically inherit checkpoints from an older job.

Recovery directories and the best-model directory are independent. The newest
three `checkpoint-*` directories rotate for resume, while one
`best-checkpoint-*` directory is retained regardless of its age. Resume loads
the newest recovery state and also remembers the best development score found
before preemption. `evaluation/metrics.json` records the complete development
curve and selected step; the model artifact also contains
`model_selection.json`.

NDCG uses graded relevance `E=3, S=2, C=1, I=0`. Direct Precision@10 and
Recall@10 treat `E` and `S` as relevant; broad Precision@10 and Recall@10 also
count `C`. Metrics are first calculated per query and then averaged. Queries
with no relevant candidates remain in precision, but are excluded from recall
and NDCG averages where the denominator is zero. The prediction artifact logs
both logits, the ranking score, and `sigmoid(score)` as an explicitly
uncalibrated diagnostic.

Each result also records the YAML filename and SHA-256, generated display name,
resolved sampler and learning parameters, and realized pair-type counts. The
model output contains the original YAML as `experiment_config_source.yaml`;
`training_metadata.json` records the effective values after any CLI overrides.

## Review a completed run

[`notebooks/qwen_esci_v1_run_review.ipynb`](notebooks/qwen_esci_v1_run_review.ipynb)
contains the executed analysis of the first LoRA run. It reports aggregate and
query-level ordering metrics, joins predictions to product text, shows strong
`E > S > C > I` examples, surfaces the largest score/label inversions, and
provides a manual annotation-review sheet. By default it reads the archived
first run from `run_results/`; set `ESCI_RUN_ARCHIVE` or `ESCI_DATA_PATH` when
using another artifact or dataset location.
