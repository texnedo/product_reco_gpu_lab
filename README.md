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

## Train the v1 ESCI reranker

The versioned instruction is stored in `config/prompts/esci_v1.txt`. The job
keeps Qwen3-Reranker's native `yes`/`no` output and defines the product ranking
score as `yes_logit - no_logit`. It applies LoRA to the transformer and trains
same-query product pairs with RankNet under the order `E > S > C > I`.
Same-label pairs are not constrained because ESCI provides no ordering within
a label.

Submit the initial T4-sized experiment after both input assets are registered:

```bash
source local.aml.env
python jobs/submit_train_esci.py
```

The default run selects complete candidate groups for 5,000 training queries
and 1,000 test queries from ESCI's `small_version`. It samples up to 12
different-label pairs per training query on each epoch. It uses one epoch,
sequence length 512, pair batch size 4 (eight independently scored products),
and eight-step gradient accumulation. Scale these limits only after the first
run is healthy:

```bash
python jobs/submit_train_esci.py \
    --train-queries 20000 \
    --eval-queries 5000 \
    --epochs 2
```

The submission script creates the immutable `qwen-esci-training:1` environment
from `environments/train/` the first time it runs. The job consumes:

```text
azureml:esci-ranking:1
azureml:qwen3-reranker-0-6b:1
```

It emits two AML outputs:

- `trained_model`: the LoRA adapter and a merged, self-contained model;
- `evaluation`: aggregate metrics, per-query metrics, and product scores.

NDCG uses graded relevance `E=3, S=2, C=1, I=0`. Direct Precision@10 and
Recall@10 treat `E` and `S` as relevant; broad Precision@10 and Recall@10 also
count `C`. Metrics are first calculated per query and then averaged. Queries
with no relevant candidates remain in precision, but are excluded from recall
and NDCG averages where the denominator is zero. The prediction artifact logs
both logits, the ranking score, and `sigmoid(score)` as an explicitly
uncalibrated diagnostic.
