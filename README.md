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
