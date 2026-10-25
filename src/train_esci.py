#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-Reranker on ESCI with a pairwise RankNet loss.

The central idea is:

    one (query, product) -> Qwen "yes" and "no" logits
    ranking score         -> yes_logit - no_logit
    two products          -> RankNet asks the better ESCI item to score higher

ESCI labels are training supervision, not model output classes. At serving
time each product is still scored independently with the model's native
yes/no head, then candidates for a query are sorted by that scalar score.
This file is intentionally well commented because the project is a learning
lab as well as an Azure ML training entry point.

LoRA basics:
    Attention:
        Q = (Wq + ΔWq)x
        K = (Wk + ΔWk)x
        V = (Wv + ΔWv)x
        output = (Wo + ΔWo) attention(Q, K, V)

    MLP:
        gate = (Wgate + ΔWgate)x
        up   = (Wup + ΔWup)x
        output = (Wdown + ΔWdown) activation(gate, up)

So the final model design is:
Original Qwen architecture
    + LoRA adapters inside selected linear layers
    + custom scalar interpretation of existing yes/no logits
    + external RankNet pairwise loss
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import duckdb
import pandas as pd
import torch
import torch.nn.functional as functional
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from checkpoint_core import (
    CheckpointProgress,
    checkpoint_signature,
    remaining_epoch_items,
    shuffled_epoch_items,
)
from esci_core import (
    BROAD_RELEVANT,
    DIRECT_RELEVANT,
    GRADE,
    LABEL_ORDER,
    TEXT_CLEANING_MODES,
    TrainingPair,
    flatten_metrics,
    product_document,
    ranking_metrics,
    safe_text,
    sample_training_pairs,
)
from training_checkpoint import (
    best_checkpoint_metadata,
    load_best_model_checkpoint,
    load_training_checkpoint,
    save_best_training_checkpoint,
    save_training_checkpoint,
)

try:
    # MLflow is supplied by Azure ML at runtime but is intentionally optional
    # for local unit tests and code-reading environments.
    import mlflow
except ImportError:
    mlflow = None


EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"


def selection_history_sort_key(row: dict[str, Any]) -> int:
    return int(row["optimizer_step"])


# Native Qwen3-Reranker prompt contract. The contents of esci_v1.txt become
# <Instruct>; they explain what "meets the requirements" means for products.
RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANKER_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def training_run_configuration(
    args: argparse.Namespace,
    instruction: str,
    train_frame: Any,
    initial_pair_count: int,
    development_frame: Any = None,
) -> dict[str, Any]:
    """Describe every setting that must match when state is restored."""
    training_rows = hashlib.sha256()
    for query_id, product_id, label in train_frame[
        ["query_id", "product_id", "esci_label"]
    ].itertuples(index=False, name=None):
        training_rows.update(
            f"{query_id!r}\0{product_id!r}\0{label}\n".encode("utf-8")
        )
    model_config = args.model_dir / "config.json"
    model_config_hash = (
        hashlib.sha256(model_config.read_bytes()).hexdigest()
        if model_config.is_file()
        else None
    )
    development_rows_hash = None
    if development_frame is not None:
        development_rows = hashlib.sha256()
        for query_id, product_id, label in development_frame[
            ["query_id", "product_id", "esci_label"]
        ].itertuples(index=False, name=None):
            development_rows.update(
                f"{query_id!r}\0{product_id!r}\0{label}\n".encode("utf-8")
            )
        development_rows_hash = development_rows.hexdigest()
    return {
        "prompt_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "model_config_sha256": model_config_hash,
        "training_rows_sha256": training_rows.hexdigest(),
        "initial_pair_count": initial_pair_count,
        "development_rows_sha256": development_rows_hash,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "text_cleaning": args.text_cleaning,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "pairs_per_query": args.pairs_per_query,
        "pair_weighting": args.pair_weighting,
        "sampling_strategy": args.sampling_strategy,
        "balance_fraction": args.balance_fraction,
        "balanced_pairs_per_type_per_query": (
            args.balanced_pairs_per_type_per_query
        ),
        "selection_queries": args.selection_queries,
        "selection_every_steps": args.selection_every_steps,
        "selection_min_delta": args.selection_min_delta,
        "eval_queries": args.eval_queries,
        "ndcg_k": args.ndcg_k,
        "seed": args.seed,
    }


def experiment_config_sha256(args: argparse.Namespace) -> Optional[str]:
    if args.experiment_config_file is None:
        return None
    return hashlib.sha256(args.experiment_config_file.read_bytes()).hexdigest()


def sample_pairs(
    args: argparse.Namespace,
    query_ids: Sequence[Any],
    labels: Sequence[str],
    seed: int,
) -> list[TrainingPair]:
    """Apply the configured sampler consistently to setup and every epoch."""
    return sample_training_pairs(
        query_ids,
        labels,
        args.pairs_per_query,
        seed,
        sampling_strategy=args.sampling_strategy,
        balance_fraction=args.balance_fraction,
        balanced_pairs_per_type_per_query=(
            args.balanced_pairs_per_type_per_query
        ),
    )


@dataclass(frozen=True)
class PromptEncoding:
    """Tokenized constant prompt pieces and the remaining variable budget."""

    prefix_ids: tuple[int, ...]
    instruction_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    document_separator: str
    max_variable_tokens: int

    @classmethod
    def build(
        cls, tokenizer: Any, instruction: str, max_length: int
    ) -> "PromptEncoding":
        prefix_ids = tuple(
            tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
        )
        suffix_ids = tuple(
            tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
        )
        instruction_ids = tuple(
            tokenizer.encode(
                f"<Instruct>: {instruction}\n<Query>: ",
                add_special_tokens=False,
            )
        )
        document_separator = "\n<Document>: "
        separator_length = len(
            tokenizer.encode(document_separator, add_special_tokens=False)
        )
        max_variable_tokens = (
            max_length
            - len(prefix_ids)
            - len(instruction_ids)
            - len(suffix_ids)
        )
        content_budget = max_variable_tokens - separator_length
        if content_budget < 64:
            raise ValueError(
                f"max-length={max_length} leaves only {content_budget} "
                "query/product tokens"
            )
        return cls(
            prefix_ids=prefix_ids,
            instruction_ids=instruction_ids,
            suffix_ids=suffix_ids,
            document_separator=document_separator,
            max_variable_tokens=max_variable_tokens,
        )


class ProductDataset(Dataset):
    """Pointwise encoded products, reused by pair training and evaluation."""

    def __init__(
        self,
        frame: Any,
        tokenizer: Any,
        prompt_encoding: PromptEncoding,
        text_cleaning: str,
    ) -> None:
        self.columns = {name: frame[name].tolist() for name in frame.columns}
        self.tokenizer = tokenizer
        self.prompt_encoding = prompt_encoding
        self.text_cleaning = text_cleaning

    def __len__(self) -> int:
        return len(self.columns["esci_label"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = {name: values[index] for name, values in self.columns.items()}
        variable_text = (
            f"{safe_text(row['query'])}"
            f"{self.prompt_encoding.document_separator}"
            f"{product_document(row, self.text_cleaning)}"
        )
        variable_ids = self.tokenizer.encode(
            variable_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.prompt_encoding.max_variable_tokens,
        )
        return {
            "input_ids": [
                *self.prompt_encoding.prefix_ids,
                *self.prompt_encoding.instruction_ids,
                *variable_ids,
                *self.prompt_encoding.suffix_ids,
            ],
            "gold_label": row["esci_label"],
            "query_id": row["query_id"],
            "product_id": row["product_id"],
        }


class ProductCollator:
    """Right-pad prompts; the last non-padding position produces the score."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        max_tokens = max(len(item["input_ids"]) for item in items)
        input_ids, attention_masks = [], []
        for item in items:
            tokens = item["input_ids"]
            padding = max_tokens - len(tokens)
            input_ids.append(tokens + [self.pad_token_id] * padding)
            attention_masks.append([1] * len(tokens) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "gold_labels": [item["gold_label"] for item in items],
            "query_ids": [item["query_id"] for item in items],
            "product_ids": [item["product_id"] for item in items],
        }


class PairDataset(Dataset):
    """Resolve sampled row-index pairs into two independently scored products."""

    def __init__(
        self, products: ProductDataset, pairs: Sequence[TrainingPair]
    ) -> None:
        self.products = products
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        return {
            "higher": self.products[pair.higher_index],
            "lower": self.products[pair.lower_index],
            "grade_gap": pair.grade_gap,
            "pair_type": pair.pair_type,
        }


class PairCollator:
    """Flatten pairs into pointwise inputs while retaining pair metadata."""

    def __init__(self, product_collator: ProductCollator) -> None:
        self.product_collator = product_collator

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        # [higher_0, lower_0, ...] can be recovered by scores.reshape(-1, 2).
        flattened = [
            product
            for item in items
            for product in (item["higher"], item["lower"])
        ]
        batch = self.product_collator(flattened)
        batch["grade_gaps"] = torch.tensor(
            [item["grade_gap"] for item in items], dtype=torch.float32
        )
        batch["pair_types"] = [item["pair_type"] for item in items]
        return batch


def native_answer_token_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve and validate Qwen's existing single-token yes/no answers."""
    answer_token_ids = {}
    for answer in ("yes", "no"):
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                f"Reranker answer {answer!r} is not one token: {token_ids}"
            )
        answer_token_ids[answer] = token_ids[0]
    return answer_token_ids


def forward_yes_no(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    yes_no_token_tensor: Any,
) -> tuple[Any, Any]:
    """Return [batch, 2] yes/no logits and [batch] ranking scores.

    Let H have shape [B, T, D], where B is batch size, T is sequence
    length, and D is Qwen's hidden size. For item i we select h_i, the
    hidden vector at its last non-padding token. Qwen's existing language
    model head contains W_vocab with shape [V, D] and normally computes:

        z_i = W_vocab @ h_i + b                 # V vocabulary logits

    We only need the rows for the existing "yes" and "no" tokens:

        z_yes = w_yes @ h_i + b_yes
        z_no  = w_no  @ h_i + b_no
        score = z_yes - z_no
              = (w_yes - w_no) @ h_i + (b_yes - b_no)

    Qwen's reference reranker applies a two-class softmax and returns:

        p_yes = exp(z_yes) / (exp(z_yes) + exp(z_no))
              = sigmoid(z_yes - z_no)
              = sigmoid(score)

    Sigmoid is strictly increasing, so sorting by p_yes and sorting by this
    raw score produce exactly the same ranking. The raw score is the log-odds
    log(p_yes / p_no); unlike a probability, it does not saturate to [0, 1]
    before RankNet computes differences between two products.

    With the default logits_to_keep=0, a normal causal-LM forward allocates
    logits shaped [B, T, V]. This custom Qwen implementation also supports
    logits_to_keep=1, which can reduce this to [B, 1, V] when the score
    position is the final padded position, but it still projects to all V
    vocabulary entries. This implementation ensures that right-padded rows have different final real-token
    positions, and we only require two vocabulary entries.

    Running the decoder directly and projecting only two rows therefore gives
    the same z_yes/z_no values while avoiding the other positions and V-2
    output logits. No values in W_vocab or b are zeroed or modified:
    index_select constructs the smaller matrix [w_yes; w_no]. Algebraically,

        (W_vocab @ h + b)[yes/no]
        == W_vocab[yes/no] @ h + b[yes/no]

    This does not bypass LoRA: get_base_model().model still contains PEFT's
    adapted transformer layers. It only bypasses the unnecessary parts of the
    existing vocabulary projection.
    """
    # PEFT exposes get_base_model(); an unwrapped Hugging Face model does not.
    # Supporting both is what lets --eval-only score the untouched base model
    # through exactly the same math used after LoRA training.
    causal_model = (
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )
    # hidden_states: [B, T, D]. LoRA participates in this decoder computation.
    hidden_states = causal_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    # Prompts are right-padded, so sum(mask)-1 is each row's last real token.
    # This is the position where the assistant is about to emit yes or no.
    last_positions = attention_mask.sum(dim=1) - 1
    final_hidden = hidden_states[
        torch.arange(input_ids.shape[0], device=input_ids.device),
        last_positions,
    ]

    # The LM head is Qwen's original vocabulary projection, not a new
    # classifier. index_select copies/views the chosen rows into
    # [w_yes; w_no], shape [2, D]; it does not mask or mutate the other rows.
    output_layer = causal_model.get_output_embeddings()
    weights = output_layer.weight.index_select(0, yes_no_token_tensor)
    bias = (
        output_layer.bias.index_select(0, yes_no_token_tensor)
        if getattr(output_layer, "bias", None) is not None
        else None
    )
    # functional.linear computes final_hidden @ weights.T + bias, producing
    # [B, 2] in the token order [yes, no].
    logits = functional.linear(final_hidden, weights, bias)

    # This is both the default reranker's log-odds and a shift-invariant score:
    # adding the same constant to yes/no logits cancels in their difference.
    score = logits[:, 0] - logits[:, 1]
    return logits, score


def load_split(
    data_dir: Path,
    split: str,
    max_queries: int,
    small_version: bool,
    seed: int,
) -> Any:
    """Join examples to products while retaining complete query groups.

    Sampling individual rows would damage ranking evaluation and could leave
    no valid training comparison for a query. We choose query IDs first and
    then retain every candidate belonging to each selected query.
    """
    examples = data_dir / EXAMPLES_FILE
    products = data_dir / PRODUCTS_FILE
    missing = [str(path) for path in (examples, products) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AML data input file(s): " + ", ".join(missing))

    small_filter = "AND e.small_version = 1" if small_version else ""
    query_limit = f"LIMIT {int(max_queries)}" if max_queries > 0 else ""
    query = f"""
        WITH eligible AS (
            SELECT *
            FROM read_parquet(?) e
            WHERE e.split = ? {small_filter}
        ),
        selected_queries AS (
            SELECT query_id
            FROM eligible
            GROUP BY query_id
            ORDER BY hash(query_id + {int(seed)})
            {query_limit}
        )
        SELECT
            e.example_id,
            e.query_id,
            e.query,
            e.product_id,
            e.product_locale,
            e.esci_label,
            p.product_title,
            p.product_description,
            p.product_bullet_point,
            p.product_brand,
            p.product_color
        FROM eligible e
        JOIN selected_queries q USING (query_id)
        LEFT JOIN read_parquet(?) p
          ON e.product_id = p.product_id
         AND e.product_locale = p.product_locale
        ORDER BY e.query_id, e.example_id
    """
    frame = duckdb.connect().execute(
        query, [str(examples), split, str(products)]
    ).fetch_df()
    if frame.empty:
        raise RuntimeError(f"No rows selected for split={split!r}")
    invalid = sorted(set(frame["esci_label"]) - set(LABEL_ORDER))
    if invalid:
        raise ValueError(f"Unexpected ESCI labels in {split}: {invalid}")
    missing_products = int(frame["product_title"].isna().sum())
    if missing_products:
        print(f"warning: {missing_products:,} rows have no matching product metadata")
    print(
        f"Loaded {split}: {len(frame):,} products across "
        f"{frame['query_id'].nunique():,} complete queries"
    )
    return frame


def select_development_frame(
    eval_frame: Any, query_count: int, seed: int
) -> Any:
    """Choose a stable complete-query subset from the larger evaluation frame."""
    query_ids = list(eval_frame["query_id"].unique())
    hashed_query_ids = sorted(
        (
            hashlib.sha256(
                f"{seed}:development:{query_id}".encode("utf-8")
            ).digest(),
            query_id,
        )
        for query_id in query_ids
    )
    ordered = [query_id for _, query_id in hashed_query_ids]
    selected = set(ordered[:query_count])
    frame = eval_frame[eval_frame["query_id"].isin(selected)].copy()
    print(
        f"Selected development subset: {frame['query_id'].nunique():,} queries, "
        f"{len(frame):,} products",
        flush=True,
    )
    return frame


def score_product_loader(
    model: Any,
    loader: Any,
    device: Any,
    yes_no_token_tensor: Any,
    description: str,
) -> dict[str, list[Any]]:
    """Score a pointwise loader and restore the caller's train/eval mode."""
    was_training = model.training
    python_rng_state = random.getstate()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state_all = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    model.eval()
    output: dict[str, list[Any]] = {
        "gold_labels": [],
        "query_ids": [],
        "product_ids": [],
        "yes_logits": [],
        "no_logits": [],
        "ranking_scores": [],
    }
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc=description, leave=False):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    device, non_blocking=True
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, scores = forward_yes_no(
                        model,
                        input_ids,
                        attention_mask,
                        yes_no_token_tensor,
                    )
                cpu_logits = logits.float().cpu()
                output["yes_logits"].extend(cpu_logits[:, 0].tolist())
                output["no_logits"].extend(cpu_logits[:, 1].tolist())
                output["ranking_scores"].extend(scores.float().cpu().tolist())
                output["gold_labels"].extend(batch["gold_labels"])
                output["query_ids"].extend(batch["query_ids"])
                output["product_ids"].extend(batch["product_ids"])
    finally:
        # Creating worker iterators and some CUDA kernels can advance RNG state.
        # Evaluation must observe training without altering its subsequent
        # dropout masks or retry determinism.
        random.setstate(python_rng_state)
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available() and cuda_rng_state_all:
            torch.cuda.set_rng_state_all(cuda_rng_state_all)
        if was_training:
            model.train()
    return output


def evaluate_development(
    args: argparse.Namespace,
    model: Any,
    development_loader: Any,
    device: Any,
    yes_no_token_tensor: Any,
    global_step: int,
) -> dict[str, Any]:
    """Evaluate the fixed selection subset without writing prediction files."""
    scored = score_product_loader(
        model,
        development_loader,
        device,
        yes_no_token_tensor,
        description=f"development step {global_step}",
    )
    ranking, _ = ranking_metrics(
        scored["query_ids"],
        scored["gold_labels"],
        scored["ranking_scores"],
        args.ndcg_k,
    )
    print(
        f"Development metrics at optimizer step {global_step}: "
        + json.dumps(ranking, sort_keys=True),
        flush=True,
    )
    if mlflow is not None:
        try:
            mlflow.log_metrics(
                flatten_metrics({"development": ranking}), step=global_step
            )
        except Exception as exc:
            print(f"warning: MLflow development logging was unavailable: {exc}")
    return ranking


def evaluate_and_write_outputs(
    args: argparse.Namespace,
    model: Any,
    eval_loader: Any,
    device: Any,
    yes_no_token_tensor: Any,
    eval_frame: Any,
    instruction: str,
    training_history: list[dict[str, Any]],
    train_frame: Any = None,
    selection_history: Optional[list[dict[str, Any]]] = None,
    selected_checkpoint: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the serving-time scorer and persist one comparable evaluation."""
    scored = score_product_loader(
        model,
        eval_loader,
        device,
        yes_no_token_tensor,
        description="final evaluation",
    )
    gold_labels = scored["gold_labels"]
    query_ids = scored["query_ids"]
    product_ids = scored["product_ids"]
    yes_logits = scored["yes_logits"]
    no_logits = scored["no_logits"]
    ranking_scores = scored["ranking_scores"]

    ranking, per_query = ranking_metrics(
        query_ids, gold_labels, ranking_scores, args.ndcg_k
    )
    configuration = {
        "run_mode": "base_model_evaluation" if args.eval_only else "lora_training",
        "model_variant": (
            "unmodified_base_model" if args.eval_only else "lora_fine_tuned"
        ),
        "objective": "none" if args.eval_only else "ranknet_pairwise",
        "score": "yes_logit - no_logit",
        "prompt_version": "esci_v1",
        "prompt_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "label_order": list(LABEL_ORDER),
        "graded_relevance": GRADE,
        "direct_relevant_labels": sorted(DIRECT_RELEVANT),
        "broad_relevant_labels": sorted(BROAD_RELEVANT),
        "train_products": 0 if train_frame is None else len(train_frame),
        "train_queries": (
            0 if train_frame is None else int(train_frame["query_id"].nunique())
        ),
        "eval_products": len(eval_frame),
        "eval_queries": int(eval_frame["query_id"].nunique()),
        "max_length": args.max_length,
        "text_cleaning": args.text_cleaning,
        "small_version": args.small_version,
        "epochs": 0 if args.eval_only else args.epochs,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "experiment_config": (
            args.experiment_config_file.name
            if args.experiment_config_file is not None
            else None
        ),
        "experiment_config_sha256": experiment_config_sha256(args),
        "run_display_name": args.run_display_name,
    }
    if not args.eval_only:
        configuration.update(
            {
                "pairs_per_query": args.pairs_per_query,
                "pair_weighting": args.pair_weighting,
                "sampling_strategy": args.sampling_strategy,
                "balance_fraction": args.balance_fraction,
                "balanced_pairs_per_type_per_query": (
                    args.balanced_pairs_per_type_per_query
                ),
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "selection_queries": args.selection_queries,
                "selection_every_steps": args.selection_every_steps,
                "selection_min_delta": args.selection_min_delta,
            }
        )
    metrics = {
        "training": {"epochs": training_history},
        "ranking": ranking,
        "configuration": configuration,
    }
    if not args.eval_only:
        metrics["model_selection"] = {
            "metric": f"ndcg@{args.ndcg_k}",
            "history": selection_history or [],
            "selected_checkpoint": selected_checkpoint,
        }
    print(json.dumps(metrics, indent=2))

    args.metrics_output.mkdir(parents=True, exist_ok=True)
    (args.metrics_output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.metrics_output / "per_query_metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in per_query:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # sigmoid(score) is a useful diagnostic, but neither the base reranker nor
    # RankNet guarantees that it is calibrated as an ESCI relevance probability.
    yes_probability = [
        1.0 / (1.0 + math.exp(-max(min(score, 60.0), -60.0)))
        for score in ranking_scores
    ]
    prediction_frame = pd.DataFrame(
        {
            "query_id": query_ids,
            "product_id": product_ids,
            "gold_label": gold_labels,
            "yes_logit": yes_logits,
            "no_logit": no_logits,
            "ranking_score": ranking_scores,
            "yes_probability_uncalibrated": yes_probability,
        }
    )
    prediction_frame["predicted_rank"] = (
        prediction_frame.groupby("query_id")["ranking_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    prediction_frame.to_parquet(
        args.metrics_output / "predictions.parquet", index=False
    )

    mlflow_metrics: dict[str, Any] = {"ranking": ranking}
    if training_history:
        mlflow_metrics["final_training"] = {
            "mean_ranknet_loss": training_history[-1]["mean_ranknet_loss"],
            "pair_accuracy": training_history[-1]["pair_accuracy"],
        }
    if selected_checkpoint is not None:
        mlflow_metrics["model_selection"] = {
            "selected_step": int(selected_checkpoint["global_step"]),
            "selected_development_metric": float(
                selected_checkpoint["selection_metric_value"]
            ),
        }
    if mlflow is None:
        print("warning: MLflow is not installed; metrics remain in the AML output")
    else:
        try:
            mlflow.log_metrics(flatten_metrics(mlflow_metrics))
            mlflow.log_artifacts(
                str(args.metrics_output), artifact_path="evaluation"
            )
        except Exception as exc:
            print(f"warning: MLflow logging was unavailable: {exc}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--experiment-config-file",
        type=Path,
        help="Source YAML used by the submitter; hashed into run metadata.",
    )
    parser.add_argument(
        "--run-display-name",
        help="Self-describing AML display name recorded with evaluation metadata.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Required for training; omitted for a base-model evaluation.",
    )
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the unmodified base model without LoRA or optimization.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Durable directory for periodic state; use an AML rw_mount output.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=25,
        help="Save after this many optimizer updates, plus every epoch boundary.",
    )
    parser.add_argument(
        "--checkpoints-to-keep",
        type=int,
        default=3,
        help="Retain this many newest complete checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume the latest complete checkpoint when checkpoint-dir is set.",
    )
    parser.add_argument("--train-queries", type=int, default=5_000)
    parser.add_argument("--eval-queries", type=int, default=1_000)
    parser.add_argument(
        "--small-version", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--text-cleaning",
        choices=sorted(TEXT_CLEANING_MODES),
        default="none",
        help="Versioned product-field cleanup applied before token truncation.",
    )
    # This batch size counts pairs, so four means eight model inputs.
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--pairs-per-query", type=int, default=12)
    parser.add_argument(
        "--sampling-strategy",
        choices=("query_round_robin", "mixed_type_balanced"),
        default="query_round_robin",
    )
    parser.add_argument("--balance-fraction", type=float, default=0.0)
    parser.add_argument(
        "--balanced-pairs-per-type-per-query",
        type=int,
        default=12,
        help="Global balanced-pool cap for one pair type from one query.",
    )
    parser.add_argument(
        "--pair-weighting",
        choices=("none", "grade_gap"),
        default="none",
        help="Optionally weight E>I more than adjacent-label comparisons.",
    )
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument(
        "--selection-queries",
        type=int,
        default=0,
        help="Fixed evaluation-query subset used for in-training model selection.",
    )
    parser.add_argument(
        "--selection-every-steps",
        type=int,
        default=0,
        help="Evaluate development NDCG after this many optimizer updates.",
    )
    parser.add_argument(
        "--selection-min-delta",
        type=float,
        default=0.0,
        help="Required strict NDCG improvement before replacing the best model.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    positive_values = {
        "max_length": args.max_length,
        "eval_batch_size": args.eval_batch_size,
        "ndcg_k": args.ndcg_k,
    }
    if not args.eval_only:
        positive_values.update(
            {
                "epochs": args.epochs,
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "pairs_per_query": args.pairs_per_query,
                "balanced_pairs_per_type_per_query": (
                    args.balanced_pairs_per_type_per_query
                ),
                "checkpoint_every_steps": args.checkpoint_every_steps,
                "checkpoints_to_keep": args.checkpoints_to_keep,
            }
        )
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if not args.eval_only and args.model_output is None:
        raise ValueError("--model-output is required unless --eval-only is set")
    selection_enabled = args.selection_queries > 0 or args.selection_every_steps > 0
    if not args.eval_only and selection_enabled:
        if args.selection_queries <= 0 or args.selection_every_steps <= 0:
            raise ValueError(
                "--selection-queries and --selection-every-steps must both be "
                "positive when model selection is enabled"
            )
        if args.selection_queries > args.eval_queries:
            raise ValueError("--selection-queries cannot exceed --eval-queries")
        if args.checkpoint_dir is None:
            raise ValueError(
                "--checkpoint-dir is required for durable best-model selection"
            )
    if args.selection_min_delta < 0.0:
        raise ValueError("--selection-min-delta cannot be negative")
    if not selection_enabled and args.selection_min_delta != 0.0:
        raise ValueError(
            "--selection-min-delta requires development model selection"
        )
    if not 0.0 <= args.balance_fraction <= 1.0:
        raise ValueError("--balance-fraction must be in [0, 1]")
    if (
        args.sampling_strategy == "query_round_robin"
        and args.balance_fraction != 0.0
    ):
        raise ValueError("query_round_robin requires --balance-fraction 0")
    if (
        args.sampling_strategy == "mixed_type_balanced"
        and args.balance_fraction <= 0.0
    ):
        raise ValueError("mixed_type_balanced requires --balance-fraction > 0")
    if (
        args.experiment_config_file is not None
        and not args.experiment_config_file.is_file()
    ):
        raise FileNotFoundError(args.experiment_config_file)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This Qwen scoring job requires a CUDA GPU")

    instruction = args.prompt_file.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError(f"Prompt file is empty: {args.prompt_file}")
    train_frame = None
    if not args.eval_only:
        train_frame = load_split(
            args.data_dir, "train", args.train_queries, args.small_version, args.seed
        )
    eval_frame = load_split(
        args.data_dir, "test", args.eval_queries, args.small_version, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Read only these two rows from Qwen's existing language-model head. No
    # E/C/S/I classification layer is added. Although the head stays frozen,
    # gradients flow through it into the hidden state and LoRA matrices.
    answer_token_ids = native_answer_token_ids(tokenizer)
    print("Native answer token IDs:", answer_token_ids)

    prompt_encoding = PromptEncoding.build(tokenizer, instruction, args.max_length)
    eval_products = ProductDataset(
        eval_frame, tokenizer, prompt_encoding, args.text_cleaning
    )
    product_collator = ProductCollator(tokenizer.pad_token_id)

    eval_loader = DataLoader(
        eval_products,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=product_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    development_frame = None
    development_loader = None
    if not args.eval_only and selection_enabled:
        development_frame = select_development_frame(
            eval_frame, args.selection_queries, args.seed
        )
        development_loader = DataLoader(
            ProductDataset(
                development_frame,
                tokenizer,
                prompt_encoding,
                args.text_cleaning,
            ),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=product_collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float16
    )
    model.config.use_cache = False

    if args.eval_only:
        model.to(device)
        yes_no_token_tensor = torch.tensor(
            [answer_token_ids["yes"], answer_token_ids["no"]],
            dtype=torch.long,
            device=device,
        )
        print("Evaluation-only mode: using the unmodified base Qwen model")
        evaluate_and_write_outputs(
            args,
            model,
            eval_loader,
            device,
            yes_no_token_tensor,
            eval_frame,
            instruction,
            training_history=[],
        )
        return 0

    # Everything below this branch is training-only. The baseline path never
    # creates PEFT adapters, an optimizer, a scheduler, or checkpoints.
    train_products = ProductDataset(
        train_frame, tokenizer, prompt_encoding, args.text_cleaning
    )
    pair_collator = PairCollator(product_collator)
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        ),
    )
    model.enable_input_require_grads()
    model.to(device)
    model.print_trainable_parameters()

    yes_no_token_tensor = torch.tensor(
        [answer_token_ids["yes"], answer_token_ids["no"]],
        dtype=torch.long,
        device=device,
    )

    train_query_ids = train_frame["query_id"].tolist()
    train_labels = train_frame["esci_label"].tolist()
    initial_pairs = sample_pairs(
        args, train_query_ids, train_labels, args.seed
    )
    if not initial_pairs:
        raise RuntimeError("Training selection produced no cross-label query pairs")
    print(
        f"Sampled {len(initial_pairs):,} training pairs "
        f"with strategy={args.sampling_strategy!r} "
        f"(reference budget {args.pairs_per_query} per query)"
    )
    initial_pair_type_counts = Counter(
        pair.pair_type for pair in initial_pairs
    )
    print(
        "Initial pair-type distribution:",
        json.dumps(dict(sorted(initial_pair_type_counts.items()))),
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(len(initial_pairs) / args.train_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    run_configuration = training_run_configuration(
        args,
        instruction,
        train_frame,
        len(initial_pairs),
        development_frame,
    )
    run_signature = checkpoint_signature(run_configuration)
    checkpoint_progress = None
    if args.checkpoint_dir is not None and args.resume:
        checkpoint_progress = load_training_checkpoint(
            args.checkpoint_dir,
            run_signature,
            model,
            optimizer,
            scheduler,
            scaler,
        )
    if checkpoint_progress is None:
        checkpoint_progress = CheckpointProgress()
    if checkpoint_progress.epoch_index > args.epochs:
        raise RuntimeError(
            "Checkpoint epoch is beyond configured training epochs: "
            f"{checkpoint_progress.epoch_index} > {args.epochs}"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = checkpoint_progress.global_step
    training_history = list(checkpoint_progress.training_history)
    selection_history = list(checkpoint_progress.selection_history)
    selection_metric = f"ndcg@{args.ndcg_k}"
    selected_checkpoint_metadata = None
    best_selection_value = -math.inf
    if development_loader is not None and args.resume:
        resolved_best = best_checkpoint_metadata(
            args.checkpoint_dir, run_signature
        )
        if resolved_best is not None:
            _, selected_checkpoint_metadata = resolved_best
            best_selection_value = float(
                selected_checkpoint_metadata["selection_metric_value"]
            )
            print(
                "Recovered best model-selection state "
                f"step={selected_checkpoint_metadata['global_step']} "
                f"{selection_metric}={best_selection_value:.6f}",
                flush=True,
            )
            selected_step = int(selected_checkpoint_metadata["global_step"])
            if not any(
                int(row["optimizer_step"]) == selected_step
                for row in selection_history
            ):
                selection_history.append(
                    {
                        "optimizer_step": selected_step,
                        "epoch_index": int(
                            selected_checkpoint_metadata["epoch_index"]
                        ),
                        "next_batch_index": int(
                            selected_checkpoint_metadata["next_batch_index"]
                        ),
                        "became_best": True,
                        "ranking": selected_checkpoint_metadata[
                            "development_metrics"
                        ],
                        "recovered_from_best_checkpoint": True,
                    }
                )
                selection_history.sort(key=selection_history_sort_key)

    # Step zero is useful evidence: training should have to beat the untouched
    # reranker. On a retry with an existing best checkpoint, this work is not
    # repeated.
    if development_loader is not None and selected_checkpoint_metadata is None:
        development_ranking = evaluate_development(
            args,
            model,
            development_loader,
            device,
            yes_no_token_tensor,
            global_step,
        )
        best_selection_value = float(development_ranking[selection_metric])
        selection_history.append(
            {
                "optimizer_step": global_step,
                "epoch_index": checkpoint_progress.epoch_index,
                "next_batch_index": checkpoint_progress.next_batch_index,
                "became_best": True,
                "ranking": development_ranking,
            }
        )
        save_best_training_checkpoint(
            args.checkpoint_dir,
            run_signature,
            model,
            optimizer,
            scheduler,
            scaler,
            CheckpointProgress(
                epoch_index=checkpoint_progress.epoch_index,
                next_batch_index=checkpoint_progress.next_batch_index,
                global_step=global_step,
                training_history=list(training_history),
                accumulated_loss=checkpoint_progress.accumulated_loss,
                correct_pairs=checkpoint_progress.correct_pairs,
                observed_pairs=checkpoint_progress.observed_pairs,
                pair_type_counts=dict(checkpoint_progress.pair_type_counts),
                selection_history=list(selection_history),
            ),
            selection_metric,
            best_selection_value,
            development_ranking,
        )
        _, selected_checkpoint_metadata = best_checkpoint_metadata(
            args.checkpoint_dir, run_signature
        )

    for epoch in range(checkpoint_progress.epoch_index, args.epochs):
        epoch_pairs = shuffled_epoch_items(
            sample_pairs(args, train_query_ids, train_labels, args.seed + epoch),
            seed=args.seed + 1_000_003 + epoch,
        )
        total_epoch_batches = math.ceil(len(epoch_pairs) / args.train_batch_size)
        if epoch == checkpoint_progress.epoch_index:
            first_batch_index = checkpoint_progress.next_batch_index
            accumulated_loss = checkpoint_progress.accumulated_loss
            correct_pairs = checkpoint_progress.correct_pairs
            observed_pairs = checkpoint_progress.observed_pairs
            pair_type_counts: dict[str, int] = defaultdict(
                int, checkpoint_progress.pair_type_counts
            )
        else:
            first_batch_index = 0
            accumulated_loss = 0.0
            correct_pairs = 0
            observed_pairs = 0
            pair_type_counts = defaultdict(int)
        if first_batch_index > total_epoch_batches:
            raise RuntimeError(
                f"Checkpoint batch {first_batch_index} exceeds epoch size "
                f"{total_epoch_batches}"
            )
        remaining_pairs = remaining_epoch_items(
            epoch_pairs, args.train_batch_size, first_batch_index
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed + 2_000_003 + epoch)
        train_loader = DataLoader(
            PairDataset(train_products, remaining_pairs),
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=pair_collator,
            num_workers=args.num_workers,
            pin_memory=True,
            generator=loader_generator,
        )
        progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            total=total_epoch_batches,
            initial=first_batch_index,
        )
        for batch_index, batch in enumerate(
            progress, start=first_batch_index + 1
        ):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            grade_gaps = batch["grade_gaps"].to(device, non_blocking=True)
            for pair_type in batch["pair_types"]:
                pair_type_counts[pair_type] += 1

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, flat_scores = forward_yes_no(
                    model,
                    input_ids,
                    attention_mask,
                    yes_no_token_tensor,
                )
                paired_scores = flat_scores.reshape(-1, 2)
                margins = paired_scores[:, 0] - paired_scores[:, 1]

                # Each row above is one product's native Qwen log-odds:
                #
                #   s(p) = z_yes(p) - z_no(p)
                #
                # PairCollator orders rows [higher, lower], so RankNet uses:
                #
                #   margin = s(higher) - s(lower)
                #   loss   = -log(sigmoid(margin))
                #          = softplus(-margin)
                #
                # This is the customization relative to Qwen's reference
                # pointwise inference. ESCI does not become a four-class head;
                # its labels only decide which same-query product is "higher".
                # Backpropagation follows loss -> margin -> yes/no logits ->
                # decoder hidden states. Frozen base/head weights are not
                # updated, while PEFT's trainable LoRA A/B matrices receive
                # gradients through that unchanged computation graph.
                per_pair_loss = functional.softplus(-margins.float())
                if args.pair_weighting == "grade_gap":
                    loss = (per_pair_loss * grade_gaps).sum() / grade_gaps.sum()
                else:
                    loss = per_pair_loss.mean()

                # Most accumulation groups contain the configured number of
                # batches. The final group can be shorter; dividing it by the
                # full size would silently shrink its gradient contribution.
                final_group_size = (
                    total_epoch_batches % args.gradient_accumulation_steps
                    or args.gradient_accumulation_steps
                )
                final_group_start = total_epoch_batches - final_group_size + 1
                accumulation_divisor = (
                    final_group_size
                    if batch_index >= final_group_start
                    else args.gradient_accumulation_steps
                )
                scaled_loss = loss / accumulation_divisor

            # Sample query: "wireless mouse"
            # Product A: wireless gaming mouse       E
            # Product B: Bluetooth trackball         S
            # Product C: mouse pad                   C
            # Product D: USB coffee warmer           I
            #
            # Then the model learns:
            # score(query, exact product)
            #     >
            # score(query, substitute)
            #     >
            # score(query, complement)
            #    >
            # score(query, irrelevant product)
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach())
            correct_pairs += int((margins.detach() > 0).sum())
            observed_pairs += int(margins.numel())

            should_step = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == total_epoch_batches
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(
                    loss=f"{accumulated_loss / batch_index:.4f}",
                    pair_acc=f"{correct_pairs / observed_pairs:.3f}",
                    step=global_step,
                )
                final_training_step = (
                    epoch == args.epochs - 1
                    and batch_index == total_epoch_batches
                )
                should_evaluate_development = (
                    development_loader is not None
                    and (
                        global_step % args.selection_every_steps == 0
                        or final_training_step
                    )
                )
                if should_evaluate_development:
                    development_ranking = evaluate_development(
                        args,
                        model,
                        development_loader,
                        device,
                        yes_no_token_tensor,
                        global_step,
                    )
                    selection_value = float(
                        development_ranking[selection_metric]
                    )
                    became_best = (
                        selection_value
                        > best_selection_value + args.selection_min_delta
                    )
                    selection_history.append(
                        {
                            "optimizer_step": global_step,
                            "epoch_index": epoch,
                            "next_batch_index": batch_index,
                            "became_best": became_best,
                            "ranking": development_ranking,
                        }
                    )
                    if became_best:
                        best_selection_value = selection_value
                        save_best_training_checkpoint(
                            args.checkpoint_dir,
                            run_signature,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            CheckpointProgress(
                                epoch_index=epoch,
                                next_batch_index=batch_index,
                                global_step=global_step,
                                training_history=list(training_history),
                                accumulated_loss=accumulated_loss,
                                correct_pairs=correct_pairs,
                                observed_pairs=observed_pairs,
                                pair_type_counts=dict(pair_type_counts),
                                selection_history=list(selection_history),
                            ),
                            selection_metric,
                            selection_value,
                            development_ranking,
                        )
                        _, selected_checkpoint_metadata = (
                            best_checkpoint_metadata(
                                args.checkpoint_dir, run_signature
                            )
                        )

                should_save_recovery = (
                    args.checkpoint_dir is not None
                    and (
                        global_step % args.checkpoint_every_steps == 0
                        or should_evaluate_development
                    )
                )
                if should_save_recovery:
                    save_training_checkpoint(
                        args.checkpoint_dir,
                        run_signature,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        CheckpointProgress(
                            epoch_index=epoch,
                            next_batch_index=batch_index,
                            global_step=global_step,
                            training_history=list(training_history),
                            accumulated_loss=accumulated_loss,
                            correct_pairs=correct_pairs,
                            observed_pairs=observed_pairs,
                            pair_type_counts=dict(pair_type_counts),
                            selection_history=list(selection_history),
                        ),
                        args.checkpoints_to_keep,
                    )
        training_history.append(
            {
                "epoch": epoch + 1,
                "mean_ranknet_loss": accumulated_loss / total_epoch_batches,
                "pair_accuracy": correct_pairs / observed_pairs,
                "sampled_pair_count": len(epoch_pairs),
                "pair_type_counts": dict(sorted(pair_type_counts.items())),
                "optimizer_steps_total": global_step,
            }
        )
        # Always persist a clean epoch boundary. If eviction occurred after the
        # last periodic checkpoint but before this write, the retry finishes
        # only the missing tail and then creates this boundary checkpoint.
        if args.checkpoint_dir is not None:
            save_training_checkpoint(
                args.checkpoint_dir,
                run_signature,
                model,
                optimizer,
                scheduler,
                scaler,
                CheckpointProgress(
                    epoch_index=epoch + 1,
                    next_batch_index=0,
                    global_step=global_step,
                    training_history=list(training_history),
                    selection_history=list(selection_history),
                ),
                args.checkpoints_to_keep,
            )

    if development_loader is not None:
        _, selected_checkpoint_metadata = load_best_model_checkpoint(
            args.checkpoint_dir, run_signature, model
        )
        print(
            "Final evaluation and export will use the selected best adapter, "
            "not necessarily the latest recovery state.",
            flush=True,
        )

    metrics = evaluate_and_write_outputs(
        args,
        model,
        eval_loader,
        device,
        yes_no_token_tensor,
        eval_frame,
        instruction,
        training_history,
        train_frame,
        selection_history=selection_history,
        selected_checkpoint=selected_checkpoint_metadata,
    )

    # model_output was validated above for every non-evaluation-only run.
    args.model_output.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.model_output / "adapter"
    merged_dir = args.model_output / "merged"
    model.eval()
    model.save_pretrained(adapter_dir, safe_serialization=True)
    merged_model = model.merge_and_unload()
    merged_model.config.use_cache = True
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    shutil.copy2(args.prompt_file, args.model_output / "esci_v1.txt")
    if args.experiment_config_file is not None:
        shutil.copy2(
            args.experiment_config_file,
            args.model_output / "experiment_config_source.yaml",
        )
    (args.model_output / "training_metadata.json").write_text(
        json.dumps(metrics["configuration"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.model_output / "model_selection.json").write_text(
        json.dumps(metrics["model_selection"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter and merged model to {args.model_output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
