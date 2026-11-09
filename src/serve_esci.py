#!/usr/bin/env python3
"""Serve a minimal ESCI candidate simulation and Qwen reranking endpoint.

The service deliberately does not implement retrieval yet. At startup it
samples complete query groups from ESCI and keeps those product candidates in
memory in a shuffled, deterministic order. A request supplies only one of the
sampled query strings; the service scores its candidates with the trained
Qwen reranker and returns the top results.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

from esci_core import TEXT_CLEANING_MODES, product_document, safe_text
from reranker_runtime import (
    PromptEncoding,
    encode_reranker_input,
    forward_yes_no,
    native_answer_token_ids,
)


LOGGER = logging.getLogger("esci-serving")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"


@dataclass(frozen=True)
class CandidateGroup:
    query: str
    products: tuple[dict[str, Any], ...]


class RankRequest(BaseModel):
    """The only online input; candidates come from the in-memory simulation."""

    query: str = Field(min_length=1, max_length=2_000)

    class Config:
        extra = "forbid"


def normalize_query(value: str) -> str:
    """Make lookup forgiving without changing the text sent to Qwen."""
    return " ".join(value.split()).casefold()


def load_candidate_groups(
    data_dir: Path,
    query_count: int,
    split: str,
    small_version: bool,
    seed: int,
) -> dict[str, CandidateGroup]:
    """Load complete product groups for a deterministic random query sample."""
    if query_count <= 0:
        raise ValueError("query-count must be positive")
    examples = data_dir / EXAMPLES_FILE
    products = data_dir / PRODUCTS_FILE
    missing = [str(path) for path in (examples, products) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ESCI data file(s): " + ", ".join(missing))

    # We sample query text rather than rows, then join every associated example.
    # Consequently candidate recall is 100% for this simulation; no retrieval
    # quality claim should be inferred from the endpoint's ranking metrics.
    small_filter = "AND e.small_version = 1" if small_version else ""
    query = f"""
        WITH eligible AS (
            SELECT *
            FROM read_parquet(?) e
            WHERE e.split = ? {small_filter}
        ),
        selected_queries AS (
            SELECT query
            FROM eligible
            GROUP BY query
            ORDER BY hash(query || ':{int(seed)}')
            LIMIT {int(query_count)}
        )
        SELECT
            e.example_id,
            e.query_id,
            e.query,
            e.product_id,
            e.esci_label,
            p.product_title,
            p.product_description,
            p.product_bullet_point,
            p.product_brand,
            p.product_color
        FROM eligible e
        JOIN selected_queries q USING (query)
        LEFT JOIN read_parquet(?) p
          ON e.product_id = p.product_id
         AND e.product_locale = p.product_locale
        ORDER BY e.query, e.query_id, e.example_id
    """
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            query, [str(examples), split, str(products)]
        )
        column_names = [description[0] for description in cursor.description]
        rows = [dict(zip(column_names, values)) for values in cursor.fetchall()]
    finally:
        connection.close()
    if not rows:
        raise RuntimeError(f"No candidate groups found for split={split!r}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_queries: dict[str, str] = {}
    for row in rows:
        query_text = safe_text(row["query"])
        key = normalize_query(query_text)
        original_queries.setdefault(key, query_text)
        grouped[key].append(row)

    result = {}
    for key, candidates in grouped.items():
        # Shuffle once at startup. The same request is reproducible, but the
        # reranker cannot benefit from the dataset's original row ordering.
        random.Random(f"{seed}:{key}").shuffle(candidates)
        result[key] = CandidateGroup(
            query=original_queries[key], products=tuple(candidates)
        )
    LOGGER.info(
        "Loaded %d products across %d simulated query groups from split=%s",
        len(rows),
        len(result),
        split,
    )
    return result


def right_pad(
    encoded: list[list[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the same right-padded tensors used by training and evaluation."""
    max_length = max(len(token_ids) for token_ids in encoded)
    input_ids = torch.full(
        (len(encoded), max_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, token_ids in enumerate(encoded):
        length = len(token_ids)
        input_ids[row_index, :length] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[row_index, :length] = 1
    return input_ids, attention_mask


class EsciRankService:
    """Serve candidates, optionally owning one serialized GPU model."""

    def __init__(
        self,
        model: Any | None,
        tokenizer: Any | None,
        device: torch.device | None,
        prompt_encoding: PromptEncoding | None,
        candidate_groups: dict[str, CandidateGroup],
        text_cleaning: str,
        batch_size: int,
        top_k: int,
        model_name: str | None,
    ) -> None:
        if batch_size <= 0 or top_k <= 0:
            raise ValueError("batch-size and top-k must be positive")
        model_parts = (model, tokenizer, device, prompt_encoding)
        if any(part is not None for part in model_parts) and not all(
            part is not None for part in model_parts
        ):
            raise ValueError(
                "model, tokenizer, device, and prompt encoding must be "
                "configured together"
            )
        if tokenizer is not None and tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer has no pad token ID")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_encoding = prompt_encoding
        self.candidate_groups = candidate_groups
        self.text_cleaning = text_cleaning
        self.batch_size = batch_size
        self.top_k = top_k
        self.model_name = model_name
        self.yes_no_token_tensor = None
        if tokenizer is not None:
            answer_ids = native_answer_token_ids(tokenizer)
            self.yes_no_token_tensor = torch.tensor(
                [answer_ids["yes"], answer_ids["no"]],
                dtype=torch.long,
                device=device,
            )
        # A single process should own a single GPU model. Requests wait here
        # instead of starting overlapping forwards with unpredictable memory.
        self.model_lock = threading.Lock()

    def available_query_examples(self, limit: int = 10) -> list[str]:
        return [
            group.query
            for group in list(self.candidate_groups.values())[:limit]
        ]

    def available_queries(self) -> list[dict[str, Any]]:
        """Describe the complete in-memory candidate sample."""
        return [
            {
                "query": group.query,
                "candidate_count": len(group.products),
            }
            for group in self.candidate_groups.values()
        ]

    def unranked_candidates(
        self, group: CandidateGroup, request_started: float
    ) -> dict[str, Any]:
        """Return the shuffled retrieval simulation without model scoring."""
        preprocessing_started = time.perf_counter()
        results = [
            {
                "input_position": input_position,
                "product_id": str(product["product_id"]),
                "title": safe_text(product["product_title"]),
                "esci_label": product["esci_label"],
            }
            for input_position, product in enumerate(group.products)
        ]
        preprocessing_ms = (
            time.perf_counter() - preprocessing_started
        ) * 1_000
        total_ms = (time.perf_counter() - request_started) * 1_000
        timing = {
            "queue_ms": 0.0,
            "preprocessing_ms": round(preprocessing_ms, 3),
            "inference_ms": 0.0,
            "total_ms": round(total_ms, 3),
        }
        LOGGER.info(
            "candidates_only query=%r candidates=%d total_ms=%.3f",
            group.query,
            len(results),
            total_ms,
        )
        return {
            "mode": "candidates_only",
            "query": group.query,
            "candidate_count": len(group.products),
            "returned_count": len(results),
            "model": None,
            "results": results,
            "timing": timing,
        }

    def rank(self, requested_query: str) -> dict[str, Any]:
        request_started = time.perf_counter()
        group = self.candidate_groups.get(normalize_query(requested_query))
        if group is None:
            raise KeyError(requested_query)
        if self.model is None:
            return self.unranked_candidates(group, request_started)

        queue_started = time.perf_counter()
        with self.model_lock:
            queue_ms = (time.perf_counter() - queue_started) * 1_000
            preprocessing_started = time.perf_counter()
            encoded = [
                encode_reranker_input(
                    self.tokenizer,
                    self.prompt_encoding,
                    group.query,
                    product_document(product, self.text_cleaning),
                )
                for product in group.products
            ]
            preprocessing_ms = (
                time.perf_counter() - preprocessing_started
            ) * 1_000

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_started = time.perf_counter()
            scores: list[float] = []
            with torch.inference_mode():
                for start in range(0, len(encoded), self.batch_size):
                    batch_ids, batch_mask = right_pad(
                        encoded[start : start + self.batch_size],
                        self.tokenizer.pad_token_id,
                    )
                    batch_ids = batch_ids.to(self.device)
                    batch_mask = batch_mask.to(self.device)
                    _, batch_scores = forward_yes_no(
                        self.model,
                        batch_ids,
                        batch_mask,
                        self.yes_no_token_tensor,
                    )
                    scores.extend(batch_scores.float().cpu().tolist())
            if self.device.type == "cuda":
                # CUDA launches are asynchronous. Synchronization makes the
                # reported inference time represent completed GPU work.
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - inference_started) * 1_000

        ranked = sorted(
            enumerate(zip(group.products, scores)),
            key=lambda item: (-item[1][1], item[0]),
        )
        results = []
        for rank, (input_position, (product, score)) in enumerate(
            ranked[: self.top_k], start=1
        ):
            results.append(
                {
                    "rank": rank,
                    "product_id": str(product["product_id"]),
                    "title": safe_text(product["product_title"]),
                    "score": score,
                    "input_position": input_position,
                    # This is returned only because this endpoint is a dataset
                    # lab. It is not read by the model and would not exist in
                    # a production catalog response.
                    "esci_label": product["esci_label"],
                }
            )
        total_ms = (time.perf_counter() - request_started) * 1_000
        timing = {
            "queue_ms": round(queue_ms, 3),
            "preprocessing_ms": round(preprocessing_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "total_ms": round(total_ms, 3),
        }
        LOGGER.info(
            "rank query=%r candidates=%d returned=%d queue_ms=%.3f "
            "preprocessing_ms=%.3f inference_ms=%.3f total_ms=%.3f",
            group.query,
            len(group.products),
            len(results),
            queue_ms,
            preprocessing_ms,
            inference_ms,
            total_ms,
        )
        return {
            "mode": "reranked",
            "query": group.query,
            "candidate_count": len(group.products),
            "returned_count": len(results),
            "model": self.model_name,
            "results": results,
            "timing": timing,
        }


def create_app(service: EsciRankService) -> FastAPI:
    app = FastAPI(title="Qwen ESCI serving lab", version="0.1.0")

    @app.get("/queries")
    def queries() -> dict[str, Any]:
        loaded_queries = service.available_queries()
        return {
            "query_count": len(loaded_queries),
            "queries": loaded_queries,
        }

    @app.post("/rank")
    def rank(request: RankRequest) -> dict[str, Any]:
        try:
            return service.rank(request.query)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Query is not in the startup sample",
                    "available_query_examples": service.available_query_examples(),
                },
            ) from None

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help=(
            "Merged model directory. When omitted, /rank returns shuffled "
            "unranked candidates for API and dataset debugging."
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Defaults to esci_v1.txt beside the merged model directory.",
    )
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--small-version", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--text-cleaning",
        choices=sorted(TEXT_CLEANING_MODES),
        default="basic_v1",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolved_prompt_file(model_dir: Path, configured: Path | None) -> Path:
    if configured is not None:
        return configured
    exported = model_dir.parent / "esci_v1.txt"
    if exported.is_file():
        return exported
    return PROJECT_ROOT / "config" / "prompts" / "esci_v1.txt"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.max_length <= 0 or not 1 <= args.port <= 65_535:
        raise ValueError("max-length must be positive and port must be 1..65535")
    if (
        args.model_dir is not None
        and args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("--device cuda requested, but CUDA is not available")

    candidate_groups = load_candidate_groups(
        args.data_dir,
        args.query_count,
        args.split,
        args.small_version,
        args.seed,
    )

    if args.model_dir is None:
        service = EsciRankService(
            model=None,
            tokenizer=None,
            device=None,
            prompt_encoding=None,
            candidate_groups=candidate_groups,
            text_cleaning=args.text_cleaning,
            batch_size=args.batch_size,
            top_k=args.top_k,
            model_name=None,
        )
        LOGGER.warning(
            "No --model-dir supplied; /rank will return all shuffled "
            "candidates without scores. Sampled query examples: %s",
            service.available_query_examples(),
        )
    else:
        prompt_file = resolved_prompt_file(args.model_dir, args.prompt_file)
        instruction = prompt_file.read_text(encoding="utf-8").strip()
        device = torch.device(args.device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        load_started = time.perf_counter()
        LOGGER.info("Loading merged model from %s onto %s", args.model_dir, device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            local_files_only=True,
            dtype=dtype,
        )
        model.config.use_cache = False
        model.to(device).eval()
        prompt_encoding = PromptEncoding.build(
            tokenizer, instruction, args.max_length
        )
        service = EsciRankService(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_encoding=prompt_encoding,
            candidate_groups=candidate_groups,
            text_cleaning=args.text_cleaning,
            batch_size=args.batch_size,
            top_k=args.top_k,
            model_name=args.model_dir.parent.name,
        )
        LOGGER.info(
            "Model ready in %.3f seconds; sampled query examples: %s",
            time.perf_counter() - load_started,
            service.available_query_examples(),
        )
    # One process owns the model. Do not add multiple Uvicorn workers: every
    # worker would independently allocate another full model on the same GPU.
    uvicorn.run(
        create_app(service),
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        workers=1,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
