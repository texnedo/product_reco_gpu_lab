"""Shared prompt encoding and score computation for training and serving.

Keeping this code in one module prevents the online service from silently
using a different prompt boundary, truncation rule, or yes/no score than the
training and evaluation job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional


# Native Qwen3-Reranker prompt contract. The experiment instruction is placed
# in <Instruct>; it explains what "meets the requirements" means for products.
RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANKER_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
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


def encode_reranker_input(
    tokenizer: Any,
    prompt_encoding: PromptEncoding,
    query: str,
    document: str,
) -> list[int]:
    """Encode one pair while preserving the suffix used as the score position.

    Only the query and document are truncated. Tokenizing and truncating the
    complete prompt would remove the assistant suffix from long products and
    change the position whose yes/no logits define the ranking score.
    """
    variable_ids = tokenizer.encode(
        f"{query}{prompt_encoding.document_separator}{document}",
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_encoding.max_variable_tokens,
    )
    return [
        *prompt_encoding.prefix_ids,
        *prompt_encoding.instruction_ids,
        *variable_ids,
        *prompt_encoding.suffix_ids,
    ]


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

    The reference reranker returns sigmoid(score). Sigmoid is strictly
    increasing, so sorting by that probability or the raw logit difference
    gives the same order. Computing only two vocabulary rows avoids allocating
    logits for every sequence position and vocabulary token.
    """
    causal_model = (
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )
    hidden_states = causal_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    last_positions = attention_mask.sum(dim=1) - 1
    final_hidden = hidden_states[
        torch.arange(input_ids.shape[0], device=input_ids.device),
        last_positions,
    ]

    output_layer = causal_model.get_output_embeddings()
    weights = output_layer.weight.index_select(0, yes_no_token_tensor)
    bias = (
        output_layer.bias.index_select(0, yes_no_token_tensor)
        if getattr(output_layer, "bias", None) is not None
        else None
    )
    logits = functional.linear(final_hidden, weights, bias)
    score = logits[:, 0] - logits[:, 1]
    return logits, score
