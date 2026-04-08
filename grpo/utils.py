"""
Tokenization and log-probability utilities for language model training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_pairs(
    prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """
    Tokenize (prompt, response) pairs and build a response mask.

    For causal LM training, we only compute loss on response tokens.
    The prompt tokens are "given" — the model didn't generate them.

    Returns input_ids, labels (shifted by 1), and response_mask.
    """
    all_input_ids = []
    all_labels = []
    all_masks = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        full_ids = prompt_ids + response_ids

        # Standard causal LM shift: input[t] predicts label[t] = input[t+1]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # Mark which positions in labels correspond to response tokens
        prompt_len = len(prompt_ids)
        mask = [0] * len(labels)
        for i in range(prompt_len - 1, len(labels)):
            mask[i] = 1

        all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        all_labels.append(torch.tensor(labels, dtype=torch.long))
        all_masks.append(torch.tensor(mask, dtype=torch.bool))

    # Pad to uniform length
    max_len = max(t.shape[0] for t in all_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _pad(tensors, fill_value, dtype):
        out = torch.full((len(tensors), max_len), fill_value, dtype=dtype)
        for i, t in enumerate(tensors):
            out[i, : t.shape[0]] = t
        return out

    return {
        "input_ids": _pad(all_input_ids, pad_id, torch.long),
        "labels": _pad(all_labels, pad_id, torch.long),
        "response_mask": _pad(all_masks, False, torch.bool),
    }


def get_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    compute_entropy: bool = False,
) -> dict[str, Tensor]:
    """
    Forward pass through the model, returning per-token log probabilities
    of the actual next token at each position.
    """
    outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(
        log_probs_all, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {"log_probs": log_probs}

    if compute_entropy:
        probs = torch.exp(log_probs_all)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        result["entropy"] = entropy

    return result


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average only the elements where mask=True."""
    return (values * mask).sum() / mask.sum().clamp(min=1)


def length_normalized_sum(values: Tensor, mask: Tensor, normalizer: float) -> Tensor:
    """
    Sum masked elements and divide by a fixed normalizer rather than
    the count of masked elements. Prevents short sequences from
    receiving disproportionately large per-token weight.
    """
    return (values * mask).sum() / normalizer
