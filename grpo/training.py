"""
Training step functions for SFT and GRPO.

Each function computes the loss for a single microbatch and calls
.backward(). Gradient accumulation happens at the caller level.
"""
from __future__ import annotations

from torch import Tensor
from typing import Literal

from grpo.utils import masked_mean, length_normalized_sum
from grpo.policy_gradient import compute_pg_loss


def sft_step(
    log_probs: Tensor,
    response_mask: Tensor,
    grad_accum_steps: int,
    normalizer: float = 1.0,
) -> tuple[Tensor, dict]:
    """
    SFT loss = negative log-likelihood over response tokens only.
    """
    loss = -length_normalized_sum(log_probs, response_mask, normalizer)
    (loss / grad_accum_steps).backward()
    return loss, {}


def grpo_step(
    log_probs: Tensor,
    response_mask: Tensor,
    grad_accum_steps: int,
    variant: str,
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    clip_eps: float | None = None,
    use_length_normalization: bool = False,
    normalizer: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    GRPO training step for one microbatch.

    1. Compute per-token policy gradient loss
    2. Average over response tokens (masked)
    3. Scale by gradient accumulation factor
    4. Backward pass
    """
    # Reshape scalar-per-sequence rewards/advantages for token-level broadcasting
    if raw_rewards is not None:
        raw_rewards = raw_rewards.unsqueeze(-1)
    if advantages is not None:
        advantages = advantages.unsqueeze(-1)

    per_token_loss, stats = compute_pg_loss(
        log_probs=log_probs,
        variant=variant,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        clip_eps=clip_eps,
    )

    if use_length_normalization:
        loss = length_normalized_sum(per_token_loss, response_mask, normalizer)
    else:
        loss = masked_mean(per_token_loss, response_mask)

    (loss / grad_accum_steps).backward()

    return loss, stats
