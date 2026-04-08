"""
Policy gradient loss functions for GRPO.

Three variants implemented:
    1. REINFORCE (vanilla)         — no baseline, high variance
    2. REINFORCE with baseline     — group-normalized advantages, much lower variance
    3. Clipped policy gradient     — PPO-style trust region for off-policy training
"""
from __future__ import annotations

import torch
from torch import Tensor


def reinforce_loss(
    advantages: Tensor,
    log_probs: Tensor,
) -> Tensor:
    """
    Standard REINFORCE: loss = -advantage * log_prob(token).

    We minimize loss, so negating makes the optimizer increase
    probability of high-advantage tokens.

    advantages: (batch, 1) — one per sequence, broadcast across tokens
    log_probs:  (batch, seq_len) — per-token log probabilities
    """
    return -advantages * log_probs


def clipped_surrogate_loss(
    advantages: Tensor,
    log_probs: Tensor,
    old_log_probs: Tensor,
    clip_eps: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    PPO-style clipped surrogate objective for off-policy training.

    When reusing rollouts across multiple epochs (K > 1), the policy
    drifts from the version that generated the data. The importance
    sampling ratio corrects for this, and clipping prevents the ratio
    from growing too large (trust region).

    ratio = pi_new(token) / pi_old(token) = exp(log_new - log_old)
    L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    """
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    loss = -torch.min(surr1, surr2)

    clipped_frac = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
    stats = {"clip_fraction": clipped_frac.float(), "ratio": ratio}

    return loss, stats


def unclipped_importance_weighted_loss(
    advantages: Tensor,
    log_probs: Tensor,
    old_log_probs: Tensor,
    clip_eps: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Off-policy loss WITHOUT clipping. Uses the importance sampling ratio
    but does not constrain it. Useful as an ablation to measure the
    effect of the trust region.
    """
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    loss = -(ratio * advantages)

    clipped_frac = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
    stats = {"clip_fraction": clipped_frac.float(), "ratio": ratio}

    return loss, stats


def compute_pg_loss(
    log_probs: Tensor,
    variant: str,
    raw_rewards: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
    clip_eps: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Route to the appropriate loss function.

    Variants:
        "no_baseline"    — vanilla REINFORCE with raw rewards
        "with_baseline"  — REINFORCE with group-normalized advantages
        "clipped"        — PPO-style clipped surrogate (for K > 1)
        "unclipped"      — importance-weighted without clipping (ablation)
    """
    if variant == "no_baseline":
        return reinforce_loss(raw_rewards, log_probs), {}

    elif variant == "with_baseline":
        return reinforce_loss(advantages, log_probs), {}

    elif variant == "clipped":
        return clipped_surrogate_loss(
            advantages, log_probs, old_log_probs, clip_eps
        )

    elif variant == "unclipped":
        return unclipped_importance_weighted_loss(
            advantages, log_probs, old_log_probs, clip_eps
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")
