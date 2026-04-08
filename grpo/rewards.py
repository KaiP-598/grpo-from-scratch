"""
Reward computation and group-based advantage normalization for GRPO.

The key insight of GRPO: instead of a learned value function (as in PPO),
normalize rewards within each group of responses to the same prompt.
This eliminates the need for a separate critic model.
"""
from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable


def compute_advantages(
    reward_fn: Callable,
    responses: list[str],
    ground_truths: list[str],
    group_size: int,
    eps: float = 1e-6,
    normalize_std: bool = True,
) -> tuple[Tensor, Tensor, dict]:
    """
    Score all responses and compute group-normalized advantages.

    For each prompt, we generated `group_size` candidate responses.
    Within each group:
        advantage = (reward - group_mean) / (group_std + eps)

    This gives learning signal even when most responses are wrong:
    correct answers get positive advantage, wrong ones get negative.
    The model learns both what works and what doesn't.

    Args:
        responses: flat list, length = n_prompts * group_size
        ground_truths: same length, repeated per group
        group_size: number of responses per prompt (G)

    Returns:
        advantages: (n_total,) normalized advantages
        raw_rewards: (n_total,) original 0/1 rewards
        stats: dict with mean_reward, fraction_correct, etc.
    """
    n_total = len(responses)
    assert n_total % group_size == 0

    # Score every response
    scores = []
    for resp, gt in zip(responses, ground_truths):
        r = reward_fn(resp, gt)
        scores.append(r["reward"])

    raw_rewards = torch.tensor(scores, dtype=torch.float32)

    # Reshape into groups of G and normalize within each group
    n_prompts = n_total // group_size
    grouped = raw_rewards.view(n_prompts, group_size)

    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True)

    if normalize_std:
        advantages = (grouped - group_mean) / (group_std + eps)
    else:
        advantages = grouped - group_mean

    stats = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "fraction_correct": (raw_rewards > 0).float().mean().item(),
    }

    return advantages.view(n_total), raw_rewards, stats
