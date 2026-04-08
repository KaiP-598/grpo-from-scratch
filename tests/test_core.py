"""
Unit tests for pure-logic GRPO components.

These tests run on CPU with no model or vLLM dependency — they verify
the math behind masked reduction, policy gradient losses, and group
advantage normalization.

    pytest tests/test_core.py
"""
import math

import pytest
import torch

from grpo.utils import masked_mean, length_normalized_sum
from grpo.policy_gradient import (
    reinforce_loss,
    clipped_surrogate_loss,
    unclipped_importance_weighted_loss,
    compute_pg_loss,
)
from grpo.rewards import compute_advantages


# ---------------------------------------------------------------------------
# Masked reductions
# ---------------------------------------------------------------------------

def test_masked_mean_ignores_masked_positions():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    assert masked_mean(values, mask).item() == pytest.approx(1.5)


def test_masked_mean_empty_mask_does_not_divide_by_zero():
    values = torch.tensor([1.0, 2.0])
    mask = torch.tensor([0, 0], dtype=torch.bool)
    # Should clamp denominator to 1, not NaN
    assert masked_mean(values, mask).item() == 0.0


def test_length_normalized_sum_uses_fixed_divisor():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    # sum of masked = 3.0, normalizer = 10 -> 0.3
    assert length_normalized_sum(values, mask, 10.0).item() == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Policy gradient losses
# ---------------------------------------------------------------------------

def test_reinforce_loss_sign():
    # Positive advantage should produce negative loss for positive log_prob
    # (optimizer will then push log_prob even higher)
    adv = torch.tensor([[1.0]])
    log_probs = torch.tensor([[-0.5, -0.2]])
    loss = reinforce_loss(adv, log_probs)
    assert torch.allclose(loss, torch.tensor([[0.5, 0.2]]))


def test_clipped_surrogate_identity_when_ratio_is_one():
    # new == old -> ratio = 1, loss = -advantage
    log_probs = torch.tensor([[-1.0, -2.0]])
    old = log_probs.clone()
    adv = torch.tensor([[1.0, -1.0]])
    loss, stats = clipped_surrogate_loss(adv, log_probs, old, clip_eps=0.2)
    assert torch.allclose(loss, -adv)
    assert stats["clip_fraction"].sum().item() == 0


def test_clipped_surrogate_clips_large_ratio():
    # ratio = exp(1.0) ~ 2.718 >> 1.2, positive advantage
    # surr1 = 2.718 * 1.0, surr2 = 1.2 * 1.0
    # loss = -min(surr1, surr2) = -1.2
    log_probs = torch.tensor([[0.0]])
    old = torch.tensor([[-1.0]])
    adv = torch.tensor([[1.0]])
    loss, stats = clipped_surrogate_loss(adv, log_probs, old, clip_eps=0.2)
    assert loss.item() == pytest.approx(-1.2, abs=1e-5)
    assert stats["clip_fraction"].item() == 1.0


def test_unclipped_matches_manual_ratio():
    log_probs = torch.tensor([[0.0]])
    old = torch.tensor([[-1.0]])
    adv = torch.tensor([[1.0]])
    loss, _ = unclipped_importance_weighted_loss(adv, log_probs, old, clip_eps=0.2)
    assert loss.item() == pytest.approx(-math.exp(1.0), abs=1e-5)


def test_compute_pg_loss_dispatch():
    log_probs = torch.tensor([[-0.5]])
    old = torch.tensor([[-0.5]])
    raw = torch.tensor([[1.0]])
    adv = torch.tensor([[2.0]])

    loss_nb, _ = compute_pg_loss(log_probs, "no_baseline", raw, adv, old, 0.2)
    loss_wb, _ = compute_pg_loss(log_probs, "with_baseline", raw, adv, old, 0.2)
    # no_baseline uses raw_rewards, with_baseline uses advantages
    assert loss_nb.item() == pytest.approx(0.5)   # -1.0 * -0.5
    assert loss_wb.item() == pytest.approx(1.0)   # -2.0 * -0.5

    with pytest.raises(ValueError):
        compute_pg_loss(log_probs, "nonsense", raw, adv, old, 0.2)


# ---------------------------------------------------------------------------
# Group-normalized advantages
# ---------------------------------------------------------------------------

def _fake_reward(resp, gt):
    return {"reward": float(resp == gt), "format_reward": 1.0}


def test_compute_advantages_zero_mean_within_group():
    # 2 prompts, group_size=4; half correct in each group
    responses = ["A", "A", "B", "B",  "C", "D", "C", "D"]
    truths    = ["A", "A", "A", "A",  "C", "C", "C", "C"]
    adv, raw, stats = compute_advantages(
        _fake_reward, responses, truths, group_size=4, normalize_std=True,
    )
    assert raw.shape == (8,)
    assert adv.shape == (8,)
    # Each group has mean 0.5 -> normalized advantages sum to ~0 per group
    g1 = adv[:4].sum().item()
    g2 = adv[4:].sum().item()
    assert g1 == pytest.approx(0.0, abs=1e-5)
    assert g2 == pytest.approx(0.0, abs=1e-5)
    assert stats["fraction_correct"] == pytest.approx(0.5)


def test_compute_advantages_no_std_norm():
    responses = ["A", "A", "B", "B"]
    truths    = ["A", "A", "A", "A"]
    adv, _, _ = compute_advantages(
        _fake_reward, responses, truths, group_size=4, normalize_std=False,
    )
    # Rewards: [1,1,0,0], mean=0.5 -> advantages = [0.5,0.5,-0.5,-0.5]
    assert torch.allclose(adv, torch.tensor([0.5, 0.5, -0.5, -0.5]))
