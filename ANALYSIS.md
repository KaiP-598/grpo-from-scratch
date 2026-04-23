# GRPO from Scratch: What I Learned Implementing DeepSeek-R1's Training Algorithm

I implemented Group Relative Policy Optimization (GRPO) — the core RL algorithm behind DeepSeek-R1 — from scratch in PyTorch, then ran 10 ablation experiments to understand what actually matters. Starting from a base model with ~19% accuracy on competition math, training reaches **74.2%**.

This post covers the five findings that surprised me most.

---

## What Is GRPO?

GRPO is a policy gradient algorithm designed to teach language models to reason correctly using only outcome-based rewards — no human labels, no process reward model.

The setup:
1. Sample a group of G responses from the current policy for each question
2. Check each response: did it get the right answer? (+1 reward) or wrong? (0 reward)
3. Compute relative advantages within the group — responses that did better than the group average get positive advantage, worse get negative
4. Update the policy to increase probability of high-advantage responses

The key insight vs vanilla REINFORCE: **relative advantages within a group** eliminate the need for a learned value function. You don't need to know "how good is this state?" — you just need to know "how much better than my peers was this response?"

```
For a group of 8 responses to one question:
  [correct, wrong, wrong, correct, wrong, correct, wrong, wrong]
  rewards:  [1,     0,     0,      1,     0,      1,     0,     0    ]
  mean:     0.375
  advantages: [+0.625, -0.375, -0.375, +0.625, -0.375, +0.625, -0.375, -0.375]
```

The policy is then updated to make correct responses more likely and wrong ones less likely — purely from outcome signal.

---

## The 10 Experiments

| Experiment | Val Accuracy |
|---|---|
| **Best: LR=2e-5** | **74.2%** |
| Minimal prompt | 72.1% |
| Off-policy K=4 (unclipped) | 68.8% |
| Baseline (200 steps, LR=1e-5) | 67.4% |
| Length normalization | 67.1% |
| Off-policy K=4 (clipped) | 60.3% |
| Off-policy K=2 (clipped) | 57.3% |
| No std normalization | 51.9% |
| LR=5e-6 | 27.7% |
| **No baseline** | **19.5%** |

---

## Finding 1: Baseline Subtraction Is Not Optional

**With baseline: 67.4%. Without: 19.5%.** The model completely fails to learn without it.

Without baseline subtraction, the gradient update is:

```
∇J = E[reward × ∇log π(response)]
```

When most responses are wrong (reward ≈ 0), most gradients are near zero. The rare correct responses produce a large positive gradient — but it's noisy because it's not relative to anything. The signal is too sparse for stable learning.

With baseline subtraction (GRPO's group mean):

```
∇J = E[(reward - mean_reward) × ∇log π(response)]
```

Now every response contributes signal. Wrong responses in a group where others got it right produce a *negative* gradient — the policy is pushed away from those responses even when reward is 0. This dense relative signal is what makes training stable.

**Interview answer:** "Baseline subtraction converts sparse absolute rewards into dense relative signals. Without it, the gradient is dominated by noise from zero-reward responses. It's the difference between 'this response was wrong' and 'this response was worse than your peers' — the latter is far more informative."

---

## Finding 2: Std Normalization Matters More Than Expected

**With std norm: 67.4%. Without: 51.9%.** A 15-point drop from removing one line of code.

The full advantage normalization is:

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

Removing the std division means advantages scale with the variance of rewards in the group. When the model is uncertain (high variance), advantages are large → large gradient steps → instability. When it's confident (low variance), advantages are small → weak signal → slow learning.

Std normalization keeps the effective learning rate constant regardless of how spread out the rewards are. It's gradient scale control baked into the reward signal.

---

## Finding 3: Off-Policy Reuse — Clipping Hurts at This Scale

Reusing each rollout K=4 times **without** clipping: 68.8%.  
Reusing K=4 times **with** PPO-style clipping (ε=0.2): 60.3%.  
Reusing K=2 times with clipping: 57.3%.

This surprised me. PPO clipping is standard practice — it's supposed to prevent the policy from moving too far from the distribution that generated the data. But at 1.5B scale on MATH, clipping actively hurts.

The likely explanation: the KL divergence between consecutive updates stays small enough that clipping's constraint is never needed, but the gradient truncation it causes (clipped ratios don't contribute to the gradient) reduces the effective batch size. You're paying the cost of clipping (weaker gradients) without getting the benefit (preventing divergence).

**Practical takeaway:** PPO clipping is a stabilizer for large policy updates. At small model scales with conservative LRs, it may constrain more than it protects. Always ablate clipping — don't assume it helps.

---

## Finding 4: Higher LR Wins, Until It Doesn't

LR=2e-5: **74.2%**. LR=1e-5: 67.4%. LR=5e-6: 27.7%.

The 5e-6 result is notable — at 100 steps it hasn't even reached baseline performance. The model is learning too slowly to escape the initial policy's distribution in the training budget.

The 2e-5 result shows the other direction: more aggressive updates make better use of a limited step budget. GRPO generates rollouts at inference speed (vLLM), which is expensive. Fewer, better gradient steps is the right tradeoff.

This is a GPU budget argument: if rollout generation costs 10× more than the gradient update, you want each update to count. Higher LR = more signal extracted per rollout batch.

---

## Finding 5: Minimal Prompts Are Surprisingly Competitive

Chain-of-thought prompt (`reasoning.txt`): 67.4%.  
Direct answer prompt (`minimal.txt`): **72.1%**.

The minimal prompt just says: `{question}` with no chain-of-thought instruction. The model figures out the reasoning format on its own from the SFT warmup.

Two interpretations:
1. The chain-of-thought instruction constrains the model to a specific reasoning format that may not be optimal for all problems
2. The minimal prompt gives the model more freedom to find its own reasoning strategy, which the reward signal then reinforces

Either way: don't assume more prompting is better. The model's SFT-learned format may already be well-calibrated. Explicit chain-of-thought instructions can add noise.

---

## Training Dynamics: What Healthy RL Looks Like

A healthy GRPO run shows two things happening simultaneously:
- **Reward rises** — the policy is finding better responses
- **Entropy falls** — the policy is becoming more confident (less exploratory)

![Reward and entropy over training](results/figures/reward_entropy.png)

This is the exploration-exploitation tradeoff playing out in real time. Early training: high entropy, low reward (exploring). Late training: low entropy, high reward (exploiting).

**What to watch for:**
- Entropy collapsing before reward rises = mode collapse (policy commits to wrong answers)
- Reward rising but entropy not falling = policy isn't converging (too noisy)
- Both moving together = healthy run

Monitoring entropy is the first line of defense. If entropy hits near-zero and reward is still low, the run is lost — the policy has committed to a wrong strategy and won't recover without a reset.

---

## What I'd Do Differently

**1. Adaptive KL penalty instead of hard clipping**  
PPO clipping is binary — gradients are either included or zeroed. A soft KL penalty (like in the original GRPO paper) degrades more gracefully. Worth implementing for a fair comparison.

**2. Longer runs with early stopping**  
All experiments were 100-200 steps due to compute budget. The LR=2e-5 run was still improving at step 100. With proper early stopping on validation accuracy, it likely goes higher.

**3. Group size ablation**  
All experiments used G=8. Larger groups (G=16, G=32) give more stable advantage estimates but cost proportionally more in generation. The optimal group size is likely task-dependent — worth measuring.

---

## Applied to Financial Reasoning

I applied GRPO to a different domain in the [FilingSense project](https://github.com/KaiP-598/filing-sense): answering numerical questions from SEC 10-K filings (FinQA benchmark).

The same algorithm, same reward structure (exact-match on numerical answer), different domain. Results:

| Stage | Accuracy |
|---|---|
| Base Qwen2.5-3B + RAG | 11.5% |
| + Full SFT | 16.5% |
| + GRPO | 17.0% |
| GRPO with gold context | **68%** |

The 68% gold-context result confirms the model learned multi-step financial reasoning. The 17% end-to-end result exposes retrieval as the bottleneck — a different problem than training.

**The lesson:** GRPO teaches *reasoning skill*, not *knowledge*. The same skill transfers across domains. What doesn't transfer is retrieval — you need domain-specific chunking and indexing for each new corpus.

---

## Code

Everything is in this repo. The core algorithm is ~200 lines across:
- [`grpo/policy_gradient.py`](grpo/policy_gradient.py) — ratio, clipping, advantage-weighted loss
- [`grpo/rewards.py`](grpo/rewards.py) — group normalization
- [`grpo/training.py`](grpo/training.py) — SFT and GRPO step functions
- [`train.py`](train.py) — full training loop with vLLM generation

Unit tests cover the math with no GPU required:
```bash
pytest tests/test_core.py
```
