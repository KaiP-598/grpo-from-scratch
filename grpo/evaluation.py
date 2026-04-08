"""
Evaluation utilities for the MATH benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from vllm import LLM, SamplingParams


PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str = "reasoning") -> str:
    """Load a prompt template by name."""
    return (PROMPT_DIR / f"{name}.txt").read_text()


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate(
    llm: LLM,
    reward_fn: Callable,
    prompts: list[str],
    ground_truths: list[str],
    sampling_params: SamplingParams,
) -> dict:
    """
    Generate completions for all prompts and score them.

    Returns accuracy, format accuracy, and per-example details.
    """
    outputs = llm.generate(prompts, sampling_params)
    generated = [o.outputs[0].text for o in outputs]

    rewards = [reward_fn(gen, gt) for gen, gt in zip(generated, ground_truths)]

    n = len(rewards)
    return {
        "outputs": generated,
        "rewards": rewards,
        "accuracy": sum(r["reward"] for r in rewards) / n,
        "format_accuracy": sum(r["format_reward"] for r in rewards) / n,
    }
