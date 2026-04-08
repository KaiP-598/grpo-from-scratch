"""
GRPO training script for mathematical reasoning.

Reproduces the core training loop from DeepSeek-R1:
    1. Sample questions, generate G responses per question (vLLM)
    2. Score all responses, compute group-normalized advantages
    3. Tokenize, get reference log-probs (frozen snapshot)
    4. For K epochs on this batch: compute PG loss, backward, step
    5. Periodically evaluate on held-out MATH problems

Usage:
    python train.py \
        --model_path Qwen/Qwen2.5-Math-1.5B \
        --train_data data/train.jsonl \
        --val_data data/validation.jsonl \
        --output_dir outputs/grpo_baseline
"""
import json
import random
import argparse
import os

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams

from grpo.utils import tokenize_pairs, get_log_probs
from grpo.rewards import compute_advantages
from grpo.training import grpo_step
from grpo.grading import reasoning_reward, direct_answer_reward
from grpo.evaluation import load_prompt, load_jsonl, evaluate


# ---------------------------------------------------------------------------
# vLLM helpers (two-GPU setup: training on cuda:0, inference on cuda:1)
# ---------------------------------------------------------------------------

def create_inference_engine(model_id: str, device: str, seed: int, mem_util: float = 0.6):
    """Initialize vLLM on a dedicated GPU for fast batched generation."""
    from vllm.model_executor import set_random_seed as vllm_seed
    vllm_seed(seed)

    ws_patch = patch("torch.distributed.get_world_size", return_value=1)
    prof_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with ws_patch, prof_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=mem_util,
            enforce_eager=True,
        )


def sync_weights(policy, llm):
    """Copy training model weights into the vLLM engine."""
    state = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state.items())


def run_eval(llm, val_examples, sampling_params, prompt_name, reward_fn):
    """Evaluate on MATH validation set."""
    template = load_prompt(prompt_name)
    prompts = [template.format(question=ex["question"]) for ex in val_examples]
    truths = [ex["answer"] for ex in val_examples]
    return evaluate(llm, reward_fn, prompts, truths, sampling_params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training for math reasoning")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # GRPO hyperparameters
    parser.add_argument("--n_steps", type=int, default=200,
                        help="Number of GRPO rollout-train cycles")
    parser.add_argument("--rollout_batch", type=int, default=256,
                        help="Total rollouts per step = n_prompts * group_size")
    parser.add_argument("--group_size", type=int, default=8,
                        help="G: number of candidate responses per question")
    parser.add_argument("--train_batch", type=int, default=256)
    parser.add_argument("--grad_accum", type=int, default=256)
    parser.add_argument("--epochs_per_batch", type=int, default=1,
                        help="K: how many times to reuse each rollout batch")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--variant", type=str, default="with_baseline",
                        choices=["no_baseline", "with_baseline", "clipped", "unclipped"])
    parser.add_argument("--no_std_norm", action="store_true",
                        help="Disable std normalization in advantage computation")
    parser.add_argument("--use_length_norm", action="store_true",
                        help="Use length normalization instead of masked mean")
    parser.add_argument("--prompt", type=str, default="reasoning",
                        choices=["reasoning", "minimal"])
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0"

    micro_batch = args.train_batch // args.grad_accum
    n_prompts = args.rollout_batch // args.group_size

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        print("  Using flash_attention_2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        print("  Using SDPA fallback")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Inference engine on GPU 1
    # ------------------------------------------------------------------
    print("Starting inference engine...")
    engine = create_inference_engine(args.model_path, "cuda:1", args.seed)

    gen_params = SamplingParams(
        temperature=1.0, max_tokens=1024, min_tokens=4,
        n=args.group_size, stop=["</answer>"],
        include_stop_str_in_output=True, seed=args.seed,
    )
    eval_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True, seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 3. Data + optimizer
    # ------------------------------------------------------------------
    train_questions = load_jsonl(args.train_data)
    val_examples = load_jsonl(args.val_data)
    template = load_prompt(args.prompt)

    reward_fn = direct_answer_reward if args.prompt == "minimal" else reasoning_reward

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )

    wandb.init(project="grpo-math", config=vars(args))
    wandb.define_metric("step")
    wandb.define_metric("optim_step")
    wandb.define_metric("train/*", step_metric="optim_step")
    wandb.define_metric("eval/*", step_metric="step")

    global_optim_step = 0

    # ------------------------------------------------------------------
    # 4. GRPO loop
    # ------------------------------------------------------------------
    for step in range(args.n_steps):
        print(f"\n{'='*60}")
        print(f"Step {step + 1}/{args.n_steps}")
        print(f"{'='*60}")

        # 4a. Sample questions
        questions = random.sample(
            train_questions, min(n_prompts, len(train_questions)),
        )
        prompts = [template.format(question=q["question"]) for q in questions]
        truths = [q["answer"] for q in questions]

        # 4b. Generate G responses per question
        model.eval()
        sync_weights(model, engine)
        outputs = engine.generate(prompts, gen_params)

        # Flatten into (prompt, response, ground_truth) triples
        flat_prompts, flat_responses, flat_truths = [], [], []
        for prompt, output, gt in zip(prompts, outputs, truths):
            for completion in output.outputs:
                flat_prompts.append(prompt)
                flat_responses.append(completion.text)
                flat_truths.append(gt)

        # 4c. Compute group-normalized advantages
        advantages, raw_rewards, reward_stats = compute_advantages(
            reward_fn=reward_fn,
            responses=flat_responses,
            ground_truths=flat_truths,
            group_size=args.group_size,
            eps=args.advantage_eps,
            normalize_std=not args.no_std_norm,
        )
        print(f"  Reward: mean={reward_stats['mean_reward']:.4f}, "
              f"correct={reward_stats['fraction_correct']:.2%}")

        # 4d. Tokenize all rollouts
        batch = tokenize_pairs(flat_prompts, flat_responses, tokenizer)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)
        advantages = advantages.to(device)
        raw_rewards = raw_rewards.to(device)

        # 4e. Snapshot reference log-probs (computed once, reused across epochs)
        ref_log_probs_chunks = []
        entropy_chunks = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, input_ids.shape[0], micro_batch):
                j = i + micro_batch
                result = get_log_probs(
                    model, input_ids[i:j], labels[i:j], compute_entropy=True,
                )
                ref_log_probs_chunks.append(result["log_probs"])
                entropy_chunks.append(result["entropy"])
            ref_log_probs = torch.cat(ref_log_probs_chunks, dim=0).detach()
            token_entropy = torch.cat(entropy_chunks, dim=0).detach()
            del ref_log_probs_chunks, entropy_chunks
            torch.cuda.empty_cache()

        # 4f. Train for K epochs on this rollout batch
        model.train()
        n_rollouts = len(flat_responses)

        for epoch in range(args.epochs_per_batch):
            indices = list(range(n_rollouts))
            random.shuffle(indices)
            mb_count = 0

            for mb_start in range(0, n_rollouts, micro_batch):
                mb_idx = indices[mb_start:mb_start + micro_batch]
                if len(mb_idx) == 0:
                    continue

                mb_ids = input_ids[mb_idx]
                mb_labels = labels[mb_idx]
                mb_mask = response_mask[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_rew = raw_rewards[mb_idx]
                mb_ref = ref_log_probs[mb_idx]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    result = get_log_probs(model, mb_ids, mb_labels)

                loss, stats = grpo_step(
                    log_probs=result["log_probs"],
                    response_mask=mb_mask,
                    grad_accum_steps=args.grad_accum,
                    variant=args.variant,
                    raw_rewards=mb_rew,
                    advantages=mb_adv,
                    old_log_probs=mb_ref,
                    clip_eps=args.clip_eps,
                    use_length_normalization=args.use_length_norm,
                    normalizer=input_ids.shape[1],
                )

                mb_count += 1

                if mb_count % args.grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_optim_step += 1

                    wandb.log({
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/mean_reward": reward_stats["mean_reward"],
                        "train/fraction_correct": reward_stats["fraction_correct"],
                        "optim_step": global_optim_step,
                    })

        # 4g. Log per-step metrics
        wandb.log({
            "eval/mean_reward": reward_stats["mean_reward"],
            "eval/fraction_correct": reward_stats["fraction_correct"],
            "eval/mean_entropy": token_entropy[response_mask].mean().item()
                if response_mask.any() else 0.0,
            "step": step + 1,
        })

        # 4h. Periodic evaluation
        if (step + 1) % args.eval_every == 0:
            print(f"  Evaluating...")
            model.eval()
            sync_weights(model, engine)
            results = run_eval(engine, val_examples, eval_params, args.prompt, reward_fn)
            wandb.log({
                "eval/val_accuracy": results["accuracy"],
                "eval/val_format_accuracy": results["format_accuracy"],
                "step": step + 1,
            })
            print(f"  Val accuracy: {results['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # 5. Save + final eval
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    print("Final evaluation...")
    model.eval()
    sync_weights(model, engine)
    results = run_eval(engine, val_examples, eval_params, args.prompt, reward_fn)
    print(f"Final accuracy: {results['accuracy']:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
