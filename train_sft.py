"""
Supervised Fine-Tuning (SFT) on math reasoning data.

SFT is the first stage before GRPO: teach the model the expected
input/output format by maximizing log-likelihood on correct solutions.

Usage:
    python train_sft.py \
        --model_path Qwen/Qwen2.5-Math-1.5B \
        --train_data data/sft.jsonl \
        --val_data data/validation.jsonl \
        --output_dir outputs/sft_model
"""
import json
import random
import argparse
import os

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from grpo.utils import tokenize_pairs, get_log_probs
from grpo.training import sft_step
from grpo.grading import reasoning_reward
from grpo.evaluation import load_prompt, load_jsonl, evaluate
from train import create_inference_engine, sync_weights, run_eval


def main():
    parser = argparse.ArgumentParser(description="SFT for math reasoning")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Limit training to first N examples")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_gpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0"
    micro_batch = args.batch_size // args.grad_accum

    # Load model
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Inference engine for periodic evaluation
    engine = None
    eval_params = None
    if not args.single_gpu:
        print("Starting inference engine on cuda:1...")
        engine = create_inference_engine(args.model_path, "cuda:1", args.seed)
        eval_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024,
            stop=["</answer>"], include_stop_str_in_output=True, seed=args.seed,
        )

    # Data
    train_data = load_jsonl(args.train_data)
    if args.n_examples:
        train_data = train_data[:args.n_examples]
    val_data = load_jsonl(args.val_data)
    print(f"Training on {len(train_data)} examples")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )

    wandb.init(project="grpo-sft", config=vars(args))
    optim_step = 0
    mb_step = 0

    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        random.shuffle(train_data)

        for i in range(0, len(train_data), micro_batch):
            batch = train_data[i:i + micro_batch]
            prompts = [ex["prompt"] for ex in batch]
            responses = [ex["response"] for ex in batch]

            tokens = tokenize_pairs(prompts, responses, tokenizer)
            input_ids = tokens["input_ids"].to(device)
            labels = tokens["labels"].to(device)
            mask = tokens["response_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                result = get_log_probs(model, input_ids, labels)

            loss, _ = sft_step(
                log_probs=result["log_probs"],
                response_mask=mask,
                grad_accum_steps=args.grad_accum,
            )

            mb_step += 1

            if mb_step % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                optim_step += 1

                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train_step": optim_step,
                })
                print(f"Step {optim_step} | Loss: {loss.item():.4f}")

                if engine and optim_step % args.eval_every == 0:
                    model.eval()
                    sync_weights(model, engine)
                    results = run_eval(
                        engine, val_data, eval_params, "reasoning", reasoning_reward,
                    )
                    wandb.log({"eval/accuracy": results["accuracy"], "eval_step": optim_step})
                    print(f"  Val accuracy: {results['accuracy']:.4f}")
                    model.train()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
