"""
Generate README figures from local W&B run files and the ablation CSV.

Reads:
    - WANDB_DIR/run-*/run-*.wandb (extracted via wandb's DataStore + protobuf)
    - results/experiment_results.csv

Writes:
    - results/figures/training_curves.png   (Chart 1: LR sweep reward curves)
    - results/figures/ablations.png         (Chart 2: all experiments bar chart)
    - results/figures/reward_entropy.png    (Chart 3: reward + entropy for best run)
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb


# ---------------------------------------------------------------------------
# Config: which W&B run maps to which experiment label
# ---------------------------------------------------------------------------

WANDB_DIR = Path(
    "/Users/kaipengwu/Documents/CS336/Stanford-CS336-main/"
    "assignment5-alignment-main/logs/wandb"
)

RUN_MAP = {
    "lr_5e-6":  "run-20260314_061347-5xv28gc8",
    "lr_2e-5":  "run-20260314_075918-i36saay4",
    "baseline": "run-20260314_092234-mguiya5x",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "results" / "figures"
CSV_PATH = REPO_ROOT / "results" / "experiment_results.csv"


# ---------------------------------------------------------------------------
# W&B binary extraction
# ---------------------------------------------------------------------------

def load_history(run_dir: Path) -> list[dict]:
    """Parse a run-*.wandb protobuf stream into a list of history dicts."""
    wandb_file = next(run_dir.glob("*.wandb"))
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))

    rows = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            continue
        if data is None:
            break
        try:
            rec = pb.Record()
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for it in rec.history.item:
            key = ".".join(it.nested_key) if it.nested_key else it.key
            try:
                row[key] = json.loads(it.value_json)
            except Exception:
                row[key] = it.value_json
        rows.append(row)
    return rows


def extract_series(rows: list[dict], y_key: str, x_key: str = "grpo_step"):
    """Pick out (x, y) pairs where both keys are present, sorted by x."""
    pairs = [(r[x_key], r[y_key]) for r in rows if x_key in r and y_key in r]
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        return np.array([]), np.array([])
    xs, ys = zip(*pairs)
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Chart 1 — LR sweep training curves
# ---------------------------------------------------------------------------

def chart_training_curves(histories: dict[str, list[dict]]):
    fig, ax = plt.subplots(figsize=(8, 5))

    style = {
        "lr_2e-5":  dict(color="#2ca02c", label="LR = 2e-5 (best, 74.2%)",    lw=2.2),
        "baseline": dict(color="#1f77b4", label="LR = 1e-5 (baseline, 67.4%)", lw=2.2),
        "lr_5e-6":  dict(color="#d62728", label="LR = 5e-6 (too low, 27.7%)", lw=2.2),
    }

    for name in ["lr_2e-5", "baseline", "lr_5e-6"]:
        xs, ys = extract_series(histories[name], "grpo/mean_reward")
        # Smooth with a small moving average for readability
        if len(ys) > 5:
            ys = np.convolve(ys, np.ones(5) / 5, mode="valid")
            xs = xs[4:]
        ax.plot(xs, ys, **style[name])

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward (fraction of rollouts correct)")
    ax.set_title("GRPO training: learning rate is the dominant hyperparameter")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    out = FIG_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Chart 2 — Ablation bar chart from CSV
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "lr_2e-5":             "LR = 2e-5  (best)",
    "minimal_prompt":      "Minimal prompt",
    "offpolicy_k4_noclip": "Off-policy K=4, no clip",
    "baseline":            "Baseline  (LR = 1e-5)",
    "length_norm":         "Length normalization",
    "offpolicy_k4_clip":   "Off-policy K=4, clipped",
    "offpolicy_k2_clip":   "Off-policy K=2, clipped",
    "no_std_norm":         "No std normalization",
    "lr_5e-6":             "LR = 5e-6  (too low)",
    "no_baseline":         "No baseline  (fails)",
}


def chart_ablations():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["val_accuracy"]), reverse=True)
    names = [r["experiment"] for r in rows]
    accs  = [float(r["val_accuracy"]) for r in rows]
    labels = [LABEL_MAP.get(n, n) for n in names]

    BEST = "#f5a623"     # gold
    FAIL = "#d0021b"     # red
    BASE = "#4a90e2"     # blue
    NEUTRAL = "#cfd2d6"  # light grey

    colors = []
    for n in names:
        if n == "lr_2e-5":
            colors.append(BEST)
        elif n == "no_baseline":
            colors.append(FAIL)
        elif n == "baseline":
            colors.append(BASE)
        else:
            colors.append(NEUTRAL)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    })

    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    y_pos = np.arange(len(names))
    bars = ax.barh(
        y_pos, accs,
        color=colors, edgecolor="white", linewidth=1.5, height=0.72,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.92)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%"], fontsize=10, color="#555")

    fig.text(
        0.04, 0.965,
        "Baseline subtraction is the difference between learning and not",
        fontsize=15, fontweight="bold", color="#1a1a1a",
    )
    fig.text(
        0.04, 0.925,
        "10 GRPO ablations · Qwen2.5-Math-1.5B · GSM8K validation accuracy",
        fontsize=10.5, color="#777",
    )

    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbb")
    ax.spines["bottom"].set_color("#bbb")

    ax.grid(axis="x", alpha=0.4, linestyle="--", color="#bbb", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    for bar, acc, name in zip(bars, accs, names):
        is_highlight = name in ("lr_2e-5", "no_baseline")
        text_color = (
            BEST if name == "lr_2e-5"
            else FAIL if name == "no_baseline"
            else "#333"
        )
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc * 100:.1f}%",
            va="center",
            fontsize=11 if is_highlight else 10,
            fontweight="bold" if is_highlight else "normal",
            color=text_color,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = FIG_DIR / "ablations.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Chart 3 — Reward + entropy two panel
# ---------------------------------------------------------------------------

def chart_reward_entropy(history: list[dict]):
    xs_r, ys_r = extract_series(history, "grpo/mean_reward")
    xs_e, ys_e = extract_series(history, "grpo/mean_entropy")

    # Smooth both
    def smooth(ys, k=5):
        if len(ys) < k:
            return ys
        return np.convolve(ys, np.ones(k) / k, mode="valid")

    ys_r_s = smooth(ys_r)
    xs_r_s = xs_r[len(xs_r) - len(ys_r_s):]
    ys_e_s = smooth(ys_e)
    xs_e_s = xs_e[len(xs_e) - len(ys_e_s):]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(xs_r_s, ys_r_s, color="#2ca02c", lw=2.2)
    ax1.set_ylabel("Mean reward")
    ax1.set_title("Training dynamics: reward rises as entropy falls (healthy RL run)")
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.3)

    ax2.plot(xs_e_s, ys_e_s, color="#9467bd", lw=2.2)
    ax2.set_ylabel("Mean token entropy")
    ax2.set_xlabel("Training step")
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "reward_entropy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading W&B histories...")
    histories = {}
    for name, run in RUN_MAP.items():
        run_dir = WANDB_DIR / run
        print(f"  {name}: {run_dir.name}")
        histories[name] = load_history(run_dir)

    print("Generating chart 1: training curves...")
    chart_training_curves(histories)

    print("Generating chart 2: ablations...")
    chart_ablations()

    print("Generating chart 3: reward + entropy...")
    chart_reward_entropy(histories["lr_2e-5"])

    print("Done.")


if __name__ == "__main__":
    main()
