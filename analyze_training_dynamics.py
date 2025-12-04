#!/usr/bin/env python
import sys, json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_jsonl(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_training_dynamics.py logs/<run_id>_analysis.jsonl")
        return

    path = Path(sys.argv[1])
    rows = load_jsonl(str(path))
    df = pd.json_normalize(rows)

    out_dir = path.parent / f"{path.stem}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Global loss & deceleration (rough) --------
    train_df = df[df["type"] == "train_step"].sort_values("step")
    if not train_df.empty:
        plt.figure(figsize=(8,5))
        plt.plot(train_df["step"], train_df["loss_mean"], label="loss_mean")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("step (log)")
        plt.ylabel("loss_mean (log)")
        plt.title("Train loss (log-log)")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.savefig(out_dir / "train_loss_loglog.png")
        plt.close()

    # -------- Token-level behavior over time --------
    for col in ["acc_top1", "p_target_mean", "frac_solved_95", "overconf_wrong_frac"]:
        if col in train_df.columns:
            plt.figure(figsize=(8,5))
            plt.plot(train_df["step"], train_df[col])
            plt.xlabel("step")
            plt.ylabel(col)
            plt.title(col)
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / f"{col}.png")
            plt.close()

    # -------- Grad norms and ZSL proxies --------
    for col in ["global_grad_norm", "global_param_norm"]:
        if col in train_df.columns:
            plt.figure(figsize=(8,5))
            plt.plot(train_df["step"], train_df[col])
            plt.xlabel("step")
            plt.ylabel(col)
            plt.title(col)
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / f"{col}.png")
            plt.close()

    for col in ["zsl_mean_grad_cos", "zsl_frac_negative_cos", "zsl_grad_diversity"]:
        if col in train_df.columns:
            plt.figure(figsize=(8,5))
            plt.plot(train_df["step"], train_df[col])
            plt.xlabel("step")
            plt.ylabel(col)
            plt.title(f"ZSL proxy: {col}")
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / f"{col}.png")
            plt.close()

    # -------- Layer-wise stats (example: mlp_act_mean_mean) --------
    layer_cols = [c for c in train_df.columns if c.startswith("layers.layer_")]
    if layer_cols:
        # flatten per-layer stats manually
        # e.g. layers.layer_3.mlp_act_mean_mean
        metrics = ["mlp_act_mean_mean", "attn_gate_mean_mean", "c_fc_g_norm"]
        for metric in metrics:
            plt.figure(figsize=(10,6))
            for layer_idx in range(0, 12):  # your model has 12 blocks
                col = f"layers.layer_{layer_idx}.{metric}"
                if col in train_df.columns:
                    plt.plot(train_df["step"], train_df[col], label=f"layer_{layer_idx}")
            plt.xlabel("step")
            plt.ylabel(metric)
            plt.title(metric)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / f"{metric}.png")
            plt.close()

    print(f"Wrote plots to {out_dir}")

if __name__ == "__main__":
    main()

