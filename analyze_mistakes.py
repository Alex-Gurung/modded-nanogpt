#!/usr/bin/env python
import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_logs(path: str) -> pd.DataFrame:
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
    if not rows:
        raise RuntimeError(f"No valid JSON lines found in {path}")
    return pd.DataFrame(rows)


def basic_summary(df: pd.DataFrame):
    print("==== Overall summary by phase ====")
    print(
        df.groupby("phase")[
            ["token_loss_mean", "token_loss_q50", "token_loss_q90", "accuracy",
             "p_correct_mean", "frac_high_conf", "frac_high_conf_wrong"]
        ].describe()
    )

    print("\n==== Example rows ====")
    print(df.head())


def plot_time_series(df: pd.DataFrame, out_dir: Path):
    phases = df["phase"].unique()
    for phase in phases:
        sub = df[df["phase"] == phase].sort_values("step")
        if sub.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax = axes[0, 0]
        ax.plot(sub["step"], sub["token_loss_mean"], label="token_loss_mean")
        ax.set_title(f"{phase}: token_loss_mean")
        ax.set_xlabel("step")

        ax = axes[0, 1]
        ax.plot(sub["step"], sub["accuracy"], label="accuracy")
        ax.set_title(f"{phase}: accuracy")
        ax.set_xlabel("step")

        if "opt0_grad_norm" in sub.columns:
            ax = axes[1, 0]
            ax.plot(sub["step"], sub["opt0_grad_norm"], label="opt0_grad_norm")
            if "opt1_grad_norm" in sub.columns:
                ax.plot(sub["step"], sub["opt1_grad_norm"], label="opt1_grad_norm")
            ax.set_title(f"{phase}: grad norms")
            ax.legend()
            ax.set_xlabel("step")

        ax = axes[1, 1]
        ax.plot(sub["step"], sub["frac_high_conf"], label="frac_high_conf")
        ax.plot(sub["step"], sub["frac_high_conf_wrong"], label="frac_high_conf_wrong")
        ax.set_title(f"{phase}: high confidence behavior")
        ax.legend()
        ax.set_xlabel("step")

        fig.tight_layout()
        out_path = out_dir / f"{phase}_time_series.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")


def histogram_calibration(df: pd.DataFrame, out_dir: Path):
    # This uses the aggregate statistics; for true calibration you’d need raw buckets.
    # Still, you can see trends in mean p_correct vs accuracy.
    train = df[df["phase"] == "train"].sort_values("step")
    val = df[df["phase"] == "val"].sort_values("step")

    fig, ax = plt.subplots(figsize=(8, 5))
    if not train.empty:
        ax.scatter(train["p_correct_mean"], train["accuracy"], s=10, alpha=0.5, label="train")
    if not val.empty:
        ax.scatter(val["p_correct_mean"], val["accuracy"], s=20, alpha=0.8, label="val")

    ax.set_xlabel("mean p_correct")
    ax.set_ylabel("accuracy")
    ax.set_title("Calibration scatter (mean p_correct vs accuracy)")
    ax.legend()
    out_path = out_dir / "calibration_scatter.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_mistakes.py logs/<run_id>_train_mistakes.jsonl")
        sys.exit(1)

    path = Path(sys.argv[1])
    df = load_logs(str(path))

    basic_summary(df)

    out_dir = path.parent / f"{path.stem}_plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    plot_time_series(df, out_dir)
    histogram_calibration(df, out_dir)


if __name__ == "__main__":
    main()

