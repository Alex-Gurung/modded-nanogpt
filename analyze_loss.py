#!/usr/bin/env python
import sys
import json
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_logs(paths: List[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        run = Path(path).stem.replace("_loss", "")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["run"] = run
                rows.append(row)
    if not rows:
        raise RuntimeError("No valid rows in provided files.")
    df = pd.DataFrame(rows)
    df["step"] = df["step"].astype(int)
    df["loss"] = df["loss"].astype(float)
    return df


def plot_runs(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    phases = df["phase"].unique()
    for phase in phases:
        sub = df[df["phase"] == phase].sort_values("step")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for run, run_df in sub.groupby("run"):
            ax.plot(run_df["step"], run_df["loss"], label=run)
        # even-spaced horizontal guides (default 0.1)
        tick_step = 0.1
        ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
        ax.grid(which="major", axis="y", linestyle="--", alpha=0.3)
        ax.set_xlabel("step")
        ax.set_ylabel(f"{phase} loss")
        ax.set_title(f"{phase} loss vs step")
        ax.legend()
        out_path = out_dir / f"{phase}_loss_compare.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_loss.py logs/<run1>_loss.jsonl [logs/<run2>_loss.jsonl ...]")
        sys.exit(1)
    df = load_logs(sys.argv[1:])
    out_dir = Path("loss_plots")
    plot_runs(df, out_dir)


if __name__ == "__main__":
    main()
