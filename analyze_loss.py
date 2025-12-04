#!/usr/bin/env python
import sys
import json
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


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
        plt.figure(figsize=(8, 5))
        for run, run_df in sub.groupby("run"):
            plt.plot(run_df["step"], run_df["loss"], label=run)
        # reference lines: overall min/max and median per phase
        y_vals = sub["loss"].to_numpy()
        if y_vals.size > 0:
            y_med = float(pd.Series(y_vals).median())
            y_min = float(y_vals.min())
            y_max = float(y_vals.max())
            plt.axhline(y_med, color="gray", linestyle="--", alpha=0.6, label="median")
            plt.axhline(y_min, color="gray", linestyle=":", alpha=0.4, label="min/max")
            plt.axhline(y_max, color="gray", linestyle=":", alpha=0.4)
        plt.xlabel("step")
        plt.ylabel(f"{phase} loss")
        plt.title(f"{phase} loss vs step")
        # optional vertical guides at key steps (quartiles of logged steps)
        steps = sub["step"].to_numpy()
        if steps.size > 3:
            qs = [0.25, 0.5, 0.75]
            for q in qs:
                s_q = float(pd.Series(steps).quantile(q))
                plt.axvline(s_q, color="lightgray", linestyle="--", alpha=0.3)
        plt.legend()
        out_path = out_dir / f"{phase}_loss_compare.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
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
