#!/usr/bin/env python
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.table import Table

DEFAULT_PLOT_WIDTH = 10
DEFAULT_PLOT_HEIGHT = 8
SAVE_DPI = 120
HIGHLIGHT_STEPS = [50, 100, 200, 800]
MAX_LAYER_PANELS = 12  # grid size for layer DI plots


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


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort numeric coercion for float cols (handles None/str)."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    numeric_candidates = [
        c for c in df.columns if df[c].dtype == object and df[c].apply(lambda x: isinstance(x, (int, float, np.number))).any()
    ]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


METRIC_DESCRIPTIONS = {
    "token_loss_mean": "Mean per-token loss (sampled top-k approx).",
    "accuracy": "Top-1 accuracy within sampled tokens.",
    "p_correct_mean": "Mean prob assigned to the correct token (top-k approx).",
    "frac_high_conf": "Frac tokens with p>=0.9 on predicted token.",
    "frac_high_conf_wrong": "Frac tokens with p>=0.9 but incorrect.",
    "loss_delta": "Change in token_loss_mean vs previous logged step (coarse trend).",
    "loss_deceleration": "Change in loss_delta vs previous (are gains slowing/accelerating).",
    "loss_delta_sign_flip": "1 if loss_delta flipped sign vs previous logged step (destructive/oscillatory).",
    "logit_mean": "Mean of logits per token (zero-sum drift).",
    "logit_mean_abs": "Mean absolute logit mean per token (zero-sum magnitude).",
    "logit_l2_mean": "Mean L2 norm of logits per token.",
    "topk_prob_mass_mean_est": "Avg mass within sampled top-k slice.",
    "zero_sum_residual": "|sum logits| / ||logits|| averaged (zero-sum residual).",
    "centered_norm_ratio": "||logits - mean|| / ||logits|| averaged (how much norm remains after centering).",
    "entropy_gap_vs_uniform": "log(V) - entropy (distance from uniform probs).",
    "destructive_grad": "Destructive interference on logits gradients (Eq.4 analog).",
    "destructive_loss": "Destructive interference on per-token loss improvements (Eq.3 analog).",
    "opt0_grad_norm": "Grad norm optimizer 0.",
    "opt1_grad_norm": "Grad norm optimizer 1.",
}

# Display-friendly short names
DISPLAY_COLS = {
    "token_loss_mean": "loss_mean",
    "accuracy": "acc",
    "p_correct_mean": "p_corr",
    "frac_high_conf": "high_conf",
    "frac_high_conf_wrong": "high_conf_wrong",
    "loss_delta": "d_loss",
    "loss_deceleration": "dd_loss",
    "loss_delta_sign_flip": "loss_flip",
    "logit_mean": "logit_mean",
    "logit_mean_abs": "logit_mean_abs",
    "logit_l2_mean": "logit_l2",
    "topk_prob_mass_mean_est": "topk_mass",
    "zero_sum_residual": "zs_residual",
    "centered_norm_ratio": "centered_ratio",
    "entropy_gap_vs_uniform": "entropy_gap",
    "destructive_grad": "di_grad",
    "destructive_loss": "di_loss",
    "opt0_grad_norm": "opt0_gn",
    "opt1_grad_norm": "opt1_gn",
    "di_update_update_all": "upd_all",
}


def print_descriptions():
    print("\n==== Metric descriptions ====")
    for k, v in METRIC_DESCRIPTIONS.items():
        print(f"- {k}: {v}")


def fmt_val(v, digits=4):
    try:
        if pd.isna(v):
            return "nan"
    except Exception:
        pass
    if isinstance(v, (int, float, np.number)):
        return f"{v:.{digits}g}"
    return str(v)


def render_table(title: str, rows: list[dict], columns: list[str], digits: int = 4):
    if not rows:
        return
    table = Table(title=title, show_lines=False)
    for col in columns:
        if col in DISPLAY_COLS:
            header = DISPLAY_COLS[col]
        elif col.startswith("di_update_update_type_"):
            header = col.replace("di_update_update_type_", "upd_type_")
        elif col.startswith("di_update_update_layer"):
            header = col.replace("di_update_update_layer", "upd_layer")
        else:
            header = col
        table.add_column(header)
    for row in rows:
        table.add_row(*[fmt_val(row.get(c), digits) for c in columns])
    rprint(table)


def basic_summary(df: pd.DataFrame):
    print("==== Overall summary by phase ====")
    cols = [
        "token_loss_mean",
        "accuracy",
        "p_correct_mean",
        "frac_high_conf",
        "frac_high_conf_wrong",
        "loss_delta",
        "loss_deceleration",
        "logit_mean",
        "logit_mean_abs",
        "zero_sum_residual",
        "centered_norm_ratio",
        "entropy_gap_vs_uniform",
        "destructive_grad",
        "destructive_loss",
        "di_update_update_all",
    ]
    present_cols = [c for c in cols if c in df.columns]
    summary = df.groupby("phase")[present_cols].agg(["mean", "median", "min", "max"])
    for phase in summary.index:
        rows = []
        for col in present_cols:
            if col not in summary.columns.levels[0]:
                continue
            stats = summary.loc[phase, col]
            rows.append({"metric": col, "mean": stats.get("mean"), "median": stats.get("median")})
        render_table(f"Summary ({phase})", rows, ["metric", "mean", "median"], digits=4)

    print("\n==== Latest rows (per phase) ====")
    for phase in df["phase"].unique():
        sub = df[df["phase"] == phase].sort_values("step")
        tail_cols = [c for c in present_cols if c in sub.columns]
        if tail_cols:
            head = sub[["step"] + tail_cols].head(8)
            tail = sub[["step"] + tail_cols].tail(8)
            rows_head = head.to_dict(orient="records")
            rows_tail = tail.to_dict(orient="records")
            render_table(f"First ({phase})", rows_head, ["step"] + tail_cols, digits=4)
            render_table(f"Latest ({phase})", rows_tail, ["step"] + tail_cols, digits=4)


def plot_time_series(df: pd.DataFrame, out_dir: Path):
    phases = df["phase"].unique()
    for phase in phases:
        sub = df[df["phase"] == phase].sort_values("step")
        if sub.empty:
            continue

        fig, axes = plt.subplots(3, 2, figsize=(DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT))

        ax = axes[0, 0]
        if "token_loss_mean" in sub.columns:
            ax.plot(sub["step"], sub["token_loss_mean"], label="token_loss_mean")
        if "token_loss_q50" in sub.columns:
            ax.plot(sub["step"], sub["token_loss_q50"], label="token_loss_q50", linestyle="--")
        if "token_loss_q90" in sub.columns:
            ax.plot(sub["step"], sub["token_loss_q90"], label="token_loss_q90", linestyle=":")
        ax.set_title(f"{phase}: token loss stats")
        ax.set_xlabel("step")
        ax.legend()

        ax = axes[0, 1]
        if "accuracy" in sub.columns:
            ax.plot(sub["step"], sub["accuracy"], label="accuracy")
        ax.set_title(f"{phase}: accuracy")
        ax.set_xlabel("step")

        ax = axes[1, 0]
        if "opt0_grad_norm" in sub.columns:
            ax.plot(sub["step"], sub["opt0_grad_norm"], label="opt0_grad_norm")
        if "opt1_grad_norm" in sub.columns:
            ax.plot(sub["step"], sub["opt1_grad_norm"], label="opt1_grad_norm")
        ax.set_title(f"{phase}: grad norms")
        ax.set_xlabel("step")
        ax.legend()

        ax = axes[1, 1]
        if "frac_high_conf" in sub.columns:
            ax.plot(sub["step"], sub["frac_high_conf"], label="frac_high_conf")
        if "frac_high_conf_wrong" in sub.columns:
            ax.plot(sub["step"], sub["frac_high_conf_wrong"], label="frac_high_conf_wrong")
        ax.set_title(f"{phase}: high-confidence behavior")
        ax.set_xlabel("step")
        ax.legend()

        ax = axes[2, 0]
        if "loss_delta" in sub.columns:
            ax.plot(sub["step"], sub["loss_delta"], label="loss_delta")
        if "loss_deceleration" in sub.columns:
            ax.plot(sub["step"], sub["loss_deceleration"], label="loss_deceleration", linestyle="--")
        ax.set_title(f"{phase}: loss change & deceleration")
        ax.set_xlabel("step")
        ax.legend()

        ax = axes[2, 1]
        if "logit_mean_abs" in sub.columns:
            ax.plot(sub["step"], sub["logit_mean_abs"], label="logit_mean_abs")
        if "logit_mean" in sub.columns:
            ax.plot(sub["step"], sub["logit_mean"], label="logit_mean", linestyle="--")
        if "topk_prob_mass_mean_est" in sub.columns:
            ax.plot(sub["step"], sub["topk_prob_mass_mean_est"], label="topk_prob_mass_est", linestyle=":")
        if "zero_sum_residual" in sub.columns:
            ax.plot(sub["step"], sub["zero_sum_residual"], label="zero_sum_residual", linestyle="-")
        if "centered_norm_ratio" in sub.columns:
            ax.plot(sub["step"], sub["centered_norm_ratio"], label="centered_norm_ratio", linestyle="-.")
        if "entropy_gap_vs_uniform" in sub.columns:
            ax.plot(sub["step"], sub["entropy_gap_vs_uniform"], label="entropy_gap_vs_uniform", linestyle=":")
        if "destructive_grad" in sub.columns:
            ax.plot(sub["step"], sub["destructive_grad"], label="destructive_grad", linestyle="--")
        if "destructive_loss" in sub.columns:
            ax.plot(sub["step"], sub["destructive_loss"], label="destructive_loss", linestyle="-.")
        ax.set_title(f"{phase}: zero-sum-ish signals")
        ax.set_xlabel("step")
        ax.legend()

        fig.tight_layout()
        out_path = out_dir / f"{phase}_time_series.png"
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_di_groups(df: pd.DataFrame, out_dir: Path):
    # param type plot
    for prefix, title in [("di_grad_type_", "DI by param type")]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        for phase in df["phase"].unique():
            sub = df[df["phase"] == phase].sort_values("step")
            if sub.empty:
                continue
            plt.figure(figsize=(DEFAULT_PLOT_WIDTH, 5.5))
            for col in sorted(cols):
                plt.plot(sub["step"], sub[col], label=DISPLAY_COLS.get(col, col))
            plt.xlabel("step")
            plt.ylabel("DI (1 - |sum g|/sum|g|)")
            plt.title(f"{title} ({phase})")
            plt.legend()
            out_path = out_dir / f"{phase}_{prefix}lines.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_path}")

    # layer-level grid: one panel per layer to avoid overplotting
    layer_cols = [c for c in df.columns if c.startswith("di_grad_layer")]
    if layer_cols:
        # parse layer indices
        layer_ids = sorted({int(c.split("layer")[1].split("_")[0]) for c in layer_cols})
        # cap number of panels
        layer_ids = layer_ids[:MAX_LAYER_PANELS]
        n_cols = 4
        n_rows = int(np.ceil(len(layer_ids) / n_cols))
        for phase in df["phase"].unique():
            sub = df[df["phase"] == phase].sort_values("step")
            if sub.empty:
                continue
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3), sharex=True, sharey=True)
            axes = axes.flatten()
            for ax in axes:
                ax.axis("off")
            for ax, lid in zip(axes, layer_ids):
                ax.axis("on")
                layer_cols_here = [c for c in layer_cols if f"layer{lid}_" in c]
                for col in sorted(layer_cols_here):
                    ax.plot(sub["step"], sub[col], label=DISPLAY_COLS.get(col, col))
                ax.set_title(f"layer {lid}")
            axes[0].set_ylabel("DI")
            for ax in axes:
                ax.grid(True, alpha=0.2)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right")
            fig.suptitle(f"DI by layer/type ({phase})")
            plt.tight_layout(rect=[0, 0, 0.88, 0.95])
            out_path = out_dir / f"{phase}_di_layer_grid.png"
            fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


def plot_loss_landscapes(df: pd.DataFrame, out_dir: Path, max_snapshots: int = 8):
    rows = df.dropna(subset=["loss_landscape"]) if "loss_landscape" in df.columns else pd.DataFrame()
    if rows.empty:
        return
    rows = rows.sort_values("step").head(max_snapshots)
    for _, row in rows.iterrows():
        snap = row["loss_landscape"]
        if not isinstance(snap, dict) or "alphas" not in snap or "delta_l" not in snap:
            continue
        alphas = snap["alphas"]
        curves = snap["delta_l"]
        plt.figure(figsize=(7, 5))
        for curve in curves:
            color = "green" if curve[-1] < 0 else "red"
            plt.plot(alphas, curve, color=color, alpha=0.3)
        plt.axvline(1.0, color="gray", linestyle="--", alpha=0.6)
        plt.axvline(0.0, color="black", linestyle=":", alpha=0.4)
        plt.xlabel("stepsize (alpha)")
        plt.ylabel("Δ loss (approx)")
        plt.title(f"Loss landscape approx @ step {row.get('step')} phase {row.get('phase')}")
        out_path = out_dir / f"loss_landscape_step{int(row.get('step',0))}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


def calibration_scatter(df: pd.DataFrame, out_dir: Path):
    train = df[df["phase"] == "train"].sort_values("step")
    val = df[df["phase"] == "val"].sort_values("step")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not train.empty and "p_correct_mean" in train.columns and "accuracy" in train.columns:
        ax.scatter(train["p_correct_mean"], train["accuracy"], s=10, alpha=0.5, label="train")
    if not val.empty and "p_correct_mean" in val.columns and "accuracy" in val.columns:
        ax.scatter(val["p_correct_mean"], val["accuracy"], s=20, alpha=0.8, label="val")

    ax.set_xlabel("mean p_correct (approx from top-k)")
    ax.set_ylabel("accuracy")
    ax.set_title("Calibration scatter")
    ax.legend()

    out_path = out_dir / "calibration_scatter.png"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def nearest_rows_to_steps(df: pd.DataFrame, steps: list[int]) -> pd.DataFrame:
    res = []
    for phase in df["phase"].unique():
        phase_df = df[df["phase"] == phase].sort_values("step")
        if phase_df.empty:
            continue
        for s in steps:
            idx = (phase_df["step"] - s).abs().idxmin()
            row = phase_df.loc[idx].copy()
            row["phase"] = phase
            row["target_step"] = s
            res.append(row)
    return pd.DataFrame(res)


def plot_di_zsl(df: pd.DataFrame, out_dir: Path, steps: list[int]):
    for phase in df["phase"].unique():
        sub = df[df["phase"] == phase].sort_values("step")
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT * 0.7), sharex=True)

        ax = axes[0]
        if "destructive_loss" in sub.columns:
            ax.plot(sub["step"], sub["destructive_loss"], label="di_loss")
        if "destructive_grad" in sub.columns:
            ax.plot(sub["step"], sub["destructive_grad"], label="di_grad", linestyle="--")
        for s in steps:
            ax.axvline(s, color="gray", alpha=0.2, linestyle=":")
        ax.set_ylabel("Destructive interference")
        ax.set_title(f"{phase}: DI over training")
        ax.legend()

        ax = axes[1]
        if "zero_sum_residual" in sub.columns:
            ax.plot(sub["step"], sub["zero_sum_residual"], label="zs_residual")
        if "centered_norm_ratio" in sub.columns:
            ax.plot(sub["step"], sub["centered_norm_ratio"], label="centered_ratio", linestyle="-.")
        if "entropy_gap_vs_uniform" in sub.columns:
            ax.plot(sub["step"], sub["entropy_gap_vs_uniform"], label="entropy_gap", linestyle=":")
        for s in steps:
            ax.axvline(s, color="gray", alpha=0.2, linestyle=":")
        ax.set_xlabel("step")
        ax.set_ylabel("Zero-sum signals")
        ax.set_title(f"{phase}: ZSL proxies over training")
        ax.legend()

        fig.tight_layout()
        out_path = out_dir / f"{phase}_di_zsl.png"
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def slope_vs_step(sub: pd.DataFrame, col: str):
    if len(sub) < 2 or col not in sub.columns:
        return np.nan
    x = sub["step"].to_numpy(dtype=float)
    y = sub[col].to_numpy(dtype=float)
    if np.allclose(y, y[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def trend_report(df: pd.DataFrame, window: int = None):
    span_desc = "all rows" if window is None else f"last {window} rows"
    print(f"\n==== Trend slopes ({span_desc}) ====")
    key_cols = [
        "token_loss_mean",
        "accuracy",
        "frac_high_conf_wrong",
        "frac_high_conf",
        "opt0_grad_norm",
        "opt1_grad_norm",
        "loss_delta",
        "loss_deceleration",
        "logit_mean_abs",
        "logit_mean",
        "topk_prob_mass_mean_est",
        "zero_sum_residual",
        "centered_norm_ratio",
        "entropy_gap_vs_uniform",
        "loss_delta_sign_flip",
        "destructive_grad",
        "destructive_loss",
        "di_update_update_all",
    ]
    for phase in df["phase"].unique():
        phase_df = df[df["phase"] == phase].sort_values("step")
        sub = phase_df if window is None else phase_df.tail(window)
        rows = []
        for col in key_cols:
            if col not in sub.columns:
                continue
            rows.append({"metric": col, "slope_per_step": slope_vs_step(sub, col)})
        if rows:
            tab = pd.DataFrame(rows).sort_values("slope_per_step")
            render_table(f"Slopes ({phase})", tab.to_dict(orient="records"), ["metric", "slope_per_step"], digits=3)


def correlation_report(df: pd.DataFrame):
    print("\n==== Correlation with accuracy and loss ====")
    num_df = df.select_dtypes(include=[np.number])
    targets = [c for c in ["accuracy", "token_loss_mean"] if c in num_df.columns]
    for target in targets:
        corr = num_df.corr()[target].drop(labels=[target]).sort_values(ascending=False)
        top_corr = corr.head(10).round(3)
        rows = [{"metric": idx, "corr": val} for idx, val in top_corr.items()]
        render_table(f"Top correlations vs {target}", rows, ["metric", "corr"], digits=3)


def automated_conclusions(df: pd.DataFrame):
    print("\n==== Automated analysis ====")
    conclusions = []
    for phase in df["phase"].unique():
        sub = df[df["phase"] == phase].sort_values("step")
        if sub.empty:
            continue
        latest = sub.iloc[-1]
        def get(key, default=np.nan):
            return latest.get(key, default)

        di_loss = get("destructive_loss")
        di_grad = get("destructive_grad")
        zres = get("zero_sum_residual")
        loss_delta = get("loss_delta")
        loss_decel = get("loss_deceleration")
        acc = get("accuracy")

        messages = []
        if not pd.isna(di_loss) and di_loss > 0.3:
            messages.append(f"High destructive_loss ~{di_loss:.3f} (loss improvements cancelling).")
        if not pd.isna(di_grad) and di_grad > 0.3:
            messages.append(f"Destructive_grad ~{di_grad:.3f} (gradient opposition).")
        if not pd.isna(zres) and zres > 0.05:
            messages.append(f"Zero-sum residual {zres:.3f} (logits not centered).")
        if not pd.isna(loss_delta) and not pd.isna(loss_decel):
            if loss_decel > 0:
                messages.append(f"Loss deceleration positive ({loss_decel:.3g}) -> progress slowing.")
        if not pd.isna(acc):
            messages.append(f"Accuracy now {acc:.3f}.")

        if messages:
            conclusions.append(f"[{phase}] " + " ".join(messages))

    if conclusions:
        for line in conclusions:
            print("- " + line)
    else:
        print("No conclusions (insufficient data).")


def caveats():
    print("\n==== Caveats & interpretation hints ====")
    notes = [
        "- DI metrics use sampled logits gradients; values approximate D = 1 - |sum g|/sum|g|, not full-batch exact.",
        "- Loss landscape plots are first-order along the update direction (no re-evaluation of loss at perturbed weights); they show sign/magnitude trends, not exact loss.",
        "- Layer/param-type DI uses current gradients at logging steps; if grads are sparse/zero (e.g., not touched this step), DI may be missing for that layer/type.",
        "- Top-k-based token stats approximate full-softmax behaviour; entropy/zero-sum measures are on the sampled logits subset.",
        "- Interpret high DI (near 1) as strong cancellation/opposition; rising DI concurrent with positive dd_loss suggests ZSL-driven deceleration.",
        "- Green vs red in landscapes: green improves, red worsens; if red dominates at alpha≈1, the update hurts many tokens.",
    ]
    for n in notes:
        print("-", n)


def highlight_snapshots(df: pd.DataFrame, steps: list[int]):
    snaps = nearest_rows_to_steps(df, steps)
    if snaps.empty:
        return
    cols = [
        "phase",
        "target_step",
        "step",
        "destructive_loss",
        "destructive_grad",
        "zero_sum_residual",
        "entropy_gap_vs_uniform",
        "token_loss_mean",
        "accuracy",
    ]
    cols = [c for c in cols if c in snaps.columns]
    rows = snaps[cols].to_dict(orient="records")
    render_table("Highlight snapshots (nearest steps)", rows, cols, digits=4)


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_mistakes.py logs/<run_id>_train_mistakes.jsonl")
        sys.exit(1)

    path = Path(sys.argv[1])
    df = coerce_numeric(load_logs(str(path)))
    highlight_steps = HIGHLIGHT_STEPS

    basic_summary(df)
    highlight_snapshots(df, highlight_steps)
    trend_report(df, window=None)  # full trajectory
    trend_report(df, window=200)   # recent behavior
    correlation_report(df)
    automated_conclusions(df)
    caveats()
    print_descriptions()

    out_dir = path.parent / f"{path.stem}_plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    plot_time_series(df, out_dir)
    plot_di_zsl(df, out_dir, highlight_steps)
    plot_di_groups(df, out_dir)
    plot_loss_landscapes(df, out_dir, max_snapshots=4)
    calibration_scatter(df, out_dir)


if __name__ == "__main__":
    main()
