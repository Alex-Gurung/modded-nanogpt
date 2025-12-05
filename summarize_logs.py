#!/usr/bin/env python3
"""
Scan a folder of training logs, grab the final validation line from each file,
and report per-run values plus mean/std across runs.

Expected log line format (produced by train_gpt.py and friends):
    step:<cur>/<total> val_loss:<float> train_time:<int>ms ...

By default the script looks for common text-log suffixes (*.txt, *.log, *.out,
*.err, *.stdout, *.stderr). Pass --patterns '**/*' to scan every file.
"""

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


VAL_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)\s+train_time:(\d+)ms")
DEFAULT_PATTERNS = ("*.txt", "*.log", "*.out", "*.err", "*.stdout", "*.stderr")


@dataclass
class RunResult:
    path: Path
    step: int
    total_steps: int
    val_loss: float
    train_time_ms: int


def iter_files(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    seen = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                seen.add(path)
    yield from sorted(seen)


def parse_file(path: Path) -> RunResult | None:
    last = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = VAL_RE.search(line)
            if match:
                step, total, loss, t_ms = match.groups()
                last = RunResult(
                    path=path,
                    step=int(step),
                    total_steps=int(total),
                    val_loss=float(loss),
                    train_time_ms=int(t_ms),
                )
    return last


def human_time(ms: int) -> str:
    seconds = ms / 1000.0
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {sec:04.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {sec:04.1f}s"
    return f"{sec:.2f}s"


def mean_std(values: List[float]) -> tuple[float, float | None]:
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else None
    return mean_val, std_val


def main():
    parser = argparse.ArgumentParser(
        description="Summarize final validation loss and train_time across log files."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory containing log files (e.g., logs/).",
    )
    parser.add_argument(
        "--patterns",
        "-p",
        nargs="*",
        default=list(DEFAULT_PATTERNS),
        help="Glob patterns (relative to log_dir) to pick log files; use '**/*' to scan everything.",
    )
    args = parser.parse_args()

    root = args.log_dir
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    results = []
    candidate_files = list(iter_files(root, args.patterns))
    if not candidate_files:
        raise SystemExit("No files matched the requested patterns. Try --patterns '**/*'.")

    for path in candidate_files:
        parsed = parse_file(path)
        if parsed:
            results.append(parsed)

    if not results:
        raise SystemExit("No matching validation lines found.")

    name_width = max(len(r.path.name) for r in results)
    step_width = max(len(f"{r.step}/{r.total_steps}") for r in results)

    header = (
        f"{'file':<{name_width}}  "
        f"{'step':>{step_width}}  "
        f"{'val_loss':>10}  "
        f"{'time (ms)':>12}  "
        f"{'time (pretty)':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        step_str = f"{r.step}/{r.total_steps}"
        print(
            f"{r.path.name:<{name_width}}  "
            f"{step_str:>{step_width}}  "
            f"{r.val_loss:>10.4f}  "
            f"{r.train_time_ms:>12,d}  "
            f"{human_time(r.train_time_ms):>13}"
        )

    losses = [r.val_loss for r in results]
    times = [r.train_time_ms for r in results]
    loss_mean, loss_std = mean_std(losses)
    time_mean, time_std = mean_std(times)

    print("\nSummary:")
    loss_std_str = f"{loss_std:.4f}" if loss_std is not None else "n/a"
    time_std_str = f"{time_std:,.0f} ms" if time_std is not None else "n/a"
    print(
        f"- val_loss mean={loss_mean:.4f}, std={loss_std_str}"
    )
    print(
        f"- train_time mean={time_mean:,.0f} ms ({human_time(int(time_mean))}), std={time_std_str}"
    )


if __name__ == "__main__":
    main()
