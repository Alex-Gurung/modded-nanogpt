import re
import glob

import matplotlib.pyplot as plt

# pattern to match validation lines
VAL_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)\s+train_time:(\d+)ms"
)

def parse_log(path, label):
    steps = []
    val_losses = []
    times_s = []

    with open(path, "r") as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                step = int(m.group(1))
                total_steps = int(m.group(2))  # often constant, but we don't actually need it
                val_loss = float(m.group(3))
                train_time_ms = int(m.group(4))
                steps.append(step)
                val_losses.append(val_loss)
                times_s.append(train_time_ms / 1000.0)

    print(val_losses)
    return {
        "label": label,
        "steps": steps,
        "val_losses": val_losses,
        "times_s": times_s,
    }

def main():
    # Map log files to human-readable labels
    runs = [
        ("logs/e71fe73a-44ad-4f5a-a5e4-9127fb4ec9fa.txt", "Muon (baseline)"),
        ("logs/a6f8b210-2cd2-47ed-b8f1-ea34a318d1f9.txt", "Muon (baseline)"),
        # ("logs/factor_muon.txt", "Factorized NorMuon"),
        # add more here...
    ]

    plt.figure()

    for path, label in runs:
        data = parse_log(path, label)
        # Choose x-axis: steps or times_s
        x = data["steps"]          # or data["times_s"]
        y = data["val_losses"]
        plt.plot(x, y, label=label)

    plt.xlabel("Step")  # or "Train time (s)" if you used times_s
    plt.ylabel("Validation loss")
    plt.title("Validation loss over training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

