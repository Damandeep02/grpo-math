"""Plot the GRPO training curves from a run's log.jsonl.
    python plot.py runs/grpo
Saves runs/grpo/curves.png."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no screen on HPRC
import matplotlib.pyplot as plt


def smooth(values, window=10):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


run = sys.argv[1]
rows = [json.loads(line) for line in open(os.path.join(run, "log.jsonl"))]
steps = [r["step"] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, key, title in [
    (axes[0], "accuracy", "Training accuracy (fraction of sampled answers correct)"),
    (axes[1], "completion_tokens", "Average answer length (tokens)"),
    (axes[2], "kl", "KL from the original model"),
]:
    values = [r[key] for r in rows]
    ax.plot(steps, values, alpha=0.25, color="#2a6fdb")
    ax.plot(steps, smooth(values), color="#2a6fdb")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

plt.tight_layout()
path = os.path.join(run, "curves.png")
plt.savefig(path, dpi=150)
print("Saved", path)
