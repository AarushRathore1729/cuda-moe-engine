"""Regenerate the README figure from recorded raw samples, without a GPU."""

import json
from pathlib import Path
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
data = json.loads((root / "benchmarks/a100/results.json").read_text())
by_key = {(row["case_id"], row["implementation"], row["stage"]): row for row in data["benchmarks"]}
ratios, savings, labels = [], [], []
for case in data["configurations"]:
    case_id = case["case_id"]

    def latency(implementation, stage):
        return statistics.median(by_key[case_id, implementation, stage]["samples_us"])

    ratios.append(
        latency("PyTorch reference", "dispatch+combine")
        / latency("CUDA vectorized", "dispatch+combine")
    )
    savings.append(
        100 * (1 - latency("CUDA vectorized", "dispatch") / latency("CUDA scalar", "dispatch"))
    )
    labels.append(
        f"T={case['tokens']:,} · H={case['hidden']} · E={case['experts']} · "
        f"{'balanced' if case['pattern'] == 'balanced' else 'skewed'}"
    )

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.8), gridspec_kw={"width_ratios": [1.25, 1]})
fig.patch.set_facecolor("#f3f5fa")
colors = ["#3566c7" if c["pattern"] == "balanced" else "#9571c9" for c in data["configurations"]]
for ax, values, title, xlabel in zip(
    axes,
    (ratios, savings),
    ("Transport speed ratio", "Vector packing: lower dispatch latency"),
    ("PyTorch median / vectorized CUDA median (×)", "Reduction versus scalar CUDA (%)"),
):
    ax.barh(range(len(labels)), values, color=colors, height=0.65)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", weight="bold", pad=15)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#dce3ef", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.22)
    for y, value in enumerate(values):
        ax.text(
            value + max(values) * 0.025,
            y,
            f"{value:.2f}" if ax is axes[0] else f"{value:.1f}%",
            va="center",
            fontsize=9,
        )
axes[0].set_yticks(range(len(labels)), labels)
axes[1].set_yticks(range(len(labels)), [""] * len(labels))
fig.suptitle(
    "CUDA MoE token transport · measured on A100-SXM4-40GB",
    x=0.05,
    ha="left",
    fontsize=18,
    weight="bold",
)
fig.text(
    0.05,
    0.915,
    f"{statistics.median(ratios):.2f}× median speed ratio across 12 configurations · "
    "72.7% lower persistent transport workspace",
    fontsize=11,
)
fig.text(
    0.05,
    0.025,
    "Archived FP16 transport runner · top-2 · 10 warmups / 40 samples per stage · "
    "router and expert computation excluded",
    fontsize=9,
    color="#526179",
)
fig.subplots_adjust(left=0.295, right=0.96, top=0.85, bottom=0.12, wspace=0.25)
(root / "assets").mkdir(exist_ok=True)
fig.savefig(root / "assets/a100_transport.png", dpi=170, facecolor=fig.get_facecolor())
print(root / "assets/a100_transport.png")
