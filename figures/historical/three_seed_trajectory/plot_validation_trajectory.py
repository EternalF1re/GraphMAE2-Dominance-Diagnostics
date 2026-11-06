"""Reproduce Figure 1 from the released validation-trajectory record."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input") / "validation_trajectory.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reproduced") / "figure_1.pdf")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    styles = {
        "lambda_10": (r"$\lambda=10$", "#3B6FB6", "o"),
        "lambda_0.3": (r"$\lambda=0.3$", "#C94C4C", "s"),
    }
    fig, axis = plt.subplots(figsize=(6.6, 4.0))
    for method, (label, color, marker) in styles.items():
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["optimizer_update"]))
        axis.errorbar(
            [int(row["optimizer_update"]) for row in selected],
            [float(row["validation_accuracy_mean"]) for row in selected],
            yerr=[float(row["validation_accuracy_sample_sd"]) for row in selected],
            color=color, marker=marker, linewidth=1.6, markersize=4.5, capsize=2.5, label=label,
        )
    final_rows = {row["method"]: row for row in rows if int(row["optimizer_update"]) == 8532}
    lower = float(final_rows["lambda_10"]["validation_accuracy_mean"])
    upper = float(final_rows["lambda_0.3"]["validation_accuracy_mean"])
    axis.hlines([lower, upper], 7200, 8500, colors="#777777", linestyles="--", linewidth=0.8)
    arrow_x = 8200
    axis.annotate("", xy=(arrow_x, upper), xytext=(arrow_x, lower), arrowprops={"arrowstyle": "<->", "color": "#555555", "lw": 0.9})
    axis.text(8050, (lower + upper) / 2, "0.481 pp", ha="right", va="center", fontsize=8)
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Linear-probe top-1 validation accuracy")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )


if __name__ == "__main__":
    main()
