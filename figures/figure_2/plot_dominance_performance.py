"""Reproduce Figure 2 from the released dominance-performance records."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ARXIV = "#3B6FB6"
REDDIT = "#C94C4C"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv", type=Path, default=Path(__file__).with_name("input") / "dominance_performance.csv")
    parser.add_argument("--cross-dataset", type=Path, default=Path(__file__).with_name("input") / "cross_dataset_dominance.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reproduced") / "figure_2.pdf")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arxiv = read_csv(args.arxiv)
    cross = read_csv(args.cross_dataset)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    x = [float(row["dominance_coordinate"]) for row in arxiv]
    y = [float(row["validation_auc_mean"]) for row in arxiv]
    yerr = [float(row["validation_auc_sample_sd"]) if row["validation_auc_sample_sd"] else 0.0 for row in arxiv]
    left.errorbar(x, y, yerr=yerr, color=ARXIV, marker="o", linewidth=1.4, capsize=2.2)
    anchor_rows = [row for row in arxiv if float(row["coefficient"]) in {0.3, 1.0, 3.0}]
    anchor = sum(float(row["validation_auc_mean"]) for row in anchor_rows) / len(anchor_rows)
    left.axhline(anchor, color="#777777", linestyle="--", linewidth=0.9)
    left.text(0.75, anchor + 0.00018, "dataset-specific anchor", fontsize=7.5, color="#555555")
    offsets = {0.3: (3, 6), 1.0: (-3, 8), 3.0: (0, -12), 6.0: (-10, -11), 10.0: (-12, 5)}
    for row in arxiv:
        coefficient = float(row["coefficient"])
        if coefficient not in offsets:
            continue
        dx, dy = offsets[coefficient]
        left.annotate(rf"$\lambda={coefficient:g}$", (float(row["dominance_coordinate"]), float(row["validation_auc_mean"])), xytext=(dx, dy), textcoords="offset points", fontsize=7.3)
    left.set_xscale("log")
    left.set_xlabel(r"Reference-scaled dominance coordinate $\widetilde{R}$")
    left.set_ylabel("Normalized validation-accuracy trajectory AUC")
    left.set_title("A  Arxiv validation utility across dominance levels", loc="left", fontsize=10, fontweight="bold")
    left.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    label_offsets = {
        ("reddit", "10"): (8, -8),
        ("reddit", "20"): (6, -9),
        ("ogbn-arxiv", "6"): (6, 7),
        ("ogbn-arxiv", "10"): (-4, -9),
    }
    for row in cross:
        dataset = row["dataset"]
        coefficient = row["coefficient"]
        color = ARXIV if dataset == "ogbn-arxiv" else REDDIT
        marker = "o" if dataset == "ogbn-arxiv" else "s"
        direct = row["coordinate_provenance"] == "direct"
        value = float(row["dominance_coordinate"])
        lower = float(row["coordinate_lower"])
        upper = float(row["coordinate_upper"])
        loss = float(row["performance_loss_percentage_points"])
        right.errorbar(value, loss, xerr=[[value - lower], [upper - value]], fmt="none", color=color, capsize=2.0, linewidth=0.9)
        right.scatter(value, loss, marker=marker, s=42, facecolors=color if direct else "white", edgecolors=color, linewidths=1.2, zorder=3)
        dx, dy = label_offsets[(dataset, coefficient)]
        display = "ogbn-arxiv" if dataset == "ogbn-arxiv" else "Reddit"
        right.annotate(
            rf"{display} $\lambda={coefficient}$",
            (value, loss),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="bottom" if dy >= 0 else "top",
            fontsize=7.2,
        )
    right.set_xscale("log")
    right.set_xlim(10.8, 45.5)
    right.set_xlabel("Dominance coordinate")
    right.set_ylabel("Performance loss from dataset-specific anchor\n(percentage points)")
    right.set_title("B  Degradation across dominance levels\nin two datasets", loc="left", fontsize=9, fontweight="bold")
    right.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    dataset_legend = right.legend(handles=[
        Line2D([0], [0], marker="o", color=ARXIV, linewidth=0, markerfacecolor="white", label="ogbn-arxiv"),
        Line2D([0], [0], marker="s", color=REDDIT, linewidth=0, markerfacecolor="white", label="Reddit"),
    ], title="Dataset", loc="upper left", frameon=False, fontsize=7)
    right.add_artist(dataset_legend)
    right.legend(handles=[
        Line2D([0], [0], marker="o", color="#333333", linewidth=0, markerfacecolor="#333333", label=r"Direct $\widehat{R}$"),
        Line2D([0], [0], marker="o", color="#333333", linewidth=0, markerfacecolor="white", label="Reference-scaled coordinate"),
    ], title="Horizontal-coordinate source", loc="lower right", frameon=False, fontsize=7)
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
