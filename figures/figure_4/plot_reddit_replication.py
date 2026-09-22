"""Reproduce Figure 4 from the released Reddit validation records."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input") / "replication.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reproduced") / "figure_4.pdf")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    panel_a = sorted((row for row in rows if row["panel"] == "A"), key=lambda row: float(row["coefficient"]))
    seed_rows = [row for row in rows if row["panel"] == "B" and row["evidence_scope"] == "matched_seed"]
    means = sorted((row for row in rows if row["panel"] == "B" and row["evidence_scope"] == "three_seed_mean"), key=lambda row: float(row["coefficient"]))
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.7))
    left.plot([float(row["coefficient"]) for row in panel_a], [float(row["validation_auc"]) for row in panel_a], marker="o", color="#C94C4C")
    left.set_xscale("log")
    left.set_xlabel(r"Latent-loss coefficient $\lambda$")
    left.set_ylabel("Validation-trajectory AUC")
    left.set_title("A  Reddit exploratory response (seed 1)", loc="left", fontsize=10, fontweight="bold")
    x, y, errors = [], [], []
    for row in means:
        coefficient = float(row["coefficient"])
        values = [float(item["validation_auc"]) for item in seed_rows if float(item["coefficient"]) == coefficient]
        x.append(coefficient)
        y.append(float(row["validation_auc"]))
        errors.append(statistics.stdev(values))
    right.errorbar(x, y, yerr=errors, marker="o", color="#C94C4C", capsize=2.5)
    right.set_xscale("log")
    right.set_xlabel(r"Latent-loss coefficient $\lambda$")
    right.set_ylabel("Validation-trajectory AUC")
    right.set_title("B  Reddit matched three-seed replication", loc="left", fontsize=10, fontweight="bold")
    for axis in (left, right):
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
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
