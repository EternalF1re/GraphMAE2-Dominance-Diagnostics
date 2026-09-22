"""Reproduce Figure 2 from the released gamma/remasking response record."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input") / "protocol_parameter_response.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reproduced") / "figure_2.pdf")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.7, 3.7))
    a = sorted((row for row in rows if row["panel"] == "A"), key=lambda row: float(row["gamma"]))
    b = sorted((row for row in rows if row["panel"] == "B"), key=lambda row: int(row["remasking_passes"]))
    left.plot([float(row["gamma"]) for row in a], [float(row["dominance_ratio"]) for row in a], marker="o", color="#3B6FB6")
    right.plot([int(row["remasking_passes"]) for row in b], [float(row["dominance_ratio"]) for row in b], marker="o", color="#3B6FB6")
    unstable = next(row for row in b if row["stability"] == "unstable")
    right.scatter(int(unstable["remasking_passes"]), float(unstable["dominance_ratio"]), marker="D", color="#C94C4C", zorder=4)
    right.annotate("73.78 / unstable", (int(unstable["remasking_passes"]), float(unstable["dominance_ratio"])), xytext=(7, -13), textcoords="offset points", fontsize=7.5)
    left.set_xlabel(r"Loss-shape parameter $\gamma$")
    left.set_ylabel(r"Dominance ratio $\widehat{R}$")
    right.set_xlabel("Remasking passes K")
    right.set_ylabel(r"Dominance ratio $\widehat{R}$")
    left.set_title(r"A  Varying $\gamma$ at K=3", loc="left", fontsize=10, fontweight="bold")
    right.set_title(r"B  Varying K at $\gamma=6$", loc="left", fontsize=10, fontweight="bold")
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
