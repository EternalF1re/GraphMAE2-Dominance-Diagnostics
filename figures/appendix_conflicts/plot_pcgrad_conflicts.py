"""Reproduce the appendix pre-projection gradient-cosine histogram."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input") / "pcgrad_conflict_histogram.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reproduced") / "appendix_conflicts.pdf")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    left = [float(row["bin_left"]) for row in rows]
    width = [float(row["bin_right"]) - float(row["bin_left"]) for row in rows]
    counts = [int(row["update_count"]) for row in rows]
    colors = ["#C94C4C" if row["region"] == "conflict" else "#4C78A8" for row in rows]
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(left, counts, width=width, align="edge", color=colors, edgecolor="white", linewidth=0.25)
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Pre-projection gradient cosine")
    axis.set_ylabel("Optimizer updates")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.legend(handles=[
        Patch(facecolor="#C94C4C", label="Conflict (cosine < 0)"),
        Patch(facecolor="#4C78A8", label="Non-conflict (cosine >= 0)"),
    ], frameon=False, loc="upper left")
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
