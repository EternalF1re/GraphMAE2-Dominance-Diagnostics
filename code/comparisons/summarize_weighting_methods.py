"""Summarize the released per-seed weighting-method records."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def optional_mean(rows: list[dict[str, str]], field: str) -> str:
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return f"{statistics.mean(values):.12g}" if values else "NOT_IDENTIFIABLE_FROM_RELEASED_RECORDS"


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    output = []
    for method, group in groups.items():
        auc = [float(row["validation_auc"]) for row in group]
        output.append(
            {
                "method": method,
                "weighting_strategy": group[0]["weighting_strategy"],
                "overhead_mode": group[0]["overhead_mode"],
                "seed_count": str(len(group)),
                "validation_auc_mean": f"{statistics.mean(auc):.12g}",
                "validation_auc_sample_sd": f"{statistics.stdev(auc):.12g}",
                "training_wall_seconds_mean": optional_mean(group, "training_wall_seconds"),
                "total_gpu_time_seconds_mean": optional_mean(group, "total_gpu_time_seconds"),
                "peak_cuda_allocated_bytes_mean": optional_mean(group, "peak_cuda_allocated_bytes"),
                "additional_component_gradients": group[0]["additional_component_gradients"],
            }
        )
    return sorted(output, key=lambda row: float(row["validation_auc_mean"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize(rows)
    fields = list(summary[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
