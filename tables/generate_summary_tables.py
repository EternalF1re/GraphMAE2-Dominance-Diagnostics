"""Generate compact paper-facing tables from the released processed records."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def method_table(root: Path) -> list[dict[str, object]]:
    rows = read_csv(root / "processed_records/method_comparisons/weighting_methods.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    output = []
    for method, group in grouped.items():
        values = [float(row["validation_auc"]) for row in group]
        output.append({
            "method": method,
            "weighting_strategy": group[0]["weighting_strategy"],
            "validation_auc_mean": statistics.mean(values),
            "validation_auc_sample_sd": statistics.stdev(values),
            "additional_full_training_trajectories": 0 if method == "Ours lambda=0.3" else len(group),
            "extra_gradient_computation": group[0]["additional_component_gradients"],
            "overhead_mode": group[0]["overhead_mode"],
        })
    return sorted(output, key=lambda row: float(row["validation_auc_mean"]), reverse=True)


def sampling_table(root: Path) -> list[dict[str, object]]:
    payload = json.loads((root / "processed_records/sampling_design.json").read_text(encoding="utf-8"))
    output = []
    for dataset, values in payload["datasets"].items():
        output.append({
            "dataset": dataset,
            "checkpoint_updates": ";".join(str(value) for value in values["checkpoint_updates"]),
            "batches_per_checkpoint": values["batches_per_checkpoint"],
            "degree_strata": 4,
            "batches_per_stratum": values["batches_per_stratum"],
            "target_roots_per_batch": values["target_roots_per_batch"],
            "selection": payload["later_procedure"]["selection_within_stratum"],
            "bootstrap_resamples": 10000,
        })
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("generated"))
    args = parser.parse_args()
    write_csv(args.output_dir / "weighting_method_summary.csv", method_table(root))
    write_csv(args.output_dir / "sampling_design_summary.csv", sampling_table(root))


if __name__ == "__main__":
    main()
