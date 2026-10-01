"""Assemble the current nine-method record from released frozen rows."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD_TO_BENCHMARK = {
    "Original lambda=10": "ORIGINAL_STATIC_LAMBDA_10",
    "Ours lambda=0.3": "LABEL_FREE_CALIBRATED_STATIC_LAMBDA_0P3",
    "Equal lambda=1": "EQUAL_STATIC_LAMBDA_1",
    "GradNorm": "GRADNORM_ADAPTED",
    "Uncertainty Weighting": "UNCERTAINTY_WEIGHTING_ADAPTED",
    "CoV-Weighting": "COV_WEIGHTING_ADAPTED",
    "PCGrad@lambda=10": "PCGRAD_ADAPTED",
    "Aligned-MTL": "ALIGNED_MTL",
    "FAMO": "FAMO",
}
ADDITIONAL = {
    "Aligned-MTL": ("aligned_mtl", "dynamic gradient alignment"),
    "FAMO": ("famo", "dynamic loss weighting"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assemble(root: Path = ROOT) -> list[dict[str, object]]:
    historical = read_rows(root / "processed_records/method_comparisons/weighting_methods.csv")
    overhead = {row["method"]: row for row in read_rows(root / "results/sota_extension/overhead/OVERHEAD_SUMMARY.csv")}
    seeds: dict[str, dict[int, float]] = {}
    strategies: dict[str, str] = {}
    provenance: dict[str, str] = {}
    for row in historical:
        method = row["method"]
        seeds.setdefault(method, {})[int(row["seed"])] = float(row["validation_auc"])
        strategies[method] = row["weighting_strategy"]
        provenance[method] = "historical released per-seed method record"
    for method, (folder, strategy) in ADDITIONAL.items():
        rows = read_rows(root / f"results/sota_extension/{folder}/per_seed_auc.csv")
        seeds[method] = {int(row["pretraining_seed"]): float(row["normalized_validation_trajectory_auc"]) for row in rows}
        strategies[method] = strategy
        provenance[method] = "frozen matched-seed SOTA adjudication; released per-seed AUC"
    if set(seeds) != set(METHOD_TO_BENCHMARK):
        raise ValueError(f"unexpected completed-method set: {sorted(seeds)}")
    output = []
    for method, per_seed in seeds.items():
        if set(per_seed) != {0, 1, 2}:
            raise ValueError(f"missing AUC seed for {method}")
        benchmark = overhead[METHOD_TO_BENCHMARK[method]]
        if benchmark["status"] != "COMPLETE":
            raise ValueError(f"incomplete controlled benchmark: {method}")
        values = [per_seed[seed] for seed in (0, 1, 2)]
        output.append({
            "method": method,
            "weighting_strategy": strategies[method],
            "seed0_auc": values[0],
            "seed1_auc": values[1],
            "seed2_auc": values[2],
            "validation_auc_mean": statistics.mean(values),
            "validation_auc_sample_sd": statistics.stdev(values),
            "controlled_mean_ms_per_step": benchmark["mean_ms_per_step"],
            "peak_allocated_gpu_gib": benchmark["peak_allocated_gib"],
            "peak_reserved_gpu_gib": benchmark["peak_reserved_gib"],
            "result_provenance": provenance[method],
        })
    return sorted(output, key=lambda row: float(row["validation_auc_mean"]), reverse=True)


def main() -> None:
    rows = assemble()
    path = ROOT / "processed_records/method_comparisons/weighting_methods_current.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    overhead_rows = read_rows(ROOT / "results/sota_extension/overhead/OVERHEAD_SUMMARY.csv")
    selected = {value for value in METHOD_TO_BENCHMARK.values()}
    fields = (
        "method", "mean_ms_per_step", "median_ms_per_step", "p95_ms_per_step",
        "peak_allocated_gib", "peak_reserved_gib", "benchmark_status",
    )
    completed = [{**{key: row[key] for key in fields[:-1]}, "benchmark_status": row["status"]}
                 for row in overhead_rows if row["method"] in selected]
    if len(completed) != 9:
        raise ValueError("completed overhead comparison requires nine rows")
    overhead_path = ROOT / "results/sota_extension/overhead/OVERHEAD_COMPLETED_METHODS.csv"
    with overhead_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(completed)
    print(f"current completed methods: {len(rows)}")


if __name__ == "__main__":
    main()
