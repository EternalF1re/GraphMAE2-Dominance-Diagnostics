"""Deterministic checkpoint-and-degree-stratified bootstrap calculation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stratified_interval(
    rows: list[dict[str, str]],
    dataset: str,
    resamples: int = 10_000,
) -> dict[str, float | int]:
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = (row["checkpoint_role"], int(row["degree_stratum"]))
        groups.setdefault(key, []).append(float(row["dominance_ratio"]))
    if len(groups) != 8 or any(len(values) != 4 for values in groups.values()):
        raise ValueError("expected two checkpoints by four strata with four records per cell")
    seed = int(hashlib.sha256(f"dominance-reference:{dataset}".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [rng.choice(values) for values in groups.values() for _ in range(len(values))]
        estimates.append(float(statistics.median(sample)))
    point = float(statistics.median(float(row["dominance_ratio"]) for row in rows))
    return {
        "point_estimate": point,
        "bootstrap_lower": percentile(estimates, 0.025),
        "bootstrap_upper": percentile(estimates, 0.975),
        "bootstrap_resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = stratified_interval(rows, args.dataset, args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
