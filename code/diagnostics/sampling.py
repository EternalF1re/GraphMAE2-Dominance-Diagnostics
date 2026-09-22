"""Reader-facing implementation of the reported degree-stratified selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class CandidateBatch:
    schedule_position: int
    root_ids: tuple[int, ...]
    root_in_degrees: tuple[int, ...]

    @property
    def median_root_degree(self) -> float:
        return float(median(self.root_in_degrees))


def _canonical_hash(dataset: str, root_ids: Iterable[int], salt: str) -> str:
    payload = json.dumps(
        {"dataset": dataset, "root_ids": list(root_ids), "selection_salt": salt},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _equal_rank_groups(items: list[CandidateBatch], groups: int) -> list[list[CandidateBatch]]:
    quotient, remainder = divmod(len(items), groups)
    result: list[list[CandidateBatch]] = []
    start = 0
    for index in range(groups):
        width = quotient + (1 if index < remainder else 0)
        result.append(items[start : start + width])
        start += width
    return result


def select_degree_stratified_batches(
    dataset: str,
    candidates: Iterable[CandidateBatch],
    excluded_root_ids: Iterable[int] = (),
    batches_per_stratum: int = 4,
    salt: str = "GRAPHMAE2_DOMINANCE_DIAGNOSTIC",
) -> list[dict[str, object]]:
    """Select four complete schedule batches from each of four rank strata."""

    excluded = set(excluded_root_ids)
    eligible = [batch for batch in candidates if excluded.isdisjoint(batch.root_ids)]
    ordered = sorted(eligible, key=lambda batch: (batch.median_root_degree, batch.schedule_position))
    if len(ordered) < 4 * batches_per_stratum:
        raise ValueError("insufficient eligible batches")
    selected: list[dict[str, object]] = []
    for stratum, group in enumerate(_equal_rank_groups(ordered, 4)):
        ranked = sorted(group, key=lambda batch: _canonical_hash(dataset, batch.root_ids, salt))
        if len(ranked) < batches_per_stratum:
            raise ValueError(f"degree stratum {stratum} has too few batches")
        for batch in ranked[:batches_per_stratum]:
            selected.append(
                {
                    "degree_stratum": stratum,
                    "schedule_position": batch.schedule_position,
                    "root_ids": list(batch.root_ids),
                    "median_root_degree": batch.median_root_degree,
                    "selection_hash": _canonical_hash(dataset, batch.root_ids, salt),
                }
            )
    return sorted(selected, key=lambda row: (int(row["degree_stratum"]), str(row["selection_hash"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON list of candidate batches")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = [
        CandidateBatch(
            schedule_position=int(row["schedule_position"]),
            root_ids=tuple(int(value) for value in row["root_ids"]),
            root_in_degrees=tuple(int(value) for value in row["root_in_degrees"]),
        )
        for row in payload
    ]
    selected = select_degree_stratified_batches(args.dataset, candidates)
    args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
