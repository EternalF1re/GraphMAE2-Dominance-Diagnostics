"""Generate the current content/provenance manifest during local assembly."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RELEASE_CONTENT_MANIFEST.csv"
EXCLUDED = {"MANIFEST.sha256", OUTPUT.name}


def provenance(relative: str) -> tuple[str, str]:
    for directory, source in [
        ("factorial", "source-verified80 matched trajectories,560 checkpoint records"),
        ("gradient_geometry", "source-verified120 representation-gradient measurement records"),
        ("maskgae_pubmed", "source-verified50 completed runs,350 checkpoints,actual clipping updates"),
        ("estimator_reliability", "conditionally accepted scalar evidence; original execution gate failed")]:
        if relative.startswith("processed_records/"+directory+"/"):
            return "processed record", source
    if relative.startswith("code/analysis/"):
        return "analysis code", "CPU-only scalar reproduction; no model or asset dependency"
    if relative.startswith(("code/measurements/", "code/experiments/")):
        return "scientific implementation", "source-bound scientific formula or inspection excerpt; scope documented"
    if relative.startswith("tables/revision/"):
        return "derived table", "deterministic CPU reproduction from released scalar inputs"
    if relative.startswith("figures/"):
        return "figure", "previous public release; figure numbers aligned to manuscript"
    if relative.startswith("code/sota_extension/"):
        return "integration code", "frozen formal adapter source; private runner binding removed"
    if relative.startswith("code/"):
        return "source code", "previous public release"
    if relative.startswith("experiments/sota_extension/"):
        return "protocol", "frozen preregistration or post-result adjudication; public redaction only"
    if relative.startswith("results/sota_extension/"):
        return "processed result", "frozen SOTA closeout evidence or previous public release"
    if relative.startswith("processed_records/"):
        return "processed record", "previous public release; current comparison assembled from frozen records"
    if relative.startswith("tables/"):
        return "table", "deterministic generation from released processed records"
    return "release material", "preserved public baseline or current reader-facing documentation/verification"


def entries(root: Path = ROOT) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or {".git", "__pycache__", "reproduced"}.intersection(path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED:
            continue
        category, source = provenance(relative)
        rows.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "category": category,
            "source_provenance": source,
        })
    return rows


def main() -> None:
    rows = entries()
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"sync manifest entries: {len(rows)}")


if __name__ == "__main__":
    main()
