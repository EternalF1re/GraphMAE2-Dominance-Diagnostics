"""Static, privacy, manifest, and processed-record checks for this release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


TEXT_SUFFIXES = {".py", ".md", ".csv", ".json", ".yml", ".yaml", ".txt", ".sha256"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".npy", ".npz", ".log", ".pyc"}
GENERATED_OUTPUT_DIRS = {"reproduced"}
REQUIRED_DIRECTORIES = {
    "code", "environment", "figures", "paper_records", "processed_records", "protocols", "tables", "tests"
}


def is_generated_output(path: Path, root: Path) -> bool:
    return bool(GENERATED_OUTPUT_DIRS.intersection(path.relative_to(root).parts))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sensitive_patterns() -> list[tuple[str, re.Pattern[str]]]:
    return [
        ("linux home path", re.compile("/ho" + "me/", re.I)),
        ("mounted drive path", re.compile("/m" + "nt/", re.I)),
        ("Windows absolute path", re.compile(r"\b[A-Za-z]:[\\/]")),
        ("private username", re.compile("su" + "jingze", re.I)),
        ("internal numbered phase", re.compile("ph" + r"ase\s*\d", re.I)),
        ("internal numbered gate", re.compile("ga" + r"te\s*\d", re.I)),
        ("versioned internal estimator", re.compile("esti" + r"mator\s*v\d", re.I)),
        (
            "run timestamp",
            re.compile(
                r"(?<![\d.])(?:19|20)"
                + r"\d{2}(?:[-_](?:0[1-9]|1[0-2])[-_](?:0[1-9]|[12]\d|3[01])"
                + r"|(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))"
                + r"(?:[-_T]\d{4,6})?(?![\d.])"
            ),
        ),
        ("API secret", re.compile(r"(?:api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]+", re.I)),
        ("secret key prefix", re.compile(r"\bsk-[A-Za-z0-9]{16,}")),
    ]


def scan_text_and_paths(root: Path) -> list[str]:
    findings = []
    patterns = sensitive_patterns()
    for path in sorted(root.rglob("*")):
        if is_generated_output(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        for label, pattern in patterns:
            if pattern.search(relative):
                findings.append(f"path:{label}:{relative}")
        if path.is_symlink():
            findings.append(f"symlink:{relative}")
        if (
            not path.is_file()
            or path.suffix.lower() not in TEXT_SUFFIXES
            or path.suffix.lower() == ".sha256"
        ):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for label, pattern in patterns:
            if pattern.search(text):
                findings.append(f"content:{label}:{relative}")
    return findings


def verify_records(root: Path) -> None:
    expected = {
        "ogbn-arxiv": (31.7653148904106, 29.849883433713472, 44.0618155456329),
        "reddit": (12.62592612939547, 11.463302621964752, 14.199962981296819),
    }
    for dataset, relative in {
        "ogbn-arxiv": "processed_records/arxiv/reference_summary.csv",
        "reddit": "processed_records/reddit/reference_summary.csv",
    }.items():
        row = read_csv(root / relative)[0]
        actual = tuple(float(row[field]) for field in ("dominance_ratio", "bootstrap_lower", "bootstrap_upper"))
        assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(actual, expected[dataset]))

    for relative in (
        "processed_records/arxiv/reference_diagnostic_records.csv",
        "processed_records/reddit/reference_diagnostic_records.csv",
    ):
        rows = read_csv(root / relative)
        assert len(rows) == 32
        cells: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (row["checkpoint_role"], row["degree_stratum"])
            cells[key] = cells.get(key, 0) + 1
        assert len(cells) == 8 and set(cells.values()) == {4}

    histogram = read_csv(root / "processed_records/method_comparisons/pcgrad_conflict_histogram.csv")
    assert sum(int(row["update_count"]) for row in histogram) == 25_596
    assert sum(int(row["update_count"]) for row in histogram if row["region"] == "conflict") == 9_417

    sampling = json.loads((root / "processed_records/sampling_design.json").read_text(encoding="utf-8"))
    assert sampling["datasets"]["ogbn-arxiv"]["target_roots_per_batch"] == 128
    assert sampling["datasets"]["reddit"]["target_roots_per_batch"] == 64

    boundaries = {row["dataset"]: row for row in read_csv(root / "processed_records/small_graphs/boundary_summary.csv")}
    assert math.isclose(float(boundaries["cora"]["reference_weighted_ratio"]), 6.575011615350309e-10, rel_tol=0.0, abs_tol=1e-24)
    assert math.isclose(float(boundaries["citeseer"]["checkpoint_median_ratio"]), 29.31016570279685, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(boundaries["wikics"]["later_weighted_ratio"]), 1.1604030792105146e-8, rel_tol=0.0, abs_tol=1e-22)

    figure_inputs = {
        "figures/figure_1/input/validation_trajectory.csv": "processed_records/arxiv/validation_trajectory.csv",
        "figures/figure_2/input/dominance_performance.csv": "processed_records/arxiv/dominance_performance.csv",
        "figures/figure_2/input/cross_dataset_dominance.csv": "processed_records/method_comparisons/cross_dataset_dominance.csv",
        "figures/figure_3/input/protocol_parameter_response.csv": "processed_records/arxiv/protocol_parameter_response.csv",
        "figures/figure_4/input/replication.csv": "processed_records/reddit/replication.csv",
        "figures/appendix_conflicts/input/pcgrad_conflict_histogram.csv": "processed_records/method_comparisons/pcgrad_conflict_histogram.csv",
    }
    for figure_input, canonical_record in figure_inputs.items():
        assert (root / figure_input).read_bytes() == (root / canonical_record).read_bytes()


def expected_manifest(root: Path) -> dict[str, str]:
    entries = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if is_generated_output(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "MANIFEST.sha256" or "__pycache__" in path.parts:
            continue
        entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def verify_manifest(root: Path) -> None:
    path = root / "MANIFEST.sha256"
    assert path.is_file()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    recorded = dict(line.split("  ", 1)[::-1] for line in lines)
    assert list(recorded) == sorted(recorded)
    assert recorded == expected_manifest(root)


def verify_inventory(root: Path) -> None:
    for directory in sorted(REQUIRED_DIRECTORIES):
        assert (root / directory).is_dir(), directory
    required = [
        "README.md", "REPRODUCIBILITY.md", "LICENSE", "LICENSE-AUTHORS", "THIRD_PARTY_NOTICES.md",
        "code/graphmae2/main_large.py", "code/diagnostics/component_gradients.py",
        "code/graphmae2/config_paths.py",
        "code/graphmae2/configs/ogbn-arxiv_upstream_default.yaml",
        "paper_records/ogbn-arxiv_execution_record.json",
        "paper_records/reddit_execution_record.json",
        "paper_records/README.md",
        "figures/figure_1/plot_validation_trajectory.py",
        "figures/figure_2/plot_dominance_performance.py",
        "figures/figure_3/plot_protocol_parameters.py",
        "figures/figure_4/plot_reddit_replication.py",
        "figures/appendix_conflicts/plot_pcgrad_conflicts.py",
        "figures/figure_1/figure_1_reference.pdf",
        "figures/figure_2/figure_2_reference.pdf",
        "figures/figure_3/figure_3_reference.pdf",
        "figures/figure_4/figure_4_reference.pdf",
        "figures/appendix_conflicts/appendix_conflicts_reference.pdf",
        "tables/generate_summary_tables.py", "protocols/measurement_rules.md",
        "tables/generated/weighting_method_summary.csv",
        "tables/generated/sampling_design_summary.csv",
        "tests/generate_manifest.py", "tests/build_release_zip.py", "tests/audit_public_clis.py",
        "tests/test_r1_release.py",
    ]
    for relative in required:
        assert (root / relative).is_file(), relative
    for path in root.rglob("*"):
        if is_generated_output(path, root):
            continue
        if path.is_file():
            if "__pycache__" in path.parts:
                continue
            assert path.suffix.lower() not in FORBIDDEN_SUFFIXES, path
            assert path.stat().st_size <= 5 * 1024 * 1024, path


def _validate_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    assert "\\" not in name, f"ZIP member uses backslash: {name}"
    assert "\x00" not in name, "ZIP member contains NUL"
    assert not name.startswith("/"), f"absolute ZIP member: {name}"
    assert not re.match(r"^[A-Za-z]:", name), f"drive-absolute ZIP member: {name}"
    member = PurePosixPath(name)
    assert not member.is_absolute(), f"absolute ZIP member: {name}"
    assert ".." not in member.parts, f"ZIP traversal member: {name}"
    file_type = (info.external_attr >> 16) & 0o170000
    assert file_type != 0o120000, f"ZIP symlink member: {name}"
    return member


def verify_release(root: Path, skip_manifest: bool = False) -> None:
    verify_inventory(root)
    verify_records(root)
    findings = scan_text_and_paths(root)
    assert not findings, "\n".join(findings)
    if not skip_manifest:
        verify_manifest(root)


def verify_zip(path: Path) -> None:
    patterns = sensitive_patterns()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        assert infos, "ZIP is empty"
        members = [_validate_zip_member(info) for info in infos]
        assert len({member.as_posix() for member in members}) == len(members), "duplicate ZIP member"
        roots = {member.parts[0] for member in members if member.parts}
        assert len(roots) == 1, f"ZIP must contain one release root: {sorted(roots)}"
        for name in (member.as_posix() for member in members):
            for label, pattern in patterns:
                assert not pattern.search(name), f"ZIP member {label}: {name}"
        with tempfile.TemporaryDirectory(prefix="graphmae2-release-verify-") as directory:
            archive.extractall(directory)
            extracted_root = Path(directory) / next(iter(roots))
            assert extracted_root.is_dir(), "ZIP release root was not extracted"
            verify_release(extracted_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    verify_release(root, skip_manifest=args.skip_manifest)
    if args.zip:
        verify_zip(args.zip)
    print("release verification: PASS")


if __name__ == "__main__":
    main()
