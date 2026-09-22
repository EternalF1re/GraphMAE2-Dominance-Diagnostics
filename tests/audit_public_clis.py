"""Run --help for every reader-facing command without data or model access."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CLIS = (
    "code/graphmae2/main_large.py",
    "code/graphmae2/datasets/localclustering.py",
    "code/diagnostics/bootstrap_reference_interval.py",
    "code/diagnostics/sampling.py",
    "code/comparisons/summarize_weighting_methods.py",
    "figures/figure_1/plot_validation_trajectory.py",
    "figures/figure_2/plot_dominance_performance.py",
    "figures/figure_3/plot_protocol_parameters.py",
    "figures/figure_4/plot_reddit_replication.py",
    "figures/appendix_conflicts/plot_pcgrad_conflicts.py",
    "tables/generate_summary_tables.py",
    "tests/generate_manifest.py",
    "tests/build_release_zip.py",
    "tests/verify_release.py",
)


def main() -> None:
    failures = []
    for relative in PUBLIC_CLIS:
        command = [sys.executable, str(ROOT / relative), "--help"]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        print(f"=== {relative} ===")
        print(f"command: {' '.join(command)}")
        print(f"exit_code: {completed.returncode}")
        if completed.stdout:
            print("--- stdout ---")
            print(completed.stdout.rstrip())
        if completed.stderr:
            print("--- stderr ---")
            print(completed.stderr.rstrip())
        if completed.returncode != 0:
            failures.append(relative)
    if failures:
        raise SystemExit(f"public CLI audit failed: {', '.join(failures)}")
    print(f"public CLI audit: PASS ({len(PUBLIC_CLIS)} commands)")


if __name__ == "__main__":
    main()
