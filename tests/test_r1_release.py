"""Regression tests for portable release packaging and rerun isolation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = load_module("release_verifier", ROOT / "tests" / "verify_release.py")


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


class ZipSafetyTests(unittest.TestCase):
    def assert_zip_rejected(self, members: dict[str, bytes]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.zip"
            write_zip(path, members)
            with self.assertRaises((AssertionError, ValueError, RuntimeError)):
                VERIFY.verify_zip(path)

    def test_backslash_member_is_rejected(self) -> None:
        self.assert_zip_rejected({"release\\README.md": b"unsafe"})

    def test_parent_traversal_member_is_rejected(self) -> None:
        self.assert_zip_rejected({"release/../escape.txt": b"unsafe"})

    def test_absolute_member_is_rejected(self) -> None:
        self.assert_zip_rejected({"C" + ":/absolute.txt": b"unsafe"})

    def test_incomplete_extracted_tree_is_rejected(self) -> None:
        self.assert_zip_rejected({"release/README.md": b"incomplete"})


class ManifestTests(unittest.TestCase):
    def test_official_manifest_verifies(self) -> None:
        VERIFY.verify_manifest(ROOT)

    def test_one_byte_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / ROOT.name
            shutil.copytree(ROOT, copied)
            readme = copied / "README.md"
            readme.write_bytes(readme.read_bytes() + b"X")
            with self.assertRaises(AssertionError):
                VERIFY.verify_manifest(copied)

    def test_manifest_digest_text_is_not_scanned_as_natural_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "MANIFEST.sha256").write_text(
                "2026" + "0730deadbeef  README.md\n",
                encoding="utf-8",
            )
            self.assertEqual(VERIFY.scan_text_and_paths(root), [])

    def test_git_metadata_is_ignored_in_a_real_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / ROOT.name
            shutil.copytree(
                ROOT,
                copied,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "reproduced"),
            )
            entries = VERIFY.expected_manifest(copied)
            (copied / "MANIFEST.sha256").write_text(
                "".join(f"{digest}  {relative}\n" for relative, digest in sorted(entries.items())),
                encoding="utf-8",
            )
            metadata = copied / ".git" / "objects" / "sentinel"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("not release content", encoding="utf-8")
            VERIFY.verify_release(copied)


class FigureIsolationTests(unittest.TestCase):
    def test_paper_figure_number_mapping(self) -> None:
        figure_2 = ROOT / "figures" / "figure_2"
        figure_3 = ROOT / "figures" / "figure_3"
        self.assertTrue((figure_2 / "plot_protocol_parameters.py").is_file())
        self.assertTrue((figure_2 / "figure_2_reference.pdf").is_file())
        self.assertTrue((figure_3 / "plot_dominance_performance.py").is_file())
        self.assertTrue((figure_3 / "figure_3_reference.pdf").is_file())

    def test_figure_rerun_does_not_modify_reference_pdf(self) -> None:
        source = ROOT / "figures" / "figure_1"
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "figure_1"
            shutil.copytree(source, copied)
            reference = copied / "figure_1_reference.pdf"
            before = hashlib.sha256(reference.read_bytes()).hexdigest()
            completed = subprocess.run(
                [sys.executable, str(copied / "plot_validation_trajectory.py")],
                cwd=Path(directory),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((copied / "reproduced" / "figure_1.pdf").is_file())
            self.assertEqual(before, hashlib.sha256(reference.read_bytes()).hexdigest())


class ConfigProvenanceTests(unittest.TestCase):
    def config_module(self):
        return load_module(
            "config_paths",
            ROOT / "code" / "graphmae2" / "config_paths.py",
        )

    def test_config_resolution_is_independent_of_cwd(self) -> None:
        module = self.config_module()
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                from_elsewhere = module.resolve_config_path("ogbn-arxiv")
                os.chdir(ROOT / "code" / "graphmae2")
                from_package = module.resolve_config_path("ogbn-arxiv")
            finally:
                os.chdir(original)
        self.assertEqual(from_elsewhere, from_package)
        self.assertEqual(from_elsewhere.name, "ogbn-arxiv_upstream_default.yaml")

    def test_paper_execution_records_match_verified_evidence(self) -> None:
        record_dir = ROOT / "paper_records"
        arxiv = json.loads((record_dir / "ogbn-arxiv_execution_record.json").read_text(encoding="utf-8"))
        reddit = json.loads((record_dir / "reddit_execution_record.json").read_text(encoding="utf-8"))
        self.assertEqual(arxiv["execution"]["batch_size"], 128)
        self.assertEqual(arxiv["execution"]["epochs"], 12)
        self.assertEqual(arxiv["execution"]["updates_per_epoch"], 711)
        self.assertEqual(arxiv["execution"]["total_updates"], 8532)
        self.assertEqual(arxiv["model_and_optimizer"]["scheduler_horizon_epochs"], 60)
        self.assertEqual(reddit["execution"]["batch_size"], 64)
        self.assertEqual(reddit["execution"]["epochs"], 12)
        self.assertEqual(reddit["execution"]["updates_per_epoch"], 2398)
        self.assertEqual(reddit["execution"]["total_updates"], 28776)
        self.assertEqual(reddit["provenance"]["classification"], "ported resource-adapted protocol")

    def assert_paper_record_rejected(self, name: str) -> None:
        module = self.config_module()
        path = ROOT / "paper_records" / name
        with self.assertRaisesRegex(
            ValueError,
            r"paper execution/provenance record.*not a runnable GraphMAE2 configuration",
        ):
            module.load_runtime_config(path)

    def test_arxiv_paper_record_is_rejected_as_runtime_config(self) -> None:
        self.assert_paper_record_rejected("ogbn-arxiv_execution_record.json")

    def test_reddit_paper_record_is_rejected_as_runtime_config(self) -> None:
        self.assert_paper_record_rejected("reddit_execution_record.json")

    def test_upstream_flat_config_still_loads(self) -> None:
        module = self.config_module()
        path = module.resolve_config_path("ogbn-arxiv")
        config = module.load_runtime_config(path)
        self.assertEqual(config["batch_size"], 512)
        self.assertEqual(config["max_epoch"], 60)
        self.assertEqual(config["lr"], 0.0025)

    def test_paper_records_are_separate_from_runtime_configs(self) -> None:
        config_dir = ROOT / "code" / "graphmae2" / "configs"
        record_dir = ROOT / "paper_records"
        self.assertEqual(list(config_dir.glob("*_paper.yaml")), [])
        self.assertTrue((record_dir / "ogbn-arxiv_execution_record.json").is_file())
        self.assertTrue((record_dir / "reddit_execution_record.json").is_file())


class AnalysisCliTests(unittest.TestCase):
    def test_all_public_cli_help_commands_pass(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "audit_public_clis.py")],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_bootstrap_cli_creates_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code" / "diagnostics" / "bootstrap_reference_interval.py"),
                    "--input",
                    str(ROOT / "processed_records" / "arxiv" / "reference_diagnostic_records.csv"),
                    "--dataset",
                    "ogbn-arxiv",
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())

    def test_weighting_summary_accepts_released_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "summary.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code" / "comparisons" / "summarize_weighting_methods.py"),
                    "--input",
                    str(ROOT / "processed_records" / "method_comparisons" / "weighting_methods.csv"),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertIn("Ours lambda=0.3", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
