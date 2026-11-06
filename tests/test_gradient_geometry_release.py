"""CPU scalar-record reproduction only; no measurement or model execution."""
import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code/analysis/gradient_geometry.py"


def analysis_module():
    spec = importlib.util.spec_from_file_location("geometry_release_analysis", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeometryReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = analysis_module()
        cls.records = cls.analysis.load_records(ROOT)

    def test_complete_120_measurement_grid(self):
        expected = {(g, s, u, b) for g in ("G1", "G2") for s in range(10)
                    for u in (250, 3555, 8532) for b in (93, 118)}
        self.assertEqual(len(self.records), 120)
        self.assertEqual({self.analysis.record_key(r) for r in self.records}, expected)

    def test_each_group_uses_its_own_measured_local_ratio(self):
        for row in self.records:
            local = row["weighted_aux_gradient_norm"] / row["reconstruction_gradient_norm"]
            routed = local * (1.0 if row["group"] == "G1" else .03)
            self.assertAlmostEqual(row["weighted_ratio"], local, places=12)
            self.assertAlmostEqual(row["routed_weighted_ratio_derived"], routed, places=12)
            self.assertEqual(row["model_effective_lambda"], 10.)
            self.assertEqual(row["training_head_coefficient"], 10.)

    def test_hierarchy_and_sample_sd(self):
        result = self.analysis.aggregate(self.records)
        for group in ("G1", "G2"):
            seed_values = []
            for seed in range(10):
                checkpoint_values = [statistics.mean(
                    r["weighted_ratio"] for r in self.records
                    if (r["group"], r["seed"], r["update"]) == (group, seed, update))
                    for update in (250, 3555, 8532)]
                seed_values.append(statistics.mean(checkpoint_values))
            actual = result["groups"][group]["local_ratio"]
            self.assertEqual(actual["mean"], statistics.mean(seed_values))
            self.assertEqual(actual["sample_sd"], statistics.stdev(seed_values))
        self.assertEqual(result["independent_seed_count"], 10)
        self.assertEqual(len(result["per_checkpoint"]), 60)
        self.assertEqual(len(result["per_seed"]), 20)

    def test_published_numbers(self):
        result = self.analysis.aggregate(self.records)
        expected = {
            "G1": (2.6482906322271846, 2.6482906322271846,
                   .002680732223918117, .03474043617235971),
            "G2": (4.344149434420174, .13032448303260524,
                   .003132627953196282, .0381831393844813),
        }
        for group, values in expected.items():
            for metric, value in zip(("local_ratio", "routed_ratio_derived",
                                      "full_z_cosine", "target_cosine"), values):
                self.assertAlmostEqual(result["groups"][group][metric]["mean"], value, places=14)
        self.assertAlmostEqual(result["routed_group_mean_ratio_g1_over_g2"],
                               expected["G1"][1] / expected["G2"][1], places=12)

    def test_all_128_target_rows_including_zeros_are_retained(self):
        self.assertTrue(all(r["target_count"] == 128 for r in self.records))
        self.assertTrue(any(r["target_reconstruction_nonzero_rows"] < 128 for r in self.records))
        for row in self.records:
            self.assertLessEqual(row["target_shared_nonzero_rows"], row["target_count"])
            self.assertGreater(row["full_z_rows"], row["target_count"])
            self.assertAlmostEqual(row["target_weighted_aux_gradient_norm"],
                                   row["weighted_aux_gradient_norm"], places=15)

    def test_missing_duplicate_and_undefined_records_fail_closed(self):
        for rows in (self.records[:-1], self.records[:-1] + [self.records[0]]):
            with self.assertRaises(ValueError):
                self.analysis.aggregate(rows)
        rows = copy.deepcopy(self.records)
        rows[0]["cosine"] = None
        with self.assertRaises(ValueError):
            self.analysis.aggregate(rows)

    def test_wrong_routing_and_nonfinite_values_are_rejected(self):
        for field, value in (("routed_weighted_ratio_derived", 999.),
                             ("training_shared_coefficient", 999.),
                             ("weighted_ratio", float("nan"))):
            rows = copy.deepcopy(self.records)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                self.analysis.aggregate(rows)

    def test_identity_hashes_and_paired_stochastic_fields(self):
        path = ROOT / "processed_records/gradient_geometry/identities.json"
        identities = json.loads(path.read_text())["identities"]
        self.assertEqual(len(identities), 120)
        self.assertEqual({self.analysis.record_key(r) for r in identities},
                         {self.analysis.record_key(r) for r in self.records})
        index = {self.analysis.record_key(r): r for r in identities}
        for seed in range(10):
            for update in (250, 3555, 8532):
                for position in (93, 118):
                    a = index[("G1", seed, update, position)]
                    b = index[("G2", seed, update, position)]
                    for field in self.analysis.PAIRED_IDENTITY_FIELDS:
                        self.assertEqual(a[field], b[field])

    def test_portable_reproduce_and_table_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "portable"
            shutil.copytree(ROOT / "processed_records/gradient_geometry",
                            root / "processed_records/gradient_geometry")
            output = root / "output"
            metrics = self.analysis.reproduce(root, output)
            self.assertEqual(metrics["measurement_count"], 120)
            self.assertEqual(json.loads((output / "summary.json").read_text()), metrics)
            for name, count in (("per_checkpoint.csv", 60), ("per_seed.csv", 20),
                                ("group_summary.csv", 2), ("paired_seed_differences.csv", 10)):
                with (output / name).open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)

    def test_tampered_processed_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / "processed_records/gradient_geometry"
            shutil.copytree(ROOT / "processed_records/gradient_geometry", dest)
            with (dest / "measurements.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaises(ValueError):
                self.analysis.load_records(root)

    def test_reproduction_cannot_overwrite_inputs(self):
        with self.assertRaises(ValueError):
            self.analysis.reproduce(ROOT, ROOT / "processed_records/gradient_geometry")

    def _probe_free_subprocess(self, preloaded=False, extra=""):
        with tempfile.TemporaryDirectory() as directory:
            command = (
                "import builtins,importlib.util,pathlib,sys,types;"
                + ("sys.modules.setdefault('numpy',types.ModuleType('numpy'));"
                   "sys.modules.setdefault('pandas',types.ModuleType('pandas'));"
                   if preloaded else "") +
                "original=builtins.__import__;"
                "blocked={'torch','dgl','numpy','pandas','ogb','wandb'};"
                "preexisting=blocked.intersection(sys.modules);"
                "exec(\"def guarded(name,*a,**k):\\n "
                "if name.split('.')[0] in blocked: raise AssertionError(name)\\n "
                "return original(name,*a,**k)\");"
                "builtins.__import__=guarded;"
                "s=importlib.util.spec_from_file_location('analysis'," + repr(str(SOURCE)) + ");"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                + extra +
                "r=m.reproduce(pathlib.Path(" + repr(str(ROOT)) + "),pathlib.Path(" + repr(directory) + "));"
                "assert r['measurement_count']==120;"
                "assert not (blocked.intersection(sys.modules)-preexisting);print('SCALAR_ONLY_OK')"
            )
            return subprocess.run([sys.executable, "-I", "-B", "-c", command],
                                  capture_output=True, text=True, timeout=30)

    def test_probe_free_standard_library_subprocess(self):
        done = self._probe_free_subprocess()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("SCALAR_ONLY_OK", done.stdout)

    def test_probe_free_subprocess_accepts_preloaded_packages(self):
        # Startup sentinels avoid requiring pandas just to simulate its presence.
        done = self._probe_free_subprocess(preloaded=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("SCALAR_ONLY_OK", done.stdout)

    def test_probe_free_guard_rejects_active_import_even_when_preloaded(self):
        for preloaded in (False, True):
            with self.subTest(preloaded=preloaded):
                done = self._probe_free_subprocess(preloaded=preloaded,
                                                   extra="exec('import numpy',m.__dict__);")
                self.assertNotEqual(done.returncode, 0)
                self.assertIn("AssertionError: numpy", done.stderr)

    def test_probe_free_check_rejects_new_blocked_sys_modules_entry(self):
        done = self._probe_free_subprocess(
            extra="blocked.add('_geometry_blocked_probe');"
                  "sys.modules['_geometry_blocked_probe']=types.ModuleType('_geometry_blocked_probe');")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("AssertionError", done.stderr)

    def test_inspection_source_is_non_runnable_and_has_exact_vjp_expression(self):
        path = ROOT / "code/measurements/gradient_geometry_components.py"
        tree = ast.parse(path.read_text())
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                           ast.ClassDef)) for n in tree.body))
        self.assertIn("torch.autograd.grad(sum(recs),z,retain_graph=True)", path.read_text())
        self.assertIn("torch.autograd.grad(10.*latents[0],z,retain_graph=True)", path.read_text())

    def test_dictionary_covers_all_csv_columns_and_json_fields(self):
        directory = ROOT / "processed_records/gradient_geometry"
        text = (directory / "DATA_DICTIONARY.md").read_text()
        with (directory / "measurements.csv").open(newline="") as stream:
            for field in csv.DictReader(stream).fieldnames:
                self.assertIn("| `" + field + "` |", text)

        def fields(value, prefix=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    path = (prefix + '["' + key + '"]' if "." in key else
                            prefix + ("." if prefix else "") + key)
                    yield path
                    yield from fields(item, path)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (dict, list)):
                        yield from fields(item, prefix + "[]")

        for name in ("identities.json", "provenance.json"):
            for field in set(fields(json.loads((directory / name).read_text()))):
                self.assertIn("| `" + field + "` |", text, name)

    def test_public_owned_artifacts_have_no_private_paths_or_execution_ids(self):
        paths = list((ROOT / "processed_records/gradient_geometry").rglob("*"))
        paths += [SOURCE, ROOT / "code/measurements/gradient_geometry_components.py", Path(__file__)]
        paths += list((ROOT / "experiments/gradient_geometry").rglob("*"))
        forbidden = ("/" + "home/", "/" + "mnt/", "research_" + "extensions/",
                     "PRIVATE" + " KEY", "20" + "2611", "G1G2_" + "FORMAL",
                     "Phase" + " A", "phase" + "2c")
        for path in paths:
            if path.is_file():
                content = path.read_text()
                for token in forbidden:
                    self.assertNotIn(token, content, str(path))


if __name__ == "__main__":
    unittest.main()
