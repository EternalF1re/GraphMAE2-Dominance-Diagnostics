"""CPU scalar release checks; no model imports or research-data resampling."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "code/analysis/estimator_reliability.py"
DATA = ROOT / "processed_records/estimator_reliability"


class ReliabilityReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODULE.is_file():
            raise AssertionError("Missing CPU scalar reliability reproducer")
        spec = importlib.util.spec_from_file_location("reliability_release", MODULE)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_complete_reproduction_without_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.module, "historical_conditional_bootstrap",
                              side_effect=AssertionError("No research-data resampling")):
                metrics = self.module.reproduce(ROOT, Path(tmp) / "tables")
            self.assertEqual(metrics["independent_seed_count"], 10)
            self.assertEqual(metrics["panel_count"], 2000)
            self.assertEqual(metrics["scalar_measurement_count"], 10240)
            self.assertAlmostEqual(metrics["stratified_mae_percent"], 7.730391653991314)
            self.assertAlmostEqual(metrics["uniform_mae_percent"], 7.581631481816684)
            self.assertAlmostEqual(metrics["paired_mae_difference_pp"], .14876017217463026)
            self.assertEqual(metrics["all_retained_count"], 738)
            self.assertEqual(metrics["all_bad_retained_count"], 27)
            self.assertEqual(metrics["no_refusal_bad_count"], 37)
            self.assertEqual(metrics["refusal_gate_failures"], {"CI": 53, "PANEL": 237, "CHECKPOINT": 0})
            self.assertAlmostEqual(metrics["all_bad_retained_percent"], 100 * 27 / 738)
            self.assertAlmostEqual(metrics["apparent_reference_point_coverage_percent"], 80.9)
            self.assertEqual(metrics["new_bootstrap_resamples"], 0)
            self.assertEqual(metrics["execution_disposition"], "FAIL_FORMAL_EXECUTION_GATE")
            self.assertEqual(metrics["data_disposition"], "CONDITIONAL_ACCEPT_DATA")
            self.assertEqual(metrics["terminal_rss_failure_timing"], "UNRESOLVED")
            interval = metrics["stored_paired_mae_difference_ci_pp"]
            self.assertAlmostEqual(interval[0], -.15097501375565432)
            self.assertAlmostEqual(interval[1], .44431373836266615)
            tables = Path(tmp) / "tables"
            self.assertTrue((tables / "reference_checks.csv").is_file())
            self.assertTrue((tables / "refusal_ablations.csv").is_file())
            self.assertEqual(len(self.module.read_csv(tables / "statewise_sampling_errors.csv")), 4000)
            uniform = [r for r in self.module.read_csv(tables / "sampling_panels.csv")
                       if r["method"] == "uniform"]
            self.assertEqual(len(uniform), 1000)
            self.assertFalse(any("ci" in k.lower() or "gate" in k.lower()
                                 or "coverage" in k.lower() for k in uniform[0]))

    def test_ten_seed_equal_weighting_is_not_pooled_acceptance_weighting(self):
        groups = [[1.], [3.] * 9]
        self.assertEqual(self.module.seed_equal_mean(groups), 2.)
        self.assertAlmostEqual(self.module.pooled_mean(groups), 2.8)
        with self.assertRaises(ValueError):
            self.module.seed_equal_mean([[1.], []])
        with tempfile.TemporaryDirectory() as tmp:
            metrics = self.module.reproduce(ROOT, Path(tmp) / "tables")
            self.assertAlmostEqual(metrics["all_retained_mae_seed_equal_percent"], 7.943535419223971)
            self.assertAlmostEqual(metrics["all_retained_mae_pooled_percent"], 7.883999759117166)
            rows = self.module.read_csv(Path(tmp) / "tables/seed_summary.csv")
            self.assertEqual({int(r["seed"]) for r in rows}, set(range(10)))
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(self.module.read_csv(Path(tmp) / "tables/statewise_summary.csv")), 40)

    def test_pool_panel_separation_and_genuine_uniform(self):
        batches = self.module.read_batches(DATA / "batch_scalars.csv")
        panels = self.module.read_csv(DATA / "panels.csv")
        self.module.validate_design(batches, panels)
        stratified = [p for p in panels if p["method"] == "stratified"]
        uniform = [p for p in panels if p["method"] == "uniform"]
        self.assertEqual(len(stratified), 1000)
        self.assertEqual(len(uniform), 1000)
        self.assertTrue(any(self.module.panel_counts(p, batches) != [4] * 4 for p in uniform))
        # Accidental quotas never grant a Uniform panel historical screen eligibility.
        result = self.module.evaluate_panel(uniform[0], batches, 30.)
        self.assertNotIn("accepted", result)
        self.assertNotIn("R_ci", result)
        bad = [dict(p) for p in panels]
        bad[0]["batch_ids"] = ";".join([bad[0]["batch_ids"].split(";")[0]] * 16)
        with self.assertRaises(ValueError):
            self.module.validate_design(batches, bad)

    def test_no_seed_dropping_or_finite_support_ablation(self):
        batches = self.module.read_batches(DATA / "batch_scalars.csv")
        missing = {k: v for k, v in batches.items() if v["seed"] != 9}
        with self.assertRaises(ValueError):
            self.module.validate_design(missing, self.module.read_csv(DATA / "panels.csv"))
        self.assertTrue(math.isclose(self.module.norm_ratio(0., 1.), 1e30, rel_tol=1e-15))
        for rec, aux in [(1., 0.), (float("nan"), 1.), (-1., 1.)]:
            with self.assertRaises(ValueError):
                self.module.norm_ratio(rec, aux)

    def test_outputs_do_not_overwrite_inputs_or_existing_results(self):
        before = {p.name: p.read_bytes() for p in DATA.iterdir() if p.is_file()}
        with self.assertRaises(ValueError):
            self.module.reproduce(ROOT, DATA / "generated")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tables"
            self.module.reproduce(ROOT, out)
            saved = (out / "metrics.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self.module.reproduce(ROOT, out)
            self.assertEqual((out / "metrics.json").read_bytes(), saved)
        self.assertEqual(before, {p.name: p.read_bytes() for p in DATA.iterdir() if p.is_file()})

    def test_stored_intervals_are_identified_not_recomputed(self):
        frozen = json.loads((DATA / "frozen_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(frozen["interval_origin"], "STORED_NOT_RESAMPLED")
        self.assertFalse(frozen["population_coverage_certified"])
        with tempfile.TemporaryDirectory() as tmp:
            self.module.reproduce(ROOT, Path(tmp) / "tables")
            rows = self.module.read_csv(Path(tmp) / "tables/stored_intervals.csv")
            self.assertTrue(all(r["interval_origin"] == "STORED_NOT_RESAMPLED" for r in rows))
            self.assertTrue(any(r["metric"] == "uniform_mae" for r in rows))
            self.assertTrue(all(r["interval_target"] != "uniform_ratio" for r in rows))

    def test_dictionary_names_every_canonical_column_and_json_key(self):
        dictionary = (DATA / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
        for path in DATA.glob("*.csv"):
            self.assertIn(path.name, dictionary)
            for column in self.module.read_csv(path)[0]:
                self.assertIn("`" + column + "`", dictionary, path.name)
        for key in json.loads((DATA / "frozen_summary.json").read_text(encoding="utf-8")):
            self.assertIn("`" + key + "`", dictionary)

    def test_public_payload_has_no_private_paths_or_runtime_machinery(self):
        files = list(DATA.rglob("*")) + [MODULE]
        files += list((ROOT / "experiments/estimator_reliability").rglob("*"))
        for path in files:
            if path.is_file():
                text = path.read_text(encoding="utf-8").lower()
                for forbidden in ["/ho"+"me/", "/m"+"nt/", "d"+chr(58)+"\\", "private.pem", "approval.sig"]:
                    self.assertNotIn(forbidden, text, str(path.relative_to(ROOT)))

    def test_inspection_kernel_and_sampling_formulas_on_tiny_synthetic_values(self):
        class FixedChoices:
            def choice(self, values):
                return values[0]

            def sample(self, values, count):
                return values[:count]

            def randrange(self, count):
                return 0

        rows = [dict(checkpoint_role=cp, degree_stratum=s,
                     weighted_latent_reconstruction_norm_ratio=2. + s)
                for cp in ("early", "late") for s in range(4) for _ in range(4)]
        lambdas, ratios = self.module.historical_conditional_bootstrap(rows, FixedChoices(), 2)
        self.assertEqual(ratios, [3.5, 3.5])
        self.assertEqual(lambdas, [10 / 3.5, 10 / 3.5])
        self.assertAlmostEqual(self.module.percentile([1., 3.], .025), 1.05)
        strata = [["synthetic_%d_%03d" % (s, i) for i in range(128)] for s in range(4)]
        pools = self.module.historical_pool_partition(strata)
        self.assertEqual([len(pools[k]) for k in ("reference_a", "reference_b", "estimation")], [128, 128, 256])
        estimation = [s[64:128] for s in strata]
        self.assertEqual(len(self.module.historical_panel_sample(estimation, "stratified", FixedChoices())), 16)
        self.assertEqual(self.module.historical_panel_sample(estimation, "uniform", FixedChoices()), estimation[0][:16])
        self.assertEqual(self.module.historical_seed_bundle_choices(FixedChoices()), [(0, 0)] * 10)
        self.assertEqual(self.module.seed_from_metadata({"a": 1, "b": 2}),
                         self.module.seed_from_metadata({"b": 2, "a": 1}))
        order = self.module.historical_asset_pool_order(["a", "b", "c"], "synthetic", 0, 0)
        self.assertEqual(order, self.module.historical_asset_pool_order(["c", "a", "b"], "synthetic", 0, 0))


if __name__ == "__main__":
    unittest.main()
