"""Scalar-only release checks. No model, dataset, label or evaluator imports."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


A = module("maskgae_analysis", "code/analysis/maskgae_pubmed.py")
W = module("maskgae_weighting", "code/experiments/maskgae_weighting.py")
M = module("maskgae_measurements", "code/measurements/maskgae_components.py")


class ScalarReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = A.load_records(ROOT)

    def test_exact_checkpoint_probe_update_and_run_coverage(self):
        A.validate_records(self.data)
        self.assertEqual(len(self.data["checkpoints"]), 350)
        self.assertEqual(len(self.data["probes"]), 1050)
        self.assertEqual(len(self.data["updates"]), 25000)
        self.assertEqual(len(self.data["runs"]), 50)

    def test_missing_or_duplicate_checkpoint_is_rejected(self):
        for rows in (self.data["checkpoints"][:-1], self.data["checkpoints"] + self.data["checkpoints"][:1]):
            changed = dict(self.data, checkpoints=rows)
            with self.assertRaises(ValueError):
                A.validate_records(changed)

    def test_missing_seed_and_unplanned_dose_are_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["checkpoints"][0]["alpha"] = .75
        with self.assertRaises(ValueError):
            A.validate_records(changed)
        changed = dict(self.data, runs=self.data["runs"][:-1])
        with self.assertRaises(ValueError):
            A.validate_records(changed)

    def test_nested_probes_are_not_independent_pretraining_units(self):
        result = A.analyze(self.data)
        self.assertTrue(all(r["n"] == 10 for r in result["contrasts"]))
        self.assertEqual(result["protocol"]["probe_replicates_per_checkpoint"], 3)

    def test_checkpoint_accuracy_is_mean_of_exact_three_probes(self):
        changed = copy.deepcopy(self.data)
        changed["checkpoints"][0]["validation_accuracy"] += .01
        with self.assertRaises(ValueError):
            A.validate_records(changed)

    def test_probe_steps_seed_and_correct_count_are_checked(self):
        for field, value in [("completed_steps", 99), ("probe_seed", 1), ("best_correct", 499)]:
            changed = copy.deepcopy(self.data)
            changed["probes"][0][field] = value
            with self.assertRaises(ValueError):
                A.validate_records(changed)

    def test_ratio_is_weighted_norm_over_edge_norm(self):
        changed = copy.deepcopy(self.data)
        changed["checkpoints"][0]["weighted_degree_Z_norm"] *= 2
        with self.assertRaises(ValueError):
            A.validate_records(changed)

    def test_nonfinite_scalars_are_rejected(self):
        for table, field in [("checkpoints", "preclipping_R_Z"), ("updates", "preclip_norm")]:
            changed = copy.deepcopy(self.data)
            changed[table][0][field] = float("nan")
            with self.assertRaises(ValueError):
                A.validate_records(changed)

    def test_clipping_is_counted_from_actual_optimizer_updates(self):
        result = A.analyze(self.data)
        self.assertEqual([r["clipped_updates"] for r in result["groups"]], [0, 0, 57, 682, 3382])
        self.assertEqual([r["optimizer_updates"] for r in result["groups"]], [5000] * 5)
        self.assertEqual([r["clipped_update_percent"] for r in result["groups"]], [0., 0., 1.14, 13.64, 67.64])

    def test_clipping_denominator_and_trigger_tampering_are_rejected(self):
        for table, field, value in [("runs", "optimizer_updates", 499), ("runs", "clipped_updates", 499),
                                    ("updates", "clip_triggered", 1)]:
            changed = copy.deepcopy(self.data)
            changed[table][0][field] = value
            with self.assertRaises(ValueError):
                A.validate_records(changed)

    def test_auc_uses_all_seven_epochs_and_divisor_485(self):
        self.assertAlmostEqual(A.trajectory_auc({e: .8 for e in A.EPOCHS}), .8)
        self.assertAlmostEqual(A.trajectory_auc({e: e / 500 for e in A.EPOCHS}), (15 + 500) / 1000)
        with self.assertRaises(ValueError):
            A.trajectory_auc({15: .8, 500: .8})

    def test_two_level_median_differs_from_pooled_median(self):
        values = [[0] * 4 + [100] * 3 for _ in range(6)] + [[100] * 7 for _ in range(4)]
        self.assertEqual(A.seed_median_ratio(values), 0)
        self.assertEqual(statistics.median(sum(values, [])), 100)

    def test_ordinary_student_t_interval_not_bonferroni(self):
        row = A.paired_summary([-.03, -.02, -.01, 0, .01] * 2)
        from scipy.stats import t
        self.assertEqual(row["df"], 9)
        self.assertAlmostEqual(row["ci95_upper"] - row["mean"], t.ppf(.975, 9) * row["sd"] / math.sqrt(10))
        self.assertEqual(row["ci_method"], "ordinary_two_sided_student_t_95_df9")

    def test_holm_families_are_separate_for_each_endpoint(self):
        result = A.analyze(self.data)
        historical = next(r for r in result["contrasts"] if r["alpha"] == .5)
        self.assertIsNone(historical["auc_p_holm3"])
        self.assertIsNone(historical["epoch500_p_holm3"])
        new = [r for r in result["contrasts"] if r["alpha"] in A.NEW_ALPHAS]
        for endpoint in ("auc", "epoch500"):
            expected = A.holm3([r[endpoint + "_p_unadjusted"] for r in new])
            self.assertEqual(expected, [r[endpoint + "_p_holm3"] for r in new])
        self.assertEqual(A.holm3([.01, .04, .03]), [.03, .06, .06])

    def test_zero_sd_is_not_estimable(self):
        row = A.paired_summary([1.] * 10)
        self.assertIsNone(row["p_unadjusted"])
        self.assertEqual(row["inference_status"], "NOT_ESTIMABLE_ZERO_SD")

    def test_reproduce_writes_only_scalar_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = A.reproduce(ROOT, tmp)
            self.assertEqual(result["coverage"]["pretraining_runs"], 50)
            self.assertEqual(json.loads((Path(tmp) / "metrics.json").read_text())["coverage"], result["coverage"])
            self.assertEqual({p.suffix for p in Path(tmp).iterdir()}, {".json", ".csv"})

    def test_public_payload_has_no_internal_paths_or_signatures(self):
        paths = list((ROOT / "processed_records/maskgae_pubmed").rglob("*"))
        paths += list((ROOT / "experiments/maskgae_pubmed").rglob("*"))
        for path in paths:
            if path.is_file():
                text = path.read_text(encoding="utf-8").lower()
                for forbidden in ("/ho"+"me/", "/m"+"nt/", "phase_a", "attempt0", "approval.sig", "private.pem"):
                    self.assertFalse(forbidden in text, path.name + ": " + forbidden)
                self.assertIsNone(re.search(r"(?<![0-9a-f.])20[0-9]{6}(?![0-9a-f])", text), path.name)

    def test_dictionary_covers_every_export_column_and_protocol_section(self):
        import csv
        dictionary = (ROOT / "processed_records/maskgae_pubmed/DATA_DICTIONARY.md").read_text()
        for name in ("checkpoints", "probes", "training_updates", "run_counts"):
            with (ROOT / "processed_records/maskgae_pubmed" / (name + ".csv")).open(newline="") as stream:
                fields = next(csv.reader(stream))
            for field in fields:
                self.assertIn("`" + field + "`", dictionary)
        protocol = json.loads((ROOT / "experiments/maskgae_pubmed/executed_protocol.json").read_text())
        for key in protocol:
            self.assertIn("`" + key + "`", dictionary)


class InspectionMathTests(unittest.TestCase):
    def test_joint_alpha_affects_encoder_and_degree_head(self):
        self.assertEqual(W.joint_loss(2., 4., .25), 3.)
        self.assertEqual(W.joint_gradients([1., 0., 2.], [3., 5., 0.], .25), [1.75, 1.25, 2.])
        self.assertEqual(W.joint_loss(2., 4., .001), 2.004)

    def test_clipping_threshold_is_not_epsilon_coefficient_test(self):
        self.assertFalse(W.clip_triggered(1.))
        self.assertLess(W.clip_coefficient(1.), 1.)
        self.assertAlmostEqual(W.clip_coefficient(2.), 1 / (2 + 1e-6))

    def test_full_z_component_norms_weight_alpha_exactly_once(self):
        result = M.component_norms([[3., 0.], [0., 4.]], [[0., 12.], [5., 0.]], .25)
        self.assertEqual(result["rec_Z_norm"], 5.)
        self.assertEqual(result["weighted_degree_Z_norm"], 3.25)
        self.assertEqual(result["R"], .65)
        self.assertIsNone(M.component_norms([[0.]], [[1.]], .001)["R"])

    def test_shape_and_nonfinite_gradients_rejected(self):
        for edge, degree in [([[1.]], [[1., 2.]]), ([[float("inf")]], [[1.]])]:
            with self.assertRaises(ValueError):
                M.component_norms(edge, degree, .5)

    def test_actual_vjp_boundary_keeps_same_z_and_weights_degree_once(self):
        class Representation:
            shape = (2, 3)
            requires_grad = True
        z = Representation()
        seen = []
        def grad(loss, inputs, retain_graph):
            seen.append((loss, inputs, retain_graph))
            return (z,)
        edge, weighted = M.same_forward_vjps(z, 7., 12., .25, grad)
        self.assertIs(edge, z)
        self.assertIs(weighted, z)
        self.assertEqual(seen, [(7., (z,), True), (3., (z,), False)])


if __name__ == "__main__":
    unittest.main()
