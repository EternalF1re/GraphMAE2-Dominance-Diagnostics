"""Scalar-only tests of the ten-seed factorial analysis."""
import importlib.util
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("factorial_analysis", ROOT/"code/analysis/factorial.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


class FactorialRevisionTests(unittest.TestCase):
    def test_unhalved_interaction_and_high_minus_low(self):
        self.assertEqual(F.factorial_effects({"G1": 2., "G2": 1., "G3": 0., "G4": 3.}),
                         {"encoder": 2., "head": 0., "interaction": -2.})

    def test_actual_update_spacing_auc(self):
        points = {u: .6+.1*(u-250)/8282 for u in F.UPDATES}
        self.assertAlmostEqual(F.trajectory_auc(points), .65)
        with self.assertRaises(ValueError): F.trajectory_auc({250:.6,8532:.7})

    def test_ten_paired_units_and_simultaneous_scope(self):
        row = F.paired_summary(list(range(10)))
        self.assertEqual(row["df"], 9)
        self.assertEqual(row["sd_pp"], statistics.stdev(range(10)))
        self.assertGreater(row["ci95_simultaneous_upper_pp"], row["ci95_nominal_upper_pp"])
        with self.assertRaises(ValueError): F.paired_summary(list(range(30)))

    def test_full_coverage_and_fixed_probe_means(self):
        rows = F.read_records(ROOT)
        self.assertEqual(len(rows), 560)
        self.assertEqual(sum(r["global_update"] == 8532 for r in rows), 80)
        changed = [dict(r) for r in rows]
        changed[0]["probe_54321"] += .1
        with self.assertRaises(ValueError): F.validate_records(changed)
        with self.assertRaises(ValueError): F.validate_records(rows[:-1])

    def test_three_seed_legacy_protocol_not_pooled(self):
        rows = F.read_records(ROOT)
        with self.assertRaises(ValueError):
            F.validate_records([r for r in rows if r["pretraining_seed"] in [0,1,2]])

    def test_all_effects_match_frozen_values_not_only_encoder(self):
        _, effects, _ = F.analyze(F.read_records(ROOT), "auc")
        frozen = F.read_csv(ROOT/"processed_records/factorial/frozen_auc_effects.csv")
        lookup = {(r["protocol"], r["effect"]): r for r in effects}
        for row in frozen:
            result = lookup[row["protocol"], row["effect"]]
            for key in ["estimate_pp", "ci95_simultaneous_lower_pp", "ci95_simultaneous_upper_pp"]:
                self.assertTrue(math.isclose(result[key], float(row[key]), rel_tol=0, abs_tol=1e-10))
