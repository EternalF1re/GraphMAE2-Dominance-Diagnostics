"""Reproduce MaskGAE/PubMed results from public scalars, never models or labels.

Statistical formulas follow the authors' scalar analysis: independent units are
the ten pretraining seeds, not the three nested classifier replicates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as S

EPOCHS = (15, 42, 83, 125, 208, 333, 500)
SEEDS = tuple(range(82001, 82011))
ALPHAS = (.001, .125, .25, .5, 1.)
NEW_ALPHAS = (.125, .25, 1.)
PROBE_SEEDS = (54321, 54322, 54323)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    need(type(value) in (int, float) and math.isfinite(value), "Finite numeric scalar required")
    return value


def load_records(repo_root):
    root = Path(repo_root) / "processed_records/maskgae_pubmed"
    integers = {"pretraining_seed", "epoch", "minibatch", "probe_seed", "best_correct",
                "best_epoch", "completed_steps", "clip_triggered", "optimizer_updates",
                "clipped_updates", "completed_epochs", "training_forwards", "backwards",
                "full_z_rows", "full_z_columns", "diagnostic_seed"}
    tables = {}
    for name, filename in (("checkpoints", "checkpoints.csv"), ("probes", "probes.csv"),
                           ("updates", "training_updates.csv"), ("runs", "run_counts.csv")):
        with (root / filename).open(newline="", encoding="utf-8") as stream:
            tables[name] = [{k: int(v) if k in integers else float(v) for k, v in r.items()}
                            for r in csv.DictReader(stream)]
    return tables


def exact(rows, fields, expected, name):
    keys = [tuple(r[k] for k in fields) for r in rows]
    need(len(keys) == len(expected) and len(set(keys)) == len(keys) and set(keys) == expected,
         "Exact complete nonduplicate coverage required: " + name)


def validate_records(data):
    cp, probes, updates, runs = (data[k] for k in ("checkpoints", "probes", "updates", "runs"))
    run_keys = {(s, a) for s in SEEDS for a in ALPHAS}
    cp_keys = {(s, a, e) for s, a in run_keys for e in EPOCHS}
    exact(cp, ("pretraining_seed", "alpha", "epoch"), cp_keys, "checkpoints")
    exact(probes, ("pretraining_seed", "alpha", "epoch", "probe_seed"),
          {(*key, p) for key in cp_keys for p in PROBE_SEEDS}, "nested probes")
    exact(updates, ("pretraining_seed", "alpha", "epoch", "minibatch"),
          {(s, a, e, 0) for s, a in run_keys for e in range(1, 501)}, "observed optimizer updates")
    exact(runs, ("pretraining_seed", "alpha"), run_keys, "run counts")
    probe_lookup = {}
    for row in probes:
        need(row["completed_steps"] == 100 and 1 <= row["best_epoch"] <= 100, "Fixed 100-step probe required")
        need(type(row["best_correct"]) is int and 0 <= row["best_correct"] <= 500, "Correct-count range")
        need(math.isclose(finite(row["best_accuracy"]), row["best_correct"] / 500, abs_tol=1e-14, rel_tol=0),
             "Probe correct-count/accuracy mismatch")
        probe_lookup[tuple(row[k] for k in ("pretraining_seed", "alpha", "epoch", "probe_seed"))] = row["best_accuracy"]
    for row in cp:
        key = tuple(row[k] for k in ("pretraining_seed", "alpha", "epoch"))
        values = [probe_lookup[(*key, p)] for p in PROBE_SEEDS]
        need(0 <= finite(row["validation_accuracy"]) <= 1 and
             math.isclose(row["validation_accuracy"], S.mean(values), abs_tol=1e-14, rel_tol=0),
             "Checkpoint must be mean of three nested probes")
        for p, value in zip(PROBE_SEEDS, values):
            need(math.isclose(finite(row["probe_" + str(p)]), value, abs_tol=1e-14, rel_tol=0), "Wide/long probe mismatch")
        rec, degree, ratio = (finite(row[k]) for k in ("rec_Z_norm", "weighted_degree_Z_norm", "preclipping_R_Z"))
        need(rec > 1e-12 and degree >= 0 and ratio >= 0 and
             math.isclose(ratio, degree / rec, rel_tol=2e-12, abs_tol=1e-15), "Full-Z weighted ratio mismatch")
        need((row["full_z_rows"], row["full_z_columns"]) == (19717, 64), "Full shared Z support required")
        need(row["diagnostic_seed"] == 910000000 + 1000 * row["pretraining_seed"] + row["epoch"], "Diagnostic seed mismatch")
    observed = {key: [0, 0] for key in run_keys}
    for row in updates:
        norm, coefficient = finite(row["preclip_norm"]), finite(row["clip_coefficient"])
        need(norm >= 0 and 0 <= coefficient <= 1, "Invalid clipping scalar")
        need(row["clip_triggered"] in (0, 1) and row["clip_triggered"] == int(norm > 1), "Clipping threshold mismatch")
        need(math.isclose(coefficient, min(1., 1. / (norm + 1e-6)), rel_tol=2e-6, abs_tol=1e-7), "Native FP32 clipping coefficient mismatch")
        edge, degree, joint = (finite(row[k]) for k in ("edge_loss", "degree_loss_raw", "joint_loss"))
        need(edge >= 0 and degree >= 0 and math.isclose(joint, edge + row["alpha"] * degree, rel_tol=2e-6, abs_tol=2e-7), "Joint-alpha loss mismatch")
        counts = observed[row["pretraining_seed"], row["alpha"]]
        counts[0] += 1
        counts[1] += row["clip_triggered"]
    for row in runs:
        count, clipped = observed[row["pretraining_seed"], row["alpha"]]
        need(row["completed_epochs"] == 500 and
             row["training_forwards"] == row["backwards"] == row["optimizer_updates"] == count == 500,
             "Actual complete 500-update denominator required")
        need(row["clipped_updates"] == clipped, "Recorded clipping count mismatch")
    return data


def trajectory_auc(points):
    need(set(points) == set(EPOCHS), "Fixed seven-epoch grid required")
    need(all(0 <= finite(v) <= 1 for v in points.values()), "Accuracy fraction required")
    return math.fsum((points[a] + points[b]) * (b - a) / 2 for a, b in zip(EPOCHS, EPOCHS[1:])) / 485


def seed_median_ratio(values):
    need(len(values) == 10 and all(len(v) == 7 for v in values), "Ten seeds, seven ratios per seed required")
    need(all(finite(r) >= 0 for v in values for r in v), "Finite nonnegative ratios required")
    return S.median(S.median(v) for v in values)


def paired_summary(values):
    from scipy.stats import t
    need(len(values) == 10 and all(math.isfinite(finite(v)) for v in values), "Ten finite independent seed values required")
    mean, sd = S.mean(values), S.stdev(values)
    se = sd / math.sqrt(10)
    critical = float(t.ppf(.975, 9))
    statistic = mean / se if se else None
    return dict(n=10, df=9, mean=mean, sd=sd, ci95_lower=mean - critical * se,
                ci95_upper=mean + critical * se, t=statistic,
                p_unadjusted=float(2 * t.sf(abs(statistic), 9)) if statistic is not None else None,
                ci_method="ordinary_two_sided_student_t_95_df9",
                inference_status="ESTIMABLE" if se else "NOT_ESTIMABLE_ZERO_SD")


def holm3(values):
    need(len(values) == 3, "Exactly three exploratory family members required")
    if any(v is None for v in values):
        return [None] * 3
    need(all(0 <= finite(v) <= 1 for v in values), "Finite p-values required")
    result, previous = [None] * 3, 0.
    for rank, index in enumerate(sorted(range(3), key=lambda i: values[i])):
        previous = min(1., max(previous, (3 - rank) * values[index]))
        result[index] = previous
    return result


def analyze(data):
    validate_records(data)
    lookup = {(r["pretraining_seed"], r["alpha"], r["epoch"]): r for r in data["checkpoints"]}
    seed_rows, groups, contrasts = [], [], []
    for alpha in ALPHAS:
        for seed in SEEDS:
            points = {e: lookup[seed, alpha, e]["validation_accuracy"] for e in EPOCHS}
            seed_rows.append(dict(pretraining_seed=seed, alpha=alpha, auc=trajectory_auc(points),
                                  epoch500_accuracy=points[500], median_preclipping_R_Z=S.median(
                                      lookup[seed, alpha, e]["preclipping_R_Z"] for e in EPOCHS)))
        current = [r for r in seed_rows if r["alpha"] == alpha]
        auc = paired_summary([r["auc"] for r in current])
        endpoint = paired_summary([r["epoch500_accuracy"] for r in current])
        runs = [r for r in data["runs"] if r["alpha"] == alpha]
        total = sum(r["optimizer_updates"] for r in runs)
        clipped = sum(r["clipped_updates"] for r in runs)
        groups.append(dict(alpha=alpha, n=10, auc_mean=auc["mean"], auc_sd=auc["sd"],
            auc_ci95_lower=auc["ci95_lower"], auc_ci95_upper=auc["ci95_upper"],
            epoch500_mean=endpoint["mean"], epoch500_sd=endpoint["sd"],
            epoch500_ci95_lower=endpoint["ci95_lower"], epoch500_ci95_upper=endpoint["ci95_upper"],
            median_preclipping_R_Z=seed_median_ratio([[lookup[s, alpha, e]["preclipping_R_Z"] for e in EPOCHS] for s in SEEDS]),
            optimizer_updates=total, clipped_updates=clipped, clipped_update_percent=100 * clipped / total))
    by_seed = {(r["pretraining_seed"], r["alpha"]): r for r in seed_rows}
    for alpha in ALPHAS[1:]:
        row = dict(alpha=alpha, reference_alpha=.001, n=10, df=9,
                   comparison_role="historical_separate" if alpha == .5 else "new_exploratory",
                   auc_family="historical_separate" if alpha == .5 else "exploratory_three_new_doses",
                   epoch500_family="historical_separate" if alpha == .5 else "exploratory_secondary_three_new_doses",
                   ci_method="ordinary_two_sided_student_t_95_df9")
        for endpoint, key in (("auc", "auc"), ("epoch500", "epoch500_accuracy")):
            summary = paired_summary([100 * (by_seed[s, alpha][key] - by_seed[s, .001][key]) for s in SEEDS])
            row.update({endpoint + "_" + name: value for name, value in (
                ("delta_pp", summary["mean"]), ("sd_pp", summary["sd"]), ("ci95_lower_pp", summary["ci95_lower"]),
                ("ci95_upper_pp", summary["ci95_upper"]), ("p_unadjusted", summary["p_unadjusted"]),
                ("p_holm3", None), ("t", summary["t"]), ("inference_status", summary["inference_status"]))})
        contrasts.append(row)
    new = [r for r in contrasts if r["alpha"] in NEW_ALPHAS]
    for endpoint in ("auc", "epoch500"):
        for row, adjusted in zip(new, holm3([r[endpoint + "_p_unadjusted"] for r in new])):
            row[endpoint + "_p_holm3"] = adjusted
    return dict(dataset="PubMed", model="MaskGAE", coverage=dict(pretraining_runs=50, checkpoints=350,
                probe_replicates=1050, optimizer_updates=25000, independent_seeds_per_dose=10),
                protocol=dict(auc_divisor=485, checkpoint_epochs=list(EPOCHS), probe_replicates_per_checkpoint=3,
                    accuracy_units="fraction", contrast_units="percentage_points", ratio_summary="median_of_ten_seed_medians_of_seven",
                    clipping_definition="preclip_global_norm_strictly_greater_than_one",
                    historical_half_contrast_separate=True, shared_controls_not_independent_replication=True,
                    new_doses_selected_after_historical_results=True, ci_multiplicity_adjusted=False),
                groups=groups, contrasts=contrasts, seeds=seed_rows)


def reproduce(repo_root, outdir):
    data = load_records(repo_root)
    result = analyze(data)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name, rows in (("dose_summary", result["groups"]), ("paired_contrasts", result["contrasts"]),
                       ("seed_metrics", result["seeds"]), ("run_clipping", data["runs"])):
        with (outdir / (name + ".csv")).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.repo_root, args.outdir)
