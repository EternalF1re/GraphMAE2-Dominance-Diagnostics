"""Reproduce released sampling-error arithmetic from scalar records only.

The historical sampling functions below are inspection aids. ``reproduce`` never
calls them: conditional and seed-cluster intervals are stored observations, not
newly generated intervals. Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ABLATIONS = ("ALL", "DROP_CI", "DROP_PANEL", "DROP_CHECKPOINT", "NONE_OF_THREE")
CHECKPOINTS = ("early", "late")
METHODS = ("stratified", "uniform")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14), name)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def norm_ratio(reconstruction_norm, weighted_auxiliary_norm):
    require(all(math.isfinite(v) and v >= 0
                for v in (reconstruction_norm, weighted_auxiliary_norm)), "Invalid component norm")
    ratio = weighted_auxiliary_norm / max(reconstruction_norm, 1e-30)
    require(math.isfinite(ratio) and ratio > 0, "Nonpositive/nonfinite ratio; never ablated")
    return ratio


def seed_equal_mean(groups):
    require(bool(groups) and all(groups), "Undefined seed summary; do not drop a seed")
    return statistics.mean(statistics.mean(group) for group in groups)


def pooled_mean(groups):
    require(bool(groups) and all(groups), "Undefined seed summary; do not drop a seed")
    return statistics.mean(value for group in groups for value in group)


def percentile(values, probability):
    """Historical linear interpolation, not a new bootstrap calculation."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def round_sig(value, digits=2):
    if value == 0:
        return 0.0
    return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)


def historical_conditional_bootstrap(rows, rng, resamples=10_000):
    """Inspection-only transcription of the historical numerical kernel.

    Supply rows in canonical checkpoint/stratum/original-batch order. RNG creation
    is explicit, rather than carrying private experiment tokens into this release.
    The checkpoint cells are resampled INDEPENDENTLY, ignoring repeated batch IDs.
    This function is deliberately NOT called by the release reproduction path.
    """
    groups = {}
    for row in rows:
        groups.setdefault((row["checkpoint_role"], int(row["degree_stratum"])), []).append(
            float(row["weighted_latent_reconstruction_norm_ratio"]))
    estimates, rrefs = [], []
    for _ in range(resamples):
        sample = [rng.choice(values) for values in groups.values() for _unused in range(len(values))]
        r_ref = statistics.median(sample)
        rrefs.append(r_ref)
        estimates.append(10.0 / r_ref)
    return estimates, rrefs


def historical_panel_sample(estimation_strata, method, rng):
    """Inspection aid: no-replacement draws; Uniform has no quota conditioning."""
    require(len(estimation_strata) == 4 and all(len(s) == 64 for s in estimation_strata),
            "Four frozen estimation strata of 64 batches required")
    if method == "stratified":
        return [batch for stratum in estimation_strata for batch in rng.sample(stratum, 4)]
    require(method == "uniform", "Unknown sampling method")
    return rng.sample([batch for stratum in estimation_strata for batch in stratum], 16)


def historical_pool_partition(ranked_strata):
    """Inspection aid: value-blind rank slices, before observing gradient values."""
    require(len(ranked_strata) == 4 and all(len(s) >= 128 for s in ranked_strata),
            "At least 128 eligible candidate batches per degree stratum required")
    return {name: [batch for stratum in ranked_strata for batch in stratum[start:stop]]
            for name, start, stop in (("reference_a", 0, 32), ("reference_b", 32, 64),
                                      ("estimation", 64, 128))}


def historical_asset_pool_order(batch_hashes, protocol_token, seed, stratum):
    """Inspection aid: original value-blind canonical-payload hash ranking."""
    def digest(batch_hash):
        value = dict(protocol=protocol_token, seed=seed, stratum=stratum,
                     batch_hash=batch_hash, purpose="asset_pool")
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return sorted(batch_hashes, key=digest)


def seed_from_metadata(value):
    """Original canonical UTF-8 JSON hashing formula; original salts are not IDs."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def historical_seed_bundle_choices(rng):
    """Inspection aid: ten paired seed bundles, one stored reference draw each."""
    return [(rng.randrange(10), rng.randrange(10_000)) for _ in range(10)]


def read_batches(path):
    result = {}
    for raw in read_csv(path):
        row = dict(raw, seed=int(raw["seed"]), stratum=int(raw["stratum"]))
        for checkpoint in CHECKPOINTS:
            for prefix in ("reconstruction_norm", "weighted_aux_norm", "ratio"):
                key = prefix + "_" + checkpoint
                row[key] = float(row[key])
            close(row["ratio_" + checkpoint], norm_ratio(row["reconstruction_norm_" + checkpoint],
                  row["weighted_aux_norm_" + checkpoint]), "Stored norm/ratio mismatch")
        key = row["seed"], row["batch_id"]
        require(key not in result, "Duplicate batch")
        result[key] = row
    return result


def panel_counts(panel, batches):
    seed = int(panel["seed"])
    rows = [batches[seed, key] for key in panel["batch_ids"].split(";")]
    return [sum(r["stratum"] == s for r in rows) for s in range(4)]


def validate_design(batches, panels):
    require(len(batches) == 5120 and {r["seed"] for r in batches.values()} == set(range(10)),
            "Ten complete independent seed bundles required")
    for seed in range(10):
        for pool, quota in (("reference_a", 32), ("reference_b", 32), ("estimation", 64)):
            rows = [r for r in batches.values() if r["seed"] == seed and r["pool"] == pool]
            require(len(rows) == quota * 4 and all(sum(r["stratum"] == s for r in rows) == quota
                                                  for s in range(4)), "Pool/degree-stratum quota mismatch")
    expected = {(seed, method, trial) for seed in range(10) for method in METHODS for trial in range(100)}
    observed = set()
    for panel in panels:
        seed, method, trial = int(panel["seed"]), panel["method"], int(panel["trial"])
        key = seed, method, trial
        require(key in expected and key not in observed, "Duplicate or unexpected panel")
        observed.add(key)
        ids = panel["batch_ids"].split(";")
        require(len(ids) == len(set(ids)) == 16, "Sixteen unique batches per panel required")
        require(all((seed, batch) in batches and batches[seed, batch]["pool"] == "estimation"
                    for batch in ids), "Panel leaks into a reference pool")
        if method == "stratified":
            require(panel_counts(panel, batches) == [4] * 4, "Stratified quotas violated")
    require(observed == expected, "Missing panel or seed; no survivor-only analysis")


def evaluate_panel(panel, batches, reference, interval=None):
    seed, method = int(panel["seed"]), panel["method"]
    rows = [batches[seed, key] for key in panel["batch_ids"].split(";")]
    require(len(rows) == 16 and len({r["batch_id"] for r in rows}) == 16, "Invalid panel")
    values = [r["ratio_" + cp] for cp in CHECKPOINTS for r in rows]
    require(all(math.isfinite(v) and v > 0 for v in values) and reference > 0
            and math.isfinite(reference), "Finite-positive checks cannot be ablated")
    estimate = statistics.median(values)
    error = (estimate - reference) / reference
    result = dict(seed=seed, method=method, trial=int(panel["trial"]), estimate=estimate,
                  signed_relative_error=error, absolute_relative_error=abs(error),
                  acceptable_relative_error=abs(error) <= .20)
    for cp in CHECKPOINTS:
        result[cp + "_estimate"] = statistics.median(r["ratio_" + cp] for r in rows)
    if method == "uniform":
        require(interval is None, "Uniform receives no conditional interval or refusal screen")
        return result
    require(method == "stratified" and interval is not None, "Missing Stratified stored interval")
    require(panel_counts(panel, batches) == [4] * 4, "Stratified input domain violated")
    lo, hi = float(interval["lambda_lower"]), float(interval["lambda_upper"])
    rlo, rhi = float(interval["R_lower"]), float(interval["R_upper"])
    require(all(math.isfinite(v) and v > 0 for v in (lo, hi, rlo, rhi)) and lo <= hi and rlo <= rhi,
            "Invalid stored conditional interval")
    mapping = {}
    for s in range(4):
        ordered = sorted(r["batch_id"] for r in rows if r["stratum"] == s)
        for index, key in enumerate(ordered):
            mapping[key] = index % 2
    lambdas = [10. / statistics.median(r["ratio_" + cp] for cp in CHECKPOINTS for r in rows
                                      if mapping[r["batch_id"]] == p) for p in (0, 1)]
    checkpoints = [result[cp + "_estimate"] for cp in CHECKPOINTS]
    checks = dict(CI=hi / lo <= 1.5, PANEL=max(lambdas) / min(lambdas) <= 1.25,
                  CHECKPOINT=max(checkpoints) / min(checkpoints) <= 2.)
    for name, value in checks.items():
        require(value == (interval["expected_" + name] == "True"), "Frozen refusal check mismatch")
    accepted = {ablation: all(passed for name, passed in checks.items()
                             if ablation not in ("NONE_OF_THREE", "DROP_" + name))
                for ablation in ABLATIONS}
    result.update(checks=checks, accepted=accepted, R_ci=[rlo, rhi], lambda_ci=[lo, hi],
                  apparent_reference_point_coverage=rlo <= reference <= rhi)
    return result


def bundle_metrics(panels):
    result = {}
    for method in METHODS:
        rows = panels[method]
        require(len(rows) == 100, "Exactly 100 repeated panels per method/seed required")
        result[method + "_mae"] = statistics.mean(r["absolute_relative_error"] for r in rows)
        result[method + "_acceptable_fraction"] = statistics.mean(r["acceptable_relative_error"] for r in rows)
    result["mae_difference_stratified_minus_uniform"] = result["stratified_mae"] - result["uniform_mae"]
    result["acceptable_fraction_difference"] = result["stratified_acceptable_fraction"] - result["uniform_acceptable_fraction"]
    result["historical_apparent_reference_point_coverage"] = statistics.mean(
        r["apparent_reference_point_coverage"] for r in panels["stratified"])
    for ablation in ABLATIONS:
        selected = [r for r in panels["stratified"] if r["accepted"][ablation]]
        result[ablation + "_acceptance_fraction"] = len(selected) / 100
        result[ablation + "_accepted_mae"] = statistics.mean(r["absolute_relative_error"] for r in selected) if selected else None
    return result


def reproduce(repository_root, output_dir):
    """Verify scalar arithmetic and write NEW tables, never bootstrap or overwrite.

    The execution disposition remains failed even when numerical assertions pass.
    Stored intervals are checked for integrity/domain and used as frozen inputs;
    their resampling distributions and RNG histories are not independently rerun.
    """
    root, out = Path(repository_root).resolve(), Path(output_dir).resolve()
    data = root / "processed_records/estimator_reliability"
    require(not out.is_relative_to(root / "processed_records"), "Cannot write into canonical inputs")
    if out.exists():
        raise FileExistsError("Use a new output directory; existing results are never overwritten")
    frozen = json.loads((data / "frozen_summary.json").read_text(encoding="utf-8"))
    require(set(frozen["input_sha256"]) == {"batch_scalars.csv", "panels.csv",
            "reference_intervals.csv", "stratified_intervals.csv"}, "Unexpected scalar input paths")
    for name, expected_sha in frozen["input_sha256"].items():
        require(hashlib.sha256((data / name).read_bytes()).hexdigest() == expected_sha,
                "Scalar input digest mismatch: " + name)
    require(frozen["data_disposition"] == "CONDITIONAL_ACCEPT_DATA" and
            frozen["execution_disposition"] == "FAIL_FORMAL_EXECUTION_GATE" and
            frozen["terminal_rss_failure_timing"] == "UNRESOLVED", "Disposition must not be promoted")
    batches = read_batches(data / "batch_scalars.csv")
    panels = read_csv(data / "panels.csv")
    validate_design(batches, panels)
    intervals = read_csv(data / "stratified_intervals.csv")
    lookup = {(int(r["seed"]), int(r["trial"])): r for r in intervals}
    require(len(intervals) == len(lookup) == 1000 and set(lookup) ==
            {(s, t) for s in range(10) for t in range(100)}, "Missing Stratified interval")
    reference_inputs = read_csv(data / "reference_intervals.csv")
    require(len(reference_inputs) == 30, "Missing reference interval records")
    refs, reference_table = {}, []
    for seed in range(10):
        selected = [r for r in batches.values() if r["seed"] == seed and r["pool"] != "estimation"]
        refs[seed] = statistics.median(r["ratio_" + cp] for cp in CHECKPOINTS for r in selected)
        for cp in (*CHECKPOINTS, "pooled"):
            source = [r for r in reference_inputs if int(r["seed"]) == seed and r["checkpoint"] == cp]
            require(len(source) == 1, "Duplicate/missing reference state")
            source = source[0]
            point = refs[seed] if cp == "pooled" else statistics.median(r["ratio_" + cp] for r in selected)
            close(point, float(source["expected_reference"]), "Frozen reference median mismatch")
            discrepancy = None
            if cp != "pooled":
                ab = [statistics.median(r["ratio_" + cp] for r in selected if r["pool"] == role)
                      for role in ("reference_a", "reference_b")]
                close(ab[0], float(source["reference_a"]), "Reference A mismatch")
                close(ab[1], float(source["reference_b"]), "Reference B mismatch")
                discrepancy = abs(ab[0] - ab[1]) / statistics.median(ab)
                close(discrepancy, float(source["expected_discrepancy"]), "Reference discrepancy mismatch")
                require(discrepancy <= .10, "Reference unstable; withhold complete aggregate, never drop a seed")
            bounds = [float(source["R_lower"]), float(source["R_upper"])]
            require(all(math.isfinite(v) and v > 0 for v in bounds) and bounds[0] <= bounds[1],
                    "Invalid stored reference interval")
            reference_table.append(dict(seed=seed, checkpoint=cp, reference=point,
                                        discrepancy=discrepancy, interval_origin="STORED_NOT_RESAMPLED",
                                        R_lower=bounds[0], R_upper=bounds[1],
                                        lambda_lower=source["lambda_lower"], lambda_upper=source["lambda_upper"]))
    evaluated, refusal_rows, seed_rows, state_rows, state_summaries = [], [], [], [], []
    state_references = {(r["seed"], r["checkpoint"]): r["reference"] for r in reference_table}
    bundles = []
    require(len(frozen["seed_bundles"]) == 10 and all(value["independent_units"] == 10
            for value in frozen["aggregate_metrics"].values()), "Frozen independent unit is ten seeds")
    for seed in range(10):
        groups = {method: [] for method in METHODS}
        for panel in [p for p in panels if int(p["seed"]) == seed]:
            interval = lookup[seed, int(panel["trial"])] if panel["method"] == "stratified" else None
            result = evaluate_panel(panel, batches, refs[seed], interval)
            for field in ("estimate", "early_estimate", "late_estimate", "signed_relative_error"):
                close(result[field], float(panel["expected_" + field]), "Frozen panel mismatch: " + field)
            for cp in CHECKPOINTS:
                point, reference = result[cp + "_estimate"], state_references[seed, cp]
                error = (point - reference) / reference
                state_rows.append(dict(seed=seed, method=result["method"], trial=result["trial"],
                    checkpoint=cp, estimate=point, reference=reference, signed_relative_error=error,
                    absolute_relative_error=abs(error), acceptable_relative_error=abs(error) <= .20))
            groups[result["method"]].append(result)
            evaluated.append({k: v for k, v in result.items() if k not in
                              ("checks", "accepted", "R_ci", "lambda_ci", "apparent_reference_point_coverage")})
            if interval is not None:
                refusal_rows.append(dict(seed=seed, trial=result["trial"], **result["checks"],
                    **result["accepted"], bad_error=result["absolute_relative_error"] > .20,
                    apparent_reference_point_coverage=result["apparent_reference_point_coverage"]))
        bundle = bundle_metrics(groups)
        bundles.append(bundle)
        for field, expected in frozen["seed_bundles"][seed].items():
            if expected is None:
                require(bundle[field] is None, "Undefined seed metric must not be imputed")
            else:
                close(bundle[field], expected, "Frozen seed summary mismatch: " + field)
        for method in METHODS:
            values = [r["estimate"] for r in groups[method]]
            seed_rows.append(dict(seed=seed, method=method, trials=100, reference=refs[seed],
                mae=bundle[method + "_mae"], acceptable_error_fraction=bundle[method + "_acceptable_fraction"],
                estimate_sample_SD=statistics.stdev(values), estimate_IQR=percentile(values, .75) - percentile(values, .25)))
            for cp in CHECKPOINTS:
                state = [r for r in state_rows if r["seed"] == seed and r["method"] == method and r["checkpoint"] == cp]
                values = [r["estimate"] for r in state]
                require(len(state) == 100, "Missing statewise panel")
                state_summaries.append(dict(seed=seed, method=method, checkpoint=cp, trials=100,
                    mean_signed_relative_error=statistics.mean(r["signed_relative_error"] for r in state),
                    mae=statistics.mean(r["absolute_relative_error"] for r in state),
                    acceptable_error_fraction=statistics.mean(r["acceptable_relative_error"] for r in state),
                    estimate_sample_SD=statistics.stdev(values), estimate_IQR=percentile(values, .75) - percentile(values, .25)))
    aggregate = {field: (statistics.mean(b[field] for b in bundles)
                        if all(b[field] is not None for b in bundles) else None) for field in bundles[0]}
    for field, record in frozen["aggregate_metrics"].items():
        close(aggregate[field], record["point"], "Frozen aggregate mismatch: " + field)
    ablation_table = []
    refusal_lookup = {(r["seed"], r["trial"]): r for r in refusal_rows}
    for ablation in ABLATIONS:
        selected = [r for r in refusal_rows if r[ablation]]
        selected_errors = [[r["absolute_relative_error"] for r in evaluated if r["seed"] == seed
                            and r["method"] == "stratified" and
                            refusal_lookup[seed, r["trial"]][ablation]]
                           for seed in range(10)]
        ablation_table.append(dict(ablation=ablation, retained=len(selected),
            bad_retained=sum(r["bad_error"] for r in selected),
            seed_equal_retained_mae=seed_equal_mean(selected_errors),
            pooled_retained_mae=pooled_mean(selected_errors)))
    all_row = ablation_table[0]
    metrics = dict(independent_seed_count=10, panel_count=len(panels), scalar_measurement_count=len(batches) * 2,
        stratified_mae_percent=100 * aggregate["stratified_mae"],
        uniform_mae_percent=100 * aggregate["uniform_mae"],
        paired_mae_difference_pp=100 * aggregate["mae_difference_stratified_minus_uniform"],
        all_retained_count=all_row["retained"], all_bad_retained_count=all_row["bad_retained"],
        no_refusal_bad_count=ablation_table[-1]["bad_retained"],
        all_retention_percent=100 * all_row["retained"] / 1000,
        all_bad_retained_percent=100 * all_row["bad_retained"] / all_row["retained"],
        no_refusal_bad_percent=100 * ablation_table[-1]["bad_retained"] / 1000,
        refusal_gate_failures={name: sum(not r[name] for r in refusal_rows) for name in ("CI", "PANEL", "CHECKPOINT")},
        all_retained_mae_seed_equal_percent=100 * all_row["seed_equal_retained_mae"],
        all_retained_mae_pooled_percent=100 * all_row["pooled_retained_mae"],
        apparent_reference_point_coverage_percent=100 * aggregate["historical_apparent_reference_point_coverage"],
        stored_paired_mae_difference_ci_pp=[100 * v for v in frozen["aggregate_metrics"]["mae_difference_stratified_minus_uniform"]["seed_cluster_ci95"]],
        stored_reference_point_coverage_ci_percent=[100 * v for v in frozen["aggregate_metrics"]["historical_apparent_reference_point_coverage"]["seed_cluster_ci95"]],
        reference_max_discrepancy=max(r["discrepancy"] for r in reference_table if r["discrepancy"] is not None),
        data_disposition=frozen["data_disposition"], execution_disposition=frozen["execution_disposition"],
        terminal_rss_failure_timing=frozen["terminal_rss_failure_timing"],
        interval_origin="STORED_NOT_RESAMPLED", new_bootstrap_resamples=0,
        bootstrap_rng_history_independently_verified=False, population_coverage_certified=False)
    require(metrics["refusal_gate_failures"] == frozen["gate_failures"], "Frozen refusal counts mismatch")
    for name, expected in frozen["verification_targets"].items():
        close(metrics[name], expected, "Independent frozen check mismatch: " + name)
    for row in ablation_table:
        expected = frozen["ablation_counts"][row["ablation"]]
        require(row["retained"] == expected["accepted"] and row["bad_retained"] == expected["bad_accepted"],
                "Independent ablation count mismatch")
    stored_intervals = [dict(metric=key, point=aggregate[key], lower=value["seed_cluster_ci95"][0],
        upper=value["seed_cluster_ci95"][1], independent_seed_count=10,
        interval_origin="STORED_NOT_RESAMPLED", interval_target="seed_aggregate_metric_not_ratio")
        for key, value in frozen["aggregate_metrics"].items()]
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("reference_checks.csv", reference_table), ("sampling_panels.csv", evaluated),
                       ("stratified_refusal.csv", refusal_rows), ("seed_summary.csv", seed_rows),
                       ("statewise_sampling_errors.csv", state_rows), ("statewise_summary.csv", state_summaries),
                       ("refusal_ablations.csv", ablation_table), ("stored_intervals.csv", stored_intervals)):
        write_csv(out / name, rows)
    with (out / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.repository_root, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
