"""Recreate descriptive geometry tables from scalar records using only stdlib.

Study-authored analysis; distributed under LICENSE-AUTHORS. This module never
imports a model, reads a checkpoint, measures a gradient, or resamples data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


GROUPS = ("G1", "G2")
UPDATES = (250, 3555, 8532)
POSITIONS = (93, 118)
COEFFICIENTS = {"G1": (10.0, 10.0), "G2": (0.3, 10.0)}
DATA_DIRECTORY = Path("processed_records/gradient_geometry")
PAIRED_IDENTITY_FIELDS = (
    "batch_hash", "root_ids_sha256", "frozen_rng_sha256", "pre_forward_rng_sha256",
    "post_forward_rng_sha256", "mask_sha256", "remask_sha256", "data_sha256",
    "data_fingerprints",
)
BOOL_FIELDS = {
    "defined", "target_defined", "preclip_only", "parameter_gradients_unchanged",
    "vjp_purity", "model_teacher_unchanged", "paired_stochastic_identity",
}
INT_FIELDS = {
    "seed", "update", "batch_position", "full_z_rows", "hidden_dimension",
    "target_count", "epoch", "ema_updates", "fullZ_reconstruction_nonzero_rows",
    "fullZ_aux_nonzero_rows", "fullZ_shared_nonzero_rows",
    "target_reconstruction_nonzero_rows", "target_aux_nonzero_rows",
    "target_shared_nonzero_rows",
}
MEASURED_METRICS = {
    "local_ratio": "weighted_ratio",
    "full_z_cosine": "cosine",
    "full_z_angle_degrees": "angle_degrees",
    "full_z_reconstruction_norm": "reconstruction_gradient_norm",
    "full_z_auxiliary_norm_local": "weighted_aux_gradient_norm",
    "target_cosine": "shared_target_cosine",
    "target_reconstruction_norm": "target_reconstruction_gradient_norm",
    "target_auxiliary_norm_local": "target_weighted_aux_gradient_norm",
    "target_reconstruction_full_z_norm_fraction": "target_reconstruction_fullZ_norm_fraction",
    "target_auxiliary_full_z_norm_fraction": "target_aux_fullZ_norm_fraction",
    "full_z_reconstruction_nonzero_row_fraction": "reconstruction_nonzero_row_fraction",
    "full_z_auxiliary_nonzero_row_fraction": "auxiliary_nonzero_row_fraction",
    "full_z_shared_nonzero_row_fraction": "shared_nonzero_row_fraction",
    "target_reconstruction_nonzero_rows": "target_reconstruction_nonzero_rows",
    "target_auxiliary_nonzero_rows": "target_aux_nonzero_rows",
    "target_shared_nonzero_rows": "target_shared_nonzero_rows",
    "reconstruction_loss": "loss_reconstruction",
    "bare_auxiliary_loss": "loss_latent_bare",
}
METRICS = tuple(MEASURED_METRICS) + ("routed_ratio_derived", "target_local_ratio_derived")


def record_key(row):
    return row["group"], row["seed"], row["update"], row["batch_position"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def validate_records(rows):
    expected = {(g, s, u, b) for g in GROUPS for s in range(10)
                for u in UPDATES for b in POSITIONS}
    require(len(rows) == 120 and {record_key(r) for r in rows} == expected,
            "Expected exactly the complete 120-record measurement grid")
    for row in rows:
        require(all(row.get(k) is True for k in BOOL_FIELDS),
                "Undefined or non-pure measurements cannot form a complete-design summary")
        require(all(type(v) in (int, float) and math.isfinite(v)
                    for k, v in row.items() if k != "group" and k not in BOOL_FIELDS),
                "All recorded numeric values must be finite")
        encoder, head = COEFFICIENTS[row["group"]]
        require(row["model_effective_lambda"] == 10.0
                and row["training_shared_coefficient"] == encoder
                and row["training_head_coefficient"] == head, "Unexpected group coefficients")
        require(row["target_count"] == 128 and row["full_z_rows"] >= 128
                and row["hidden_dimension"] == 1024, "Unexpected representation shape")
        for field in ("reconstruction_gradient_norm", "weighted_aux_gradient_norm",
                      "target_reconstruction_gradient_norm", "target_weighted_aux_gradient_norm"):
            require(row[field] > 0, "Zero component norm: complete summary unavailable")
        local = row["weighted_aux_gradient_norm"] / row["reconstruction_gradient_norm"]
        require(close(row["weighted_ratio"], local), "Local ratio must use its own measured norms")
        require(close(row["routed_scale_derived"], encoder / head)
                and close(row["routed_weighted_ratio_derived"], local * encoder / head),
                "Routed ratio must be the group's own local ratio times E/H")
        require(-1 <= row["cosine"] <= 1 and -1 <= row["shared_target_cosine"] <= 1,
                "Cosine outside its finite domain")
        require(close(row["angle_degrees"], math.degrees(math.acos(row["cosine"]))),
                "Angle must correspond to full-Z cosine")
        for prefix, limit in (("fullZ", row["full_z_rows"]), ("target", 128)):
            for suffix in ("reconstruction_nonzero_rows", "aux_nonzero_rows", "shared_nonzero_rows"):
                require(0 <= row[prefix + "_" + suffix] <= limit, "Invalid support count")
        for fraction, count in (
            ("reconstruction_nonzero_row_fraction", "fullZ_reconstruction_nonzero_rows"),
            ("auxiliary_nonzero_row_fraction", "fullZ_aux_nonzero_rows"),
            ("shared_nonzero_row_fraction", "fullZ_shared_nonzero_rows"),
        ):
            require(close(row[fraction], row[count] / row["full_z_rows"]), "Support fraction mismatch")
        require(close(row["target_reconstruction_fullZ_norm_fraction"],
                      row["target_reconstruction_gradient_norm"] / row["reconstruction_gradient_norm"])
                and close(row["target_aux_fullZ_norm_fraction"],
                          row["target_weighted_aux_gradient_norm"] / row["weighted_aux_gradient_norm"]),
                "Target/full-Z norm fraction mismatch")
    return rows


def _hex_hash(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def load_records(repository_root):
    """Read only the three public scalar/identity metadata files."""
    directory = Path(repository_root) / DATA_DIRECTORY
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    require(provenance["record_count"] == 120, "Unexpected record count")
    require(set(provenance["scalar_record_files"]) == {"measurements.csv", "identities.json"},
            "Unexpected scalar input files")
    for name, expected in provenance["scalar_record_files"].items():
        require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected,
                "Processed record hash mismatch: " + name)
    with (directory / "measurements.csv").open(encoding="utf-8", newline="") as stream:
        rows = []
        for raw in csv.DictReader(stream):
            row = {}
            for name, value in raw.items():
                if name == "group":
                    row[name] = value
                elif name in BOOL_FIELDS:
                    require(value in ("true", "false"), "Invalid boolean field")
                    row[name] = value == "true"
                else:
                    row[name] = int(value) if name in INT_FIELDS else float(value)
            rows.append(row)
    validate_records(rows)
    identities = json.loads((directory / "identities.json").read_text(encoding="utf-8"))["identities"]
    require(len(identities) == 120 and {record_key(r) for r in identities}
            == {record_key(r) for r in rows}, "Diagnostic identity grid mismatch")
    index = {record_key(r): r for r in identities}
    for row in identities:
        require(all(_hex_hash(v) for k, v in row.items()
                    if k.endswith("_sha256") and k != "remask_sha256"), "Invalid identity hash")
        require(len(row["remask_sha256"]) == 3 and all(_hex_hash(h) for h in row["remask_sha256"]),
                "Expected three remask identity hashes")
    for seed in range(10):
        for update in UPDATES:
            for position in POSITIONS:
                a, b = (index[(g, seed, update, position)] for g in GROUPS)
                require(all(a[k] == b[k] for k in PAIRED_IDENTITY_FIELDS),
                        "Paired stochastic diagnostic identities differ")
    return rows


def _record_metrics(row):
    values = {name: row[field] for name, field in MEASURED_METRICS.items()}
    encoder, head = COEFFICIENTS[row["group"]]
    values["routed_ratio_derived"] = row["weighted_ratio"] * (encoder / head)
    values["target_local_ratio_derived"] = (
        row["target_weighted_aux_gradient_norm"] / row["target_reconstruction_gradient_norm"])
    return values


def _means(rows):
    return {metric: statistics.mean(r[metric] for r in rows) for metric in METRICS}


def aggregate(records):
    """Two batches -> three checkpoints -> ten equally weighted seed means.

    Undefined or incomplete records raise; no complete-design mean silently
    drops a batch, checkpoint or seed. SD is across seed means with ddof=1.
    """
    validate_records(records)
    units = [dict(group=r["group"], seed=r["seed"], update=r["update"],
                  batch_position=r["batch_position"], **_record_metrics(r)) for r in records]
    checkpoints = []
    seeds = []
    groups = {}
    for group in GROUPS:
        for seed in range(10):
            for update in UPDATES:
                pair = [r for r in units if (r["group"], r["seed"], r["update"])
                        == (group, seed, update)]
                checkpoints.append(dict(group=group, seed=seed, update=update, batches=2, **_means(pair)))
            triple = [r for r in checkpoints if (r["group"], r["seed"]) == (group, seed)]
            seeds.append(dict(group=group, seed=seed, checkpoints=3, **_means(triple)))
        ten = [r for r in seeds if r["group"] == group]
        groups[group] = {metric: dict(mean=statistics.mean(r[metric] for r in ten),
                                     sample_sd=statistics.stdev(r[metric] for r in ten))
                         for metric in METRICS}
    differences = []
    for seed in range(10):
        a, b = (next(r for r in seeds if r["group"] == g and r["seed"] == seed) for g in GROUPS)
        differences.append(dict(seed=seed, **{m + "_g2_minus_g1": b[m] - a[m] for m in METRICS}))
    return dict(measurement_count=120, independent_seed_count=10,
                aggregation="two batch means, then three checkpoint means, then ten seed means",
                statistic="descriptive mean and sample SD across seeds; no bootstrap, CI, or p-value",
                geometry_scope="representation gradients, not parameter gradients; preclip only",
                target_scope="all 128 ordered target rows retained, including zero rows",
                interpretation="Near-zero aggregate cosines do not establish per-node orthogonality or causality.",
                groups=groups, per_checkpoint=checkpoints, per_seed=seeds,
                paired_seed_differences=differences,
                routed_group_mean_ratio_g1_over_g2=(groups["G1"]["routed_ratio_derived"]["mean"]
                                                    / groups["G2"]["routed_ratio_derived"]["mean"]))


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reproduce(repository_root, output_dir):
    """Recreate tables in output_dir without any scientific dependencies."""
    root, output = Path(repository_root).resolve(), Path(output_dir).resolve()
    require(output != root and not any(output == root / name or (root / name) in output.parents
                                      for name in ("processed_records", "code", "tests")),
            "Output must not overwrite source or processed records")
    result = aggregate(load_records(root))
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_checkpoint.csv", result["per_checkpoint"])
    _write_csv(output / "per_seed.csv", result["per_seed"])
    _write_csv(output / "paired_seed_differences.csv", result["paired_seed_differences"])
    group_rows = [dict(group=g, independent_seeds=10,
                       **{m + "_" + stat: value for m, stats in result["groups"][g].items()
                          for stat, value in stats.items()}) for g in GROUPS]
    _write_csv(output / "group_summary.csv", group_rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.repository_root / "experiments/gradient_geometry/reproduced"
    result = reproduce(args.repository_root, output)
    print(json.dumps({"measurement_count": result["measurement_count"], "groups": result["groups"]},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
