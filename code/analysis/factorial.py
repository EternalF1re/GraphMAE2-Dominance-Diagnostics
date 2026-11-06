"""Ten matched-seed GraphMAE2 factorial analyses from scalar records only."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics as S

from scipy.stats import t

UPDATES = (250, 711, 1422, 2133, 3555, 5688, 8532)
GROUPS = {"G1": (10., 10.), "G2": (.3, 10.), "G3": (.3, .3), "G4": (10., .3)}
PROBES = (54321, 54322, 54323)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate_records(rows):
    clean = []
    for original in rows:
        row = dict(original)
        for key in ["pretraining_seed", "global_update"]: row[key] = int(row[key])
        for key in ["E", "H", "validation_accuracy", "validation_accuracy_pct"] + [f"probe_{s}" for s in PROBES]:
            row[key] = float(row[key])
            if not math.isfinite(row[key]): raise ValueError("Nonfinite factorial value")
        if row["group"] not in GROUPS or (row["E"], row["H"]) != GROUPS[row["group"]]:
            raise ValueError("Factorial group/factor mismatch")
        if not 0 <= row["validation_accuracy"] <= 1: raise ValueError("Accuracy fraction out of range")
        if not math.isclose(row["validation_accuracy_pct"], 100*row["validation_accuracy"], rel_tol=0, abs_tol=1e-10):
            raise ValueError("Accuracy fraction/percentage mismatch")
        probes = [row[f"probe_{s}"] for s in PROBES]
        if not all(0 <= p <= 1 for p in probes) or not math.isclose(S.mean(probes), row["validation_accuracy"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Frozen three-probe mean mismatch")
        clean.append(row)
    expected = {(p, g, s, u) for p in "AB" for g in GROUPS for s in range(10) for u in UPDATES}
    keys = [(r["protocol"], r["group"], r["pretraining_seed"], r["global_update"]) for r in clean]
    if len(keys) != 560 or set(keys) != expected: raise ValueError("Exactly80 matched runs x7 checkpoints required")
    return clean


def read_records(root):
    return validate_records(read_csv(Path(root)/"processed_records/factorial/checkpoint_validation.csv"))


def factorial_effects(g):
    a, b, c, d = (g[name] for name in GROUPS)
    return dict(encoder=((a-b)+(d-c))/2, head=((a-d)+(b-c))/2, interaction=(a-b)-(d-c))


def paired_summary(values):
    if len(values) != 10 or not all(math.isfinite(v) for v in values): raise ValueError("Ten finite matched seeds required")
    mean, sd = S.mean(values), S.stdev(values)
    se = sd/math.sqrt(10)
    critical = float(t.ppf(1-.05/6, 9))
    ordinary = float(t.ppf(.975, 9))
    return dict(n=10, df=9, estimate_pp=mean, sd_pp=sd, se_pp=se,
                ci95_simultaneous_lower_pp=mean-critical*se, ci95_simultaneous_upper_pp=mean+critical*se,
                ci95_nominal_lower_pp=mean-ordinary*se, ci95_nominal_upper_pp=mean+ordinary*se,
                p_unadjusted=float(2*t.sf(abs(mean/se), 9)) if se else None,
                interval_family="three effects within this protocol and endpoint; Bonferroni simultaneous95%")


def holm3(values):
    if len(values) != 3: raise ValueError("Three-effect family required")
    if any(p is None for p in values): return [None]*3
    result, previous = [None]*3, 0.
    for rank, index in enumerate(sorted(range(3), key=lambda i: values[i])):
        previous = min(1., max(previous, (3-rank)*values[index])); result[index] = previous
    return result


def trajectory_auc(points):
    if set(points) != set(UPDATES): raise ValueError("Seven actual optimizer checkpoints required")
    return math.fsum((points[a]+points[b])*.5*(b-a) for a, b in zip(UPDATES, UPDATES[1:]))/8282


def checkpoint_summary(rows):
    return [dict(protocol=p, group=g, E=GROUPS[g][0], H=GROUPS[g][1], global_update=u,
                 n_pretraining_seeds=10,
                 mean_accuracy_pct=100*S.mean(r["validation_accuracy"] for r in rows
                      if (r["protocol"], r["group"], r["global_update"]) == (p, g, u)),
                 sd_accuracy_pp=100*S.stdev(r["validation_accuracy"] for r in rows
                      if (r["protocol"], r["group"], r["global_update"]) == (p, g, u)))
            for p in "AB" for g in GROUPS for u in UPDATES]


def analyze(rows, endpoint):
    if endpoint not in ["auc", "final8532"]: raise ValueError("Unknown endpoint")
    lookup = {(r["protocol"], r["group"], r["pretraining_seed"], r["global_update"]): r["validation_accuracy"] for r in rows}
    values = {(p, g, s): trajectory_auc({u: lookup[p,g,s,u] for u in UPDATES}) if endpoint == "auc" else lookup[p,g,s,8532]
              for p in "AB" for g in GROUPS for s in range(10)}
    groups, results, seeds = [], [], []
    for p in "AB":
        for g, (encoder, head) in GROUPS.items():
            sample = [100*values[p,g,s] for s in range(10)]
            groups.append(dict(endpoint=endpoint, protocol=p, group=g, E=encoder, H=head, n=10,
                               mean_accuracy_pct=S.mean(sample), sample_sd_pp=S.stdev(sample)))
        per_seed = [dict(endpoint=endpoint, protocol=p, pretraining_seed=s,
                         **factorial_effects({g: 100*values[p,g,s] for g in GROUPS})) for s in range(10)]
        seeds.extend(per_seed)
        family = [dict(endpoint=endpoint, protocol=p, effect=name, **paired_summary([r[name] for r in per_seed]))
                  for name in ["encoder", "head", "interaction"]]
        for row, adjusted in zip(family, holm3([r["p_unadjusted"] for r in family])):
            results.append(dict(row, p_holm3=adjusted))
    return groups, results, seeds


def reproduce(repository_root, output_dir):
    root, output = Path(repository_root), Path(output_dir)
    rows = read_records(root)
    output.mkdir(parents=True, exist_ok=True)
    groups, estimates, seeds, metrics = [], [], [], {}
    for endpoint in ["auc", "final8532"]:
        gs, es, ss = analyze(rows, endpoint)
        groups.extend(gs); estimates.extend(es); seeds.extend(ss)
        for row in es:
            metrics[f"factorial_{row['protocol']}_{endpoint}_{row['effect']}_pp"] = row["estimate_pp"]
    frozen = read_csv(root/"processed_records/factorial/frozen_auc_effects.csv")
    by_key = {(r["protocol"], r["effect"]): r for r in estimates if r["endpoint"] == "auc"}
    for row in frozen:
        for key in ["estimate_pp", "ci95_simultaneous_lower_pp", "ci95_simultaneous_upper_pp"]:
            if not math.isclose(by_key[row["protocol"],row["effect"]][key], float(row[key]), rel_tol=0, abs_tol=1e-10):
                raise ValueError("Frozen AUC effect or simultaneous interval mismatch")
    source_auc = {(r["protocol"],r["group"],int(r["pretraining_seed"])):float(r["validation_auc"])
                  for r in read_csv(root/"processed_records/factorial/frozen_seed_auc.csv")}
    auc_rows = []
    for p in "AB":
        for g in GROUPS:
            for s in range(10):
                value = trajectory_auc({r["global_update"]:r["validation_accuracy"] for r in rows
                                        if (r["protocol"],r["group"],r["pretraining_seed"]) == (p,g,s)})
                if not math.isclose(value, source_auc[p,g,s], rel_tol=0, abs_tol=1e-12): raise ValueError("Frozen seed AUC mismatch")
                auc_rows.append(dict(protocol=p,group=g,pretraining_seed=s,validation_auc=value,validation_auc_pct=100*value))
    write_csv(output/"factorial_group_summary.csv", groups)
    write_csv(output/"factorial_effects.csv", estimates)
    write_csv(output/"factorial_per_seed_effects.csv", seeds)
    write_csv(output/"factorial_seed_auc.csv", auc_rows)
    write_csv(output/"factorial_checkpoint_summary.csv", checkpoint_summary(rows))
    write_csv(output/"factorial_final_seed_accuracy.csv", [r for r in rows if r["global_update"] == 8532])
    return metrics
