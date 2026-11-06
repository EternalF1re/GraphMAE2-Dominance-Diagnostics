"""CPU-only reproduction of the four added analyses and Figures1/5."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("released_"+name, ROOT/"code/analysis"/(name+".py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def number_at(result, path):
    value = result
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return float(value)


def run(output, figures=True):
    output = Path(output).resolve()
    for protected in [ROOT/"processed_records", ROOT/"code", ROOT/"figures", ROOT/"tests", ROOT/"experiments"]:
        if output == protected or protected in output.parents: raise ValueError("Output would overlap canonical evidence")
    if output == ROOT or output in ROOT.parents: raise ValueError("Unsafe output root")
    output.mkdir(parents=True, exist_ok=False)
    result = {name: load(name).reproduce(ROOT, output/name)
              for name in ["factorial", "gradient_geometry", "maskgae_pubmed", "estimator_reliability"]}
    factorial_records = load("factorial").read_records(ROOT)
    result["coverage"] = dict(factorial_records=len(factorial_records),
        factorial_pretraining_seeds=len({r["pretraining_seed"] for r in factorial_records}),
        factorial_checkpoints=len({r["global_update"] for r in factorial_records}))
    result["legacy"] = {name:float(read_csv(ROOT/path)[0]["dominance_ratio"]) for name,path in [
        ("arxiv_reference", "processed_records/arxiv/reference_summary.csv"),
        ("reddit_reference", "processed_records/reddit/reference_summary.csv")]}
    histogram = read_csv(ROOT/"processed_records/method_comparisons/pcgrad_conflict_histogram.csv")
    result["legacy"]["pcgrad_updates"] = sum(int(r["update_count"]) for r in histogram)
    result["legacy"]["pcgrad_conflicts"] = sum(int(r["update_count"]) for r in histogram if r["region"] == "conflict")
    result["factorial_families"] = {}
    for row in read_csv(output/"factorial/factorial_effects.csv"):
        result["factorial_families"].setdefault(row["protocol"], {}).setdefault(row["endpoint"], {})[row["effect"]] = row
    claim_rows = []
    specs = json.loads((ROOT/"protocols/revision_claims.json").read_text(encoding="utf-8"))
    for spec in specs:
        actual = number_at(result, spec["metric_path"])*spec.get("scale", 1)
        passed = math.isfinite(actual) and abs(actual-spec["expected_value"]) <= spec["absolute_tolerance"]
        claim_rows.append(dict(manuscript_item=spec["manuscript_item"], claim=spec["claim"],
            reported_value=spec["reported_value"], unit=spec["unit"], public_data=spec["public_data"],
            calculation_script=spec["calculation_script"], source_id=spec["source_id"],
            metric_path=spec["metric_path"], recomputed_value=actual, expected_value=spec["expected_value"],
            absolute_tolerance=spec["absolute_tolerance"], consistency="PASS" if passed else "MISMATCH"))
    with (output/"MANUSCRIPT_CLAIM_MAP.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(claim_rows[0])); writer.writeheader(); writer.writerows(claim_rows)
    with (output/"analysis_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    mismatches = [r for r in claim_rows if r["consistency"] != "PASS"]
    if mismatches: raise ValueError("Scientific claim mismatches retained in output: " + str(mismatches))
    if figures:
        for name, script in [("figure_1", "plot_factorial_trajectory.py"), ("figure_5", "plot_maskgae_dose_response.py")]:
            subprocess.run([sys.executable, "-B", str(ROOT/"figures"/name/script),
                            "--output", str(output/"figures"/(name+".pdf"))], check=True)
    print(json.dumps(dict(status="ANALYSIS_REPRODUCTION_PASS", verified_claims=len(claim_rows),
                         model_calls=0, new_training_calls=0, new_probes=0,
                         new_bootstrap_resamples=0, output=str(output)), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"reproduced/revision_analysis", help="A new output directory")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, figures=not args.skip_figures)


if __name__ == "__main__": main()
