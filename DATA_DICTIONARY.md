# Public Data Dictionary

## Statistical units and numerical units

Pretraining seed means one independently initialized training trajectory.
Linear-probe seeds54321/54322/54323 are three classifier restarts nested within
a checkpoint; they are averaged first, not counted as30 independent trajectories.
Checkpoints and diagnostic panels are repeated measurements, not new seeds.

- Accuracy fraction: bounded0..1, e.g.0.712874.
- Accuracy percentage:100times fraction, e.g.71.2874%.
- Effect in percentage points: difference of percentages, e.g.-0.7018pp.
- Gradient norm ratio and cosine: dimensionless, neither accuracy nor percent.
- MAE in the reliability study: mean absolute RELATIVE estimator error, not
  classification error. Its percentage and between-method pp difference are distinct.

## Factorial checkpoint records

`processed_records/factorial/checkpoint_validation.csv` has560rows:
2protocols x4groups x10matched pretraining seeds x7checkpoints.

| Column | Meaning and unit |
|---|---|
| protocol | A=global norm3 clip; B=three parameter-groupwise norm3 clips |
| group | G1/G2/G3/G4; E/H mapping in configuration.json |
| E, H | Encoder-side and auxiliary-head coefficients; dimensionless |
| pretraining_seed | Independent matched seed0..9 |
| global_update | Actual optimizer update250,711,1422,2133,3555,5688,8532 |
| validation_accuracy | Frozen mean of three probe accuracies, fraction |
| validation_accuracy_pct | Same stored accuracy multiplied by100, percent |
| probe_54321, probe_54322, probe_54323 | Individual best-validation top1 accuracies, fractions; fixed1000epoch probes |
| checkpoint_sha256 | SHA256 identity of the evaluated checkpoint; checkpoint bytes excluded |
| validation_result_sha256 | SHA256 of the original validation-result JSON |

`frozen_seed_auc.csv` has80rows. protocol/group/pretraining_seed have the same
meaning; validation_auc is normalized trajectory area in fractional accuracy
units, completed_updates=8532, source_completion_sha256 binds the original
completed-run receipt. Its AUC uses actual update distances divided by8282.

`run_identities.csv` has80rows. protocol/group/E/H/pretraining_seed are as above;
schedule_sha256 and initialization_sha256 bind same-seed matching across all
eight cells; config_sha256 binds the original effective runtime configuration.
No raw Schedule or state tensor is included.

`frozen_auc_effects.csv` has6rows. protocol/effect identify each contrast;
estimate_pp and ci95_simultaneous_lower_pp/upper_pp are source-reference effects
in pp; n=10 and df=9. They are verification targets, not raw model outcomes.
`source_receipt.json` records archive identity, coverage and source-hash columns.

## Factorial derived tables

Under `tables/revision/factorial/`, endpoint is auc or final8532; effect is
encoder/head/interaction. mean_accuracy_pct is group mean percent;
sample_sd_pp/sd_accuracy_pp are sample SD across ten pretraining seeds, in pp.
estimate_pp, sd_pp, se_pp and all ci*_pp columns are percentage points.
The simultaneous family is three effects within each protocol and endpoint,
Bonferroni95%, not the ordinary95% columns or one combined12-effect family.
p_unadjusted and p_holm3 are two-sided matched-seed t-test p-values and their
within-family Holm adjustment. n counts pretraining seeds; df=n-1.
Per-seed encoder/head/interaction columns are matched contrasts in pp, never
effect slopes divided by coefficient-level distance. No half is applied to
the interaction. Intermediate trajectory rows preserve the checkpoint schema.

## Gradient geometry

See `processed_records/gradient_geometry/DATA_DICTIONARY.md` and
`processed_records/gradient_geometry/README.md` for every scalar and identity
field. Local full-Z ratios include head coefficientH=10. Routed ratios multiply
the SAME model state's local ratio by E/H; G2 uses0.03. Full-Z and target-row
cosines have different supports. Targets include all128rows, including zero
reconstruction-gradient rows. These are representation gradients, not parameter
gradients or measured AdamW displacements. Summaries average two batches within
checkpoint, three checkpoints within seed, then ten seeds equally.

## MaskGAE PubMed

See `processed_records/maskgae_pubmed/DATA_DICTIONARY.md` and its processed-record README
for checkpoint/probe/training-update/run-count columns and exact source protocol.
There are350checkpoints,1050nested probes,25000actual training updates and50runs.
Ratios summarize median(ten seed medians of seven checkpoints), not median of
all70 values. Trigger counts use preclip global gradient norm>1 divided by
actual completed optimizer updates. Error bars in Figure5 are sample SD.
Ordinary paired95% intervals are not simultaneous; the historicalalpha=.5
contrast and the three added exploratory doses are separate families.

## Estimator reliability

See `processed_records/estimator_reliability/DATA_DICTIONARY.md` and its experiment PROTOCOL.md for the
5120 paired-batch rows/10240scalar measurements,2000sampling panels, reference
pool labels, stored interval bounds and per-seed aggregation. Only stratified
panels have historical conditional refusal/coverage. Stored seed-cluster and
conditional bootstrap bounds are not newly resampled by the CPU entry.
The80.9% coverage concerns a finite/noisy reference, not population coverage.
The numerical records are conditionally retained; the original execution gate
failed and remains failed.

## Claim and provenance records

`MANUSCRIPT_CLAIM_MAP.csv` identifies manuscript_item/claim, reported_value/unit,
public_data/calculation_script/source_id, metric_path in analysis_summary.json,
recomputed_value/expected_value/absolute_tolerance, and consistency. All PASS
labels mean the stated arithmetic/rounding check only, not execution permission,
mechanism validity, statistical significance or end-to-end reproduction.
`protocols/revision_claims.json` is the machine-readable check specification.
`FIGURE_MAP.json` binds current figure semantics to relative script/input/PDF
paths; historical Figure1 has a separate, explicit path.

The baseline three-seed Arxiv/Reddit, methods and appendix records retain their
existing definitions and are not substituted into the new ten-seed protocols.
