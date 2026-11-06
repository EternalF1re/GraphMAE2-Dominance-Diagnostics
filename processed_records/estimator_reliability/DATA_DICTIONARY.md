# Estimator Reliability Data Dictionary

## Hierarchy and Conventions

The independent experimental units are **ten pretraining seeds**, numbered 0-9.
Each seed has two fixed checkpoints: `early` is update 250 and `late` update 8532.
The same batches are measured at both checkpoints. Trials, checkpoints, and batch
measurements are repeated observations, not additional independent seeds.

There are 5,120 distinct seed/batch identities and 10,240 scalar measurements.
Each seed has disjoint batch pools: reference A 128, reference B 128, estimation
256. Each pool spans four degree strata. There are 100 trials per seed/method,
2,000 paired-checkpoint panels total, and 4,000 statewise panels. A panel has 16
unique batches, reused at both checkpoints. Stratified quotas are four per stratum;
Uniform has no quota conditioning or redraw, including accidental balanced panels.

All CSV numbers are unrounded text representations of original binary64 scalars.
An error **fraction** of 0.077 is 7.7%. A paired error difference of 0.001 is 0.1
**percentage points**. Ratios and lambda coefficients are dimensionless, not
validation accuracies. Booleans in CSVs are `True`/`False`. Empty reference lambda
or discrepancy cells mean not provided/not applicable, never zero imputation.

`batch_id` is an opaque ordinal alias. Its within-seed ordering preserves original
batch-hash order, allowing the historical alternating within-stratum split without
publishing node IDs or original workspace provenance. Identical IDs across the
two checkpoints denote the same input batch. Root-array disjointness is reported
by the original processed receipts but is not independently verifiable from these
public aliases; batch-pool disjointness and quotas are directly checkable.

## Canonical Inputs

### source_record_map.csv

10,240 provenance-only rows; key `(seed, batch_id, checkpoint)`. This table does
not change the scalar inputs or any scientific computation. All member digests
refer to the original frozen archive with outer SHA-256
`7019a5a5113a4019f6c1222695879a56034b361b7e077b1e02589c59a670b082`.

| Column | Meaning / Units |
|---|---|
| `seed` | Independent pretraining-seed identifier 0-9; joins `batch_scalars.csv`. |
| `batch_id` | Public ordinal batch alias; joins `batch_scalars.csv`. |
| `checkpoint` | `early` or `late`, identifying the corresponding norm/ratio triplet. |
| `source_member_sha256` | SHA-256 of the original scalar-unit ZIP member's uncompressed bytes, not its filename, compressed bytes, or outer archive. |

Original member names contain internal identifiers, so this reader-facing map
uses hash-only member identity. To trace a row, first verify the archive's outer
digest, then find the member whose uncompressed bytes have the listed digest.
The archive's member manifest provides the same lookup. The matched scalar-unit
JSON contains the recorded target reconstruction norm, weighted auxiliary norm,
and ratio for this public record. No original batch hashes, node arrays, private
paths, or authorization-member paths are included in this map. A content digest
establishes record identity within the bound archive, not independent scientific
validation or an execution-pass disposition.

### batch_scalars.csv

5,120 rows; key `(seed, batch_id)`. Each row contains both checkpoint observations.

| Column | Meaning / Units |
|---|---|
| `seed` | Independent pretraining-seed identifier, integer 0-9. |
| `batch_id` | Opaque seed/batch identifier, shared across checkpoints. |
| `pool` | `reference_a`, `reference_b`, or `estimation`; pools do not share batch IDs. |
| `stratum` | Degree-stratum identifier 0-3, increasing median training-root degree. |
| `reconstruction_norm_early` | Stored target-row reconstruction-gradient L2 norm at early checkpoint; gradient units. |
| `weighted_aux_norm_early` | Stored target-row auxiliary-gradient L2 norm, coefficient 10, at early checkpoint; gradient units. |
| `ratio_early` | Weighted auxiliary norm divided by `max(reconstruction_norm_early, 1e-30)`; dimensionless. |
| `reconstruction_norm_late` | Same reconstruction norm at late checkpoint; gradient units. |
| `weighted_aux_norm_late` | Same weighted auxiliary norm at late checkpoint; gradient units. |
| `ratio_late` | Late-checkpoint clamped-denominator norm ratio; dimensionless. |

The original instrument uses all 128 ordered target rows. The bare latent VJP is
FP32, multiplied once by 10 in FP32, then converted to FP64 for its norm. This
release verifies scalar division, not that tensor arithmetic history independently.
Zero reconstruction norm alone is allowed; resulting ratios must be finite positive.

### panels.csv

2,000 rows; key `(seed, method, trial)`. This CSV contains no conditional-interval,
refusal, or coverage columns for either method.

| Column | Meaning / Units |
|---|---|
| `seed` | Independent seed identifier 0-9. |
| `method` | `stratified` or `uniform`. |
| `trial` | Frozen repeated-trial identifier 0-99 within seed/method; not independent n. |
| `batch_ids` | Semicolon-separated ordered 16 unique estimation-pool IDs, reused at both checkpoints. |
| `expected_estimate` | Original median over 32 ratios (16 per checkpoint); dimensionless. |
| `expected_early_estimate` | Original early-checkpoint 16-ratio median; dimensionless. |
| `expected_late_estimate` | Original late-checkpoint 16-ratio median; dimensionless. |
| `expected_signed_relative_error` | Original `(estimate - pooled seed reference) / reference`; signed fraction. |

All four expected quantities are checked against recomputed scalar arithmetic.

### stratified_intervals.csv

1,000 rows; key `(seed, trial)`. **Stratified only.** All interval endpoints are
STORED_NOT_RESAMPLED observations from 10,000 historical conditional replicates
per panel, not recomputed distributions or newly certified confidence intervals.

| Column | Meaning / Units |
|---|---|
| `seed` | Independent seed identifier 0-9. |
| `trial` | Frozen Stratified trial 0-99. |
| `R_lower` | Stored 2.5th percentile of conditional pooled ratio medians; dimensionless. |
| `R_upper` | Stored 97.5th percentile of conditional pooled ratio medians; dimensionless. |
| `lambda_lower` | Stored 2.5th percentile of `10 / conditional median ratio`; coefficient units. |
| `lambda_upper` | Stored 97.5th percentile of that coefficient; dimensionless. |
| `expected_CI` | Original boolean: lambda upper/lower <=1.5. Endpoints are stored; decision is recomputed. |
| `expected_PANEL` | Original boolean: split-panel coefficient max/min <=1.25. Split points/decision are recomputed. |
| `expected_CHECKPOINT` | Original boolean: checkpoint ratio-median max/min <=2.0. Decision is recomputed. |

Historical percentiles use linear interpolation at `(n-1)*probability`. Conditional
sampling independently resamples checkpoint/stratum cells with replacement. It
ignores repeated batch identity across checkpoints. Neither endpoints nor apparent
finite-reference coverage establish population-valid 95% coverage or calibration.

### reference_intervals.csv

30 rows; key `(seed, checkpoint)` where checkpoint is `early`, `late`, or `pooled`.
Reference point/check values are recomputed; intervals are STORED_NOT_RESAMPLED.
There were 10,000 original reference replicates per seed, not per table row.
State medians came from the same pooled reference draws without extra resampling.

| Column | Meaning / Units |
|---|---|
| `seed` | Independent seed identifier 0-9. |
| `checkpoint` | `early`, `late`, or both-checkpoint `pooled`. |
| `expected_reference` | Original pooled-reference median: 256 ratios for a state, 512 for a seed; dimensionless. |
| `reference_a` | Original 128-ratio reference-A state median; empty for pooled rows. |
| `reference_b` | Original 128-ratio reference-B state median; empty for pooled rows. |
| `expected_discrepancy` | Original `abs(A-B)/median(A,B)` state fraction; empty for pooled rows. Must be <=0.10 at both states. |
| `R_lower` | Stored reference ratio median's 2.5th percentile, state or pooled; dimensionless. |
| `R_upper` | Stored reference ratio median's 97.5th percentile; dimensionless. |
| `lambda_lower` | Stored pooled `10/median(r)` 2.5th percentile; empty for state rows. |
| `lambda_upper` | Stored pooled coefficient's 97.5th percentile; empty for state rows. |

### frozen_summary.json

| Key | Schema / Meaning |
|---|---|
| `data_disposition` | `CONDITIONAL_ACCEPT_DATA`; arithmetic/data acceptance is conditional. |
| `execution_disposition` | `FAIL_FORMAL_EXECUTION_GATE`; never changed by successful reproduction. |
| `terminal_rss_failure_timing` | `UNRESOLVED`; a suspected terminal exit/RSS race is not proven harmless. |
| `interval_origin` | `STORED_NOT_RESAMPLED`; no research-data bootstrap runs during reproduction. |
| `population_coverage_certified` | `false`; apparent reference-point coverage is not population coverage. |
| `aggregate_metrics` | Mapping of 17 metric names to original `point`, stored `seed_cluster_ci95` two-element bounds, and `independent_units=10`. |
| `seed_bundles` | Array of ten original per-seed metric mappings, array index equal to seed 0-9; every value is checked. |
| `ablation_counts` | Mapping from each of five ablations to `accepted` and `bad_accepted` repeated-panel counts. |
| `verification_targets` | Independently recorded aggregate assertions; names ending `_percent` use percent, `_pp` use percentage points, `_count` are counts, `reference_max_discrepancy` is a fraction. |
| `input_sha256` | Filename-to-SHA256 mapping for the four canonical CSVs; checked before reproduction. |
| `gate_failures` | Original rejected-panel counts for `CI`, `PANEL`, `CHECKPOINT`; denominator 1,000 Stratified panels. |

Aggregate/seed-bundle metric meanings:

| Metric Names | Meaning / Units |
|---|---|
| `stratified_mae`, `uniform_mae` | Within-seed mean absolute relative error over 100 trials, then equal-seed mean; error fraction. |
| `stratified_acceptable_fraction`, `uniform_acceptable_fraction` | Fraction of each seed's 100 trials with error <=20%, then equal-seed mean. |
| `mae_difference_stratified_minus_uniform` | Paired difference of seed MAEs; signed error fraction, multiply by 100 for percentage points. |
| `acceptable_fraction_difference` | Paired Stratified-minus-Uniform acceptable-panel fraction difference. |
| `historical_apparent_reference_point_coverage` | Fraction of Stratified stored ratio intervals containing the finite/noisy pooled reference point. |
| `ALL_acceptance_fraction`, `DROP_CI_acceptance_fraction`, `DROP_PANEL_acceptance_fraction`, `DROP_CHECKPOINT_acceptance_fraction`, `NONE_OF_THREE_acceptance_fraction` | Retained fraction of the fixed 100 Stratified trials per seed, averaged equally over seeds. |
| `ALL_accepted_mae`, `DROP_CI_accepted_mae`, `DROP_PANEL_accepted_mae`, `DROP_CHECKPOINT_accepted_mae`, `NONE_OF_THREE_accepted_mae` | Mean error within retained trials of each seed, then equal-seed mean; error fraction, not pooled-panel weighting. |

Stored aggregate bounds used 10,000 seed-cluster replicates, sampling ten complete
seed bundles with replacement and one existing reference-replicate index per seed
choice. These include both methods/trials/checkpoints. Uniform **aggregate error**
intervals are not Uniform ratio intervals or Uniform conditional resampling.

## Reproduced Output Tables

These tables are generated into a new output directory. They are not canonical
inputs. A sample output set accompanies the experiment protocol.

### reference_checks.csv (30 rows)

| Column | Meaning / Units |
|---|---|
| `seed`, `checkpoint` | Same hierarchy as canonical reference intervals. |
| `reference` | Recomputed state/pooled reference ratio median; dimensionless. |
| `discrepancy` | Recomputed state discrepancy fraction; empty for pooled. |
| `interval_origin` | Always `STORED_NOT_RESAMPLED`. |
| `R_lower`, `R_upper` | Copied stored state/pooled reference ratio bounds. |
| `lambda_lower`, `lambda_upper` | Copied pooled coefficient bounds; empty for states. |

### sampling_panels.csv (2,000 rows)

| Column | Meaning / Units |
|---|---|
| `seed`, `method`, `trial` | Independent seed and repeated panel identifiers. |
| `estimate` | Recomputed 32-ratio pooled panel median; dimensionless. |
| `signed_relative_error` | Recomputed signed fractional error against seed reference. |
| `absolute_relative_error` | Absolute fractional error. |
| `acceptable_relative_error` | Boolean, absolute fractional error <=0.20. Not a refusal-screen acceptance. |
| `early_estimate`, `late_estimate` | Recomputed state 16-ratio medians; dimensionless. |

No Uniform ratio-interval, refusal, coverage, or conditional-resampling fields occur.

### stratified_refusal.csv (1,000 rows)

| Column | Meaning / Units |
|---|---|
| `seed`, `trial` | Stratified seed/trial identifiers. |
| `CI`, `PANEL`, `CHECKPOINT` | Recomputed boolean passes of the three historical screens. |
| `ALL` | All three screens enabled; boolean retention. |
| `DROP_CI`, `DROP_PANEL`, `DROP_CHECKPOINT` | Boolean retention with that one screen omitted. |
| `NONE_OF_THREE` | Boolean retention with all three screens omitted; finite/support checks remain mandatory. |
| `bad_error` | Boolean, actual absolute relative error >0.20. |
| `apparent_reference_point_coverage` | Boolean, stored ratio interval contains seed reference point; not population coverage. |

### seed_summary.csv (20 rows)

| Column | Meaning / Units |
|---|---|
| `seed`, `method` | Ten independent seeds, two methods per seed. |
| `trials` | Repeated trials per method/seed, always 100. |
| `reference` | Recomputed seed reference ratio median; dimensionless. |
| `mae` | Mean absolute relative error over 100 repeated panels; fraction. |
| `acceptable_error_fraction` | Fraction of those panels with error <=0.20. |
| `estimate_sample_SD` | Sample SD of 100 panel estimates; ratio units. |
| `estimate_IQR` | 75th minus 25th percentile of estimates, historical interpolation; ratio units. |

### statewise_sampling_errors.csv (4,000 rows)

| Column | Meaning / Units |
|---|---|
| `seed`, `method`, `trial`, `checkpoint` | Repeated state/panel identifiers nested in ten seeds. |
| `estimate` | State 16-ratio panel median; dimensionless. |
| `reference` | State 256-ratio reference median; dimensionless. |
| `signed_relative_error` | `(estimate-reference)/reference`; signed fraction. |
| `absolute_relative_error` | Absolute fractional error. |
| `acceptable_relative_error` | Boolean, absolute error <=0.20. |

### statewise_summary.csv (40 rows)

| Column | Meaning / Units |
|---|---|
| `seed`, `method`, `checkpoint` | Seed/method/checkpoint group. |
| `trials` | Repeated state panels, always 100. |
| `mean_signed_relative_error` | Mean signed state-panel error; fraction. |
| `mae` | Mean absolute state-panel error; fraction. |
| `acceptable_error_fraction` | Fraction of state panels with error <=0.20. |
| `estimate_sample_SD`, `estimate_IQR` | Sample SD and historical-interpolated IQR of state panel medians; ratio units. |

### refusal_ablations.csv (5 rows)

| Column | Meaning / Units |
|---|---|
| `ablation` | One of ALL, DROP_CI, DROP_PANEL, DROP_CHECKPOINT, NONE_OF_THREE. |
| `retained` | Retained repeated panels across ten seeds; denominator 1,000. |
| `bad_retained` | Retained panels with absolute error >0.20. |
| `seed_equal_retained_mae` | Mean retained-panel error within each seed, then equal-seed mean; fraction. |
| `pooled_retained_mae` | Mean over all retained panels, weighting seeds by retained counts; fraction. Different estimand. |

### stored_intervals.csv (17 rows)

| Column | Meaning / Units |
|---|---|
| `metric` | Aggregate metric name, with meaning/units in the metric table above. |
| `point` | Recomputed equal-seed aggregate, verified against frozen point. |
| `lower`, `upper` | Copied stored seed-cluster 2.5th/97.5th percentile bounds, same units as point; NOT newly resampled. |
| `independent_seed_count` | Always 10. |
| `interval_origin` | Always `STORED_NOT_RESAMPLED`. |
| `interval_target` | Aggregate scalar metric, not a Uniform or other per-panel ratio interval. |

### metrics.json

| Keys | Meaning / Units |
|---|---|
| `independent_seed_count`, `panel_count`, `scalar_measurement_count` | 10 seeds, 2,000 paired-checkpoint panels, 10,240 scalar measurements. |
| `stratified_mae_percent`, `uniform_mae_percent` | Verified equal-seed MAEs in percent. |
| `paired_mae_difference_pp` | Stratified-minus-Uniform MAE in percentage points. |
| `all_retained_count`, `all_bad_retained_count`, `no_refusal_bad_count` | Repeated-panel counts; denominators 1,000, retained count, and 1,000 respectively. |
| `all_retention_percent` | Percentage retained by all three screens. |
| `all_bad_retained_percent`, `no_refusal_bad_percent` | Bad fraction among retained panels versus all panels, in percent. |
| `all_retained_mae_seed_equal_percent`, `all_retained_mae_pooled_percent` | Two explicitly distinct accepted-error estimands, in percent. |
| `apparent_reference_point_coverage_percent` | Apparent finite/noisy reference-point coverage in percent; not population coverage. |
| `stored_paired_mae_difference_ci_pp` | Two stored aggregate bounds in percentage points. |
| `stored_reference_point_coverage_ci_percent` | Two stored apparent-coverage aggregate bounds in percent. |
| `reference_max_discrepancy` | Maximum practical A/B state discrepancy; fraction. |
| `refusal_gate_failures` | Mapping CI/PANEL/CHECKPOINT to failed repeated-panel counts out of 1,000. |
| `data_disposition`, `execution_disposition`, `terminal_rss_failure_timing` | Same fixed conditional-data/failed-execution/unresolved statuses as canonical JSON. |
| `interval_origin` | `STORED_NOT_RESAMPLED`. |
| `new_bootstrap_resamples` | Always 0 research-data resamples. |
| `bootstrap_rng_history_independently_verified`, `population_coverage_certified` | Both false; arithmetic verification does not establish these claims. |

## Scientific Implementation Availability

`code/analysis/estimator_reliability.py` contains the CPU reproducer and inspection
implementations of clamped scalar ratios, historical percentile/rounding formulas,
canonical metadata hashing, value-blind pool rank slices, genuine Uniform and
Stratified sampling, the checkpoint/stratum conditional replacement kernel, and
seed/reference bundle choice formula. Reproduction uses stored selections and
intervals and does not call sampling/bootstrap helpers. The historical kernel's
private token-based RNG construction is replaced by explicit RNG injection for
inspection; it is not presented as an independently replayed RNG history.
