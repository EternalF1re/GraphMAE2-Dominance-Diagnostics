# Estimator Sampling Reliability

## Reproduction

From the repository root, using only the Python standard library:

```sh
python -B code/analysis/estimator_reliability.py --output-dir results/reproduced_reliability
python -B -m unittest discover -s tests -p test_estimator_reliability_release.py -v
```

The output directory must be new. The API is
`reproduce(repository_root, output_dir) -> dict` of named metrics. It writes
reference checks, all sampling panels, per-seed summaries, refusal ablations,
stored intervals, and a metrics JSON. Inputs are never overwritten. No model,
GPU, graph, label, training, or new research-data bootstrap is needed or invoked.
The original three-seed estimator and its input records are a separate experiment
and are not replaced or modified by this release.

## Fixed Scientific Design

GraphMAE2/ogbn-arxiv, ten pretraining seeds, two fixed checkpoints (250 and 8532
optimizer updates). Each seed has 512 distinct batch identities: 128 reference A,
128 reference B, and 256 estimation batches. Degree ranking defines four strata;
each reference has 32 batches per stratum and estimation has 64. The two references
and estimation pool have disjoint batch identities. The original processed
identity receipts also report disjoint training roots; global root arrays are
not released here, so that stronger assertion is provenance-only in this payload.
Context graphs may overlap: disjoint roots do not establish independent graphs.
Strata originate from sorting median training-root degree, breaking ties by frozen
training-batch position, and splitting the ordered candidates into four near-equal
parts. Within stratum, value-blind SHA-256 ranking of canonical JSON containing
protocol token, seed, stratum, batch hash, and purpose `asset_pool` selects ranks
0:32 for reference A, 32:64 for reference B, and 64:128 for estimation. The source
excludes batch positions 0/355/710 before pool selection; that exclusion was a
prospective amendment, not a historical outcome-blind ten-seed registration.

There are 100 frozen trials per method and seed. A Stratified panel draws four
unique batches without replacement per stratum (16 total). A genuine Uniform
panel draws 16 without replacement from the balanced 256-batch estimation pool,
without conditioning, rejection, or redraw to achieve equal stratum counts.
Exactly the same selected identities are used at both checkpoints. Even a Uniform
panel with accidental 4/4/4/4 counts remains sampling-quality-only.

The public batch IDs are per-seed ordinal aliases, not node identifiers. Their
order preserves original batch-hash ordering, including within-stratum alternating
splits. The inputs retain actual pool/stratum assignments and panel memberships;
no selection or new panel sampling occurs during reproduction.

## Measurement and Point Estimates

The historical instrument differentiates three summed native reconstruction
losses and the **bare** latent loss on the same full online representation Z.
The FP32 bare latent VJP is multiplied once by 10 **before** conversion to FP64
for the target-row norm. Select all 128 ordered target rows, including rows whose
reconstruction derivative is zero:

```text
r = norm_FP64((10 * bare_latent_VJP_FP32)[T])
    / max(norm_FP64(reconstruction_VJP_FP32[T]), 1e-30)
```

Zero reconstruction norm alone is not an added refusal. Resulting ratios must be
finite and positive; support, identity, and finite checks are never ablated.
The released scalar norms verify this ratio calculation, not autograd, FP32
weighting history, model state, or tensor support independently.

For each checkpoint, reference A/B medians are computed independently. Require
`abs(A-B)/median(A,B) <= 0.10` at **both** checkpoints. State reference is the
median of the pooled 256 reference values; seed reference is the pooled median
of 512 values over both checkpoints. A panel estimate is a state 16-value median
or seed 32-value pooled median. This targets an equal-stratum batch mixture, not
a node-weighted population median.

Signed error is `(estimate-reference)/reference`; MAE is mean **absolute relative
error**, not accuracy loss. An error is operationally acceptable when <=20%.
First average 100 repeated trials within seed/method, then average ten seed means
equally. Checkpoints/trials do not increase the independent sample size beyond ten.
For retained-panel errors, seed-equal and pooled-panel weighting are distinct.
An undefined seed summary must not be repaired by dropping that seed.

## Historical Conditional Method and Refusal Screens

The minimal historical kernel, sampling formulas, pool rank slices, interpolation,
and seed-bundle choice formula are in `code/analysis/estimator_reliability.py` for
inspection. The conditional kernel groups canonical rows by checkpoint and degree
stratum and resamples each cell with replacement, preserving cell sizes, before
taking the pooled median and `lambda=10/median(r)`. It resamples checkpoint cells
independently, **ignoring repeated batch identity across checkpoints**. Original
canonical JSON hashing is shown, but private experiment salt strings are omitted;
stored panel selections/intervals, not new pseudonym-based RNG streams, are inputs.

Only Stratified panels have historical conditional intervals and refusal screens:

- lambda interval upper/lower <=1.5;
- alternating sorted within-stratum split-panel lambda max/min <=1.25;
- checkpoint median ratio max/min <=2.0.

All screens, each single-screen omission, and omission of all three operate on
identical cached Stratified records. Finite-positive/support requirements never
disappear. Uniform has no conditional interval, refusal screen, acceptance,
coverage, or conditional resampling, regardless of its realized stratum counts.

The stored calculation used 10,000 conditional replicates per Stratified panel
(1,000 panels) and 10,000 per seed reference. Stored aggregate intervals used
10,000 seed-cluster draws: ten seed bundles with replacement per draw, retaining
both methods/checkpoints/trials, and selecting one existing reference replicate
per selected seed to recompute its errors. Seed-cluster intervals for Uniform
**aggregate sampling error** are not Uniform ratio intervals. The reproducer
checks endpoints as stored records and recomputes their deterministic use in
screens and coverage; it does **not** regenerate any bootstrap distribution or
certify RNG history from recorded counts. All newly drawn research resamples=0.

## Results and Limits

Stratified MAE 7.730391653991314%, Uniform 7.581631481816684%, paired difference
+0.14876017217463026 percentage points. Stored seed-cluster interval for that
difference: [-0.15097501375565432, 0.44431373836266615] percentage points.
The full refusal rule retains 738/1000 panels with 27/738 unacceptable errors,
versus 37/1000 without refusal. Seed-equal retained MAE is 7.943535419223971%,
not the differently weighted pooled retained MAE of 7.883999759117166%.
The checkpoint screen fails no panels. Stratified superiority is not established;
neither is Uniform superiority. Reduced absolute false acceptances mostly reflect
fewer retained panels, not an established reliability benefit.

Apparent coverage of the finite/noisy reference point is 80.9% (stored cluster
interval 76.0%-83.6%). This is **not population-valid 95% coverage**, interval
calibration, downstream utility, architecture generalization, or causal mediation.
The operational thresholds were fixed after earlier outcomes were visible, not
fully outcome-blind. References pass the practical discrepancy screen here;
reference instability is not the reason the formal execution gate failed.

Disposition remains **CONDITIONAL_ACCEPT_DATA / FAIL_FORMAL_EXECUTION_GATE**.
The worker completed the recorded numerical work, but the parent failed a terminal
RSS observation. The precise exit/RSS race and write/failure chronology remain
**UNRESOLVED**, not proven harmless. Successful scalar arithmetic reproduction
does not repair execution status or establish continuous resource enforcement.
