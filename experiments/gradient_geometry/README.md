# Representation-Gradient Geometry

This experiment supports manuscript Table 3. The column-by-column dictionary is
`processed_records/gradient_geometry/DATA_DICTIONARY.md`.

This is analysis-level reproduction from 120 already measured scalar records:
Protocol A, groups G1/G2, seeds 0 through 9, updates 250/3555/8532, and fixed
zero-based batch positions 93/118. No model, dataset, checkpoint, Schedule body,
embedding, gradient tensor, label, probe, or training launcher is included here.

From the repository root:

```bash
python code/analysis/gradient_geometry.py --output-dir experiments/gradient_geometry/reproduced
python -B -m unittest discover -s tests -p test_gradient_geometry_release.py -v
```

The standard-library API is `reproduce(repository_root, output_dir) -> dict` in
`code/analysis/gradient_geometry.py`. It recreates `per_checkpoint.csv`,
`per_seed.csv`, `group_summary.csv`, `paired_seed_differences.csv`, and
`summary.json`. Frozen examples are under `reference/`; reruns use a separate
`reproduced/` directory. No bootstrap or significance test is performed.

## Scientific Definitions

`Z` is the entire online encoder output for the sampled graph; `Z_T` restricts
the same two representation gradients to all 128 ordered training-target rows.
Zeros remain present: no shared-support or nonzero-row filtering is performed.
These are matrix/Frobenius norms and flattened matrix cosines, not averages of
per-node cosines and not parameter-gradient or optimizer-displacement geometry.

The reconstruction gradient differentiates the sum of three native remask
reconstruction losses with respect to the same full Z. The measured auxiliary
gradient differentiates `10 * bare_latent_loss` with respect to that Z in BOTH
groups. Full-Z cosines are primary; target-matrix cosines are supplementary.
The executed source excerpts in `code/measurements/gradient_geometry_components.py`
are non-runnable inspection text. They are not a public measurement entry point.

Training coefficients are G1 `(E=10, H=10)` and G2 `(E=0.3, H=10)`. The local
full-Z ratio is each record's own measured `weighted_aux_gradient_norm /
reconstruction_gradient_norm`. Its routed counterpart is derived as
`(E/H) * local_ratio`: unchanged for G1, and `.03 * G2_local_ratio` for G2.
It is not a separately measured routed gradient, a clipped ratio, or a ratio
computed by scaling the G1 measurement. The local auxiliary coefficient is 10
in both groups. Bare auxiliary loss is reported separately in the CSV.

## Aggregation And Limits

Average the two batch values per seed/group/checkpoint, then average the three
checkpoint means per seed/group, then average the ten seed means. Sample SD
uses those ten seed means (`ddof=1`). There are ten independent seeds, not 120
independent training replicates. Group ratios average per-measurement ratios;
they are NOT ratios of group-average norms. The approximately 20.32-fold routed
contrast is a ratio of group means, not a mean of paired seed ratios.

The overall local/routed ratios are approximately 2.6482906/2.6482906 for G1
and 4.3441494/0.13032448 for G2. Full-Z cosine means are approximately
0.0026807322/0.0031326280, while all-target cosine means are
0.0347404362/0.0381831394. Near-zero aggregate cosine does not imply strict
per-node orthogonality. These fixed-state descriptive results do not establish
a causal mechanism, downstream utility, or an accuracy prediction. Clipping
geometry was not measured; these records are preclip only.

Missing, duplicate, undefined, or nonfinite records fail complete-design
reproduction; no seed is silently filtered or replaced. Identity hashes permit
checking paired stochastic identities, but hashes cannot regenerate model/data
payloads or independently rerun the original VJPs.

## Attribution

The study-authored records, analysis, and scientific capture excerpts are
distributed under `LICENSE-AUTHORS`. The original GraphMAE2 context is THUDM
GraphMAE2, distributed under the MIT terms in `LICENSE` and described in
`THIRD_PARTY_NOTICES.md`. No upstream model implementation is added by this
extension, and no operating-system, execution-authorization, or signing framework
is redistributed in the scientific excerpt.
