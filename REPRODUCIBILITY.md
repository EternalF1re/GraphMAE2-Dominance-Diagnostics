# Reproducibility Map

## Scope boundary

This is an analysis-level release. Included processed records are sufficient to
recompute the reported summaries, bootstrap intervals, tables, and figures.
Datasets and checkpoints are not included. Running the attributed training core
or checkpoint-gradient diagnostics requires user-provided compatible assets and
is outside the zero-data analysis workflow.

## Figures

| Item | Script | Canonical input | Frozen artifact | Default rerun output |
|---|---|---|---|---|
| Figure 1 | `figures/figure_1/plot_factorial_trajectory.py` | `processed_records/factorial/checkpoint_validation.csv` | `figures/figure_1/figure_1_reference.pdf` | `figures/figure_1/reproduced/figure_1.pdf` |
| Figure 2 | `figures/figure_2/plot_reddit_replication.py` | `processed_records/reddit/replication.csv` | `figures/figure_2/figure_2_reference.pdf` | `figures/figure_2/reproduced/figure_2.pdf` |
| Figure 3 | `figures/figure_3/plot_dominance_performance.py` | Arxiv response and cross-dataset records | `figures/figure_3/figure_3_reference.pdf` | `figures/figure_3/reproduced/figure_3.pdf` |
| Figure 4 | `figures/figure_4/plot_protocol_parameters.py` | `processed_records/arxiv/protocol_parameter_response.csv` | `figures/figure_4/figure_4_reference.pdf` | `figures/figure_4/reproduced/figure_4.pdf` |
| Figure 5 | `figures/figure_5/plot_maskgae_dose_response.py` | `processed_records/maskgae_pubmed/checkpoints.csv` | `figures/figure_5/figure_5_reference.pdf` | `figures/figure_5/reproduced/figure_5.pdf` |
| Appendix conflicts | `figures/appendix_conflicts/plot_pcgrad_conflicts.py` | PCGrad histogram bins | `figures/appendix_conflicts/appendix_conflicts_reference.pdf` | `figures/appendix_conflicts/reproduced/appendix_conflicts.pdf` |

Rerun directories are excluded from `MANIFEST.sha256` and release ZIPs. The
verifier checks canonical input equality and frozen reference hashes, not PDF
binary equality across plotting backends.

The previous three-seed Figure1 is retained unchanged at
`figures/historical/three_seed_trajectory/`. It is historical, not a current
main figure. Figures2/3/4 and AppendixD.1 retain the remote baseline bytes.

## New manuscript evidence

`python -B reproduce_analysis.py` reproduces the four added analyses from
processed scalar data. The ten-seed factorial protocol is not pooled with the
earlier three-seed scans or method comparisons. AUC integrates actual update
distances250..8532 and divides by8282; the fixed final8532 endpoint is separately
post-hoc. Both use matched-seed Student-t df9 with three-effect Bonferroni families
separately for each protocol/endpoint. Nominal95% intervals are labelled separately.

Geometry uses two batches per checkpoint, three checkpoints per seed, ten seed
means, with full-Z and all128-target-row descriptors kept separate. G2's local
ratio is measured in its own model state; encoder routing multiplies by E/H=.03.
The approximately20-fold difference is a ratio of group means, not a per-node
orthogonality or optimizer-update claim.

MaskGAE AUC integrates seven epochs15..500 and divides by485. Both endpoints
use ten paired pretraining seeds, with three probes nested within each checkpoint.
Ratios use the median of ten within-seed seven-checkpoint medians. Clip frequency
is counted from actual preclip norm>1 updates, not inferred from rounded percentages.
The historical alpha=.5 contrast stays separate; the three added doses have
exploratory Holm3 comparisons, separately for AUC and epoch500, with ordinary
paired95% intervals. No joint six-test correction, optimal-dose rule or independent
replication from shared controls is claimed.

Reliability reuses10240 scalar measurements,2000 panels and stored interval
endpoints. The independent unit is ten seeds, not panels. Refusal/conditional
bootstrap apply only to stratified panels. Uniform is an equal-budget sampling
comparator, without conditional refusal/coverage. The80.9% apparent coverage is
against a finite/noisy reference, not population coverage. Original execution
failed its final monitoring gate; the numeric evidence was conditionally retained.
The CPU workflow does not repair that status or rerun scientific resampling.

Analysis-level reproduction is supported; exact historical full training,
raw VJP regeneration, private Schedule replay, node/root-disjointness from
unreleased node arrays and original bootstrap RNG replay are not claimed.

## Tables

### Numerical reproducibility policy

Manifest verification concerns the exact bytes of the frozen artifacts, not
whether every environment regenerates identical CSV text. The recorded CPU
acceptance environment in `environment/analysis_environment.json` used Python
3.13.7, NumPy 2.3.3, SciPy 1.16.2 and Matplotlib 3.10.7. All 18 revision tables
and the claim map matched byte-for-byte in that environment. This is an
environment-specific observation, not a cross-environment guarantee.

An independent review using SciPy 1.17.0 reported 15 of 18 tables byte-identical.
The other three were `factorial/factorial_effects.csv`,
`maskgae_pubmed/dose_summary.csv` and `maskgae_pubmed/paired_contrasts.csv`,
with only last-digit numerical differences (maximum absolute difference below
2e-11). The claim map had 17 differing numerical cells (maximum 5.6e-17), and
all 84 scientific claim checks passed. These are reviewer-reported results;
they are not a claim that both environments were rerun in this release.

For an environment comparison of these generated tables, require identical
column names, row order, row counts and nonnumeric fields, and finite numeric
values differing by at most 2e-11 in each cell's stored units (absolute bound,
zero relative tolerance). Report byte identity separately. This is an
analysis-output comparison bound motivated by the independent review, not a
change to scientific inference, frozen numeric gates or claim tolerances.
The 84 claim-specific acceptance tolerances remain in
`protocols/revision_claims.json`. Differences exceeding the comparison bound
must be reported and investigated, not fixed by rewriting frozen reference CSVs.
PDF byte identity across plotting backends is likewise not a scientific
requirement; the approved reference PDFs and their manifest hashes stay fixed.

`tables/generate_summary_tables.py` deterministically creates:

- `weighting_method_summary.csv` from the historical seven-method record;
- `weighting_method_summary_current.csv` from the current nine-method record; and
- `sampling_design_summary.csv` from the released sampling description.

## Finalized reference estimator

For each dataset, the released finalized estimator contains two checkpoints,
four degree-rank groups, and four deterministic batches per group. The pooled
median over all 32 records is the point estimate. The bootstrap utility draws
10,000 deterministic resamples while preserving the record count in every
checkpoint-by-degree-group cell.

The earlier compact diagnostic used a different sampling and estimation design.
It motivated later measurement work but is not the authoritative cross-dataset
reference. The finalized values are:

| Dataset | Point estimate | 95% bootstrap interval | Released record |
|---|---:|---:|---|
| ogbn-arxiv | 31.7653148904106 | [29.849883433713472, 44.0618155456329] | `processed_records/arxiv/reference_summary.csv` |
| Reddit | 12.62592612939547 | [11.463302621964752, 14.199962981296819] | `processed_records/reddit/reference_summary.csv` |

## Recent baseline extension

| Item | Public processed record | Scope |
| --- | --- | --- |
| Aligned-MTL seed AUC and validation checkpoints | `results/sota_extension/aligned_mtl/` | Three seeds, seven validation checkpoints each |
| FAMO seed AUC and validation checkpoints | `results/sota_extension/famo/` | Three seeds, seven validation checkpoints each |
| Direct checkpoint dominance | `results/sota_extension/direct_rhat/` | Seed-by-checkpoint median of three diagnostic rows; method semantics in README |
| Nash-MTL attempt | `results/sota_extension/nash_failure/` | Failure provenance only; no completed AUC |
| Online training overhead | `results/sota_extension/overhead/` | 50 warm-up plus 300 measured optimizer steps |
| Offline diagnostic overhead | `results/sota_extension/diagnostic_overhead/` | 32 gradient batches plus 10,000 CPU bootstrap resamples |

Method attribution and the fixed protocol are in `experiments/sota_extension/`. The extension's public processed records permit comparison and arithmetic checks, not replay of the original training jobs. The full historical SOTA training adapter and private schedules are not in this package; a blanket assertion of end-to-end reproduction for this extension would be inaccurate.

The public method integration core in `code/sota_extension/` preserves the frozen Aligned-MTL/FAMO method functions, but excludes the private historical schedule and runner. It is not a one-command recreation of the formal jobs. Current Table 2 is `processed_records/method_comparisons/weighting_methods_current.csv` and `tables/generated/weighting_method_summary_current.csv`; use `python tables/assemble_current_method_record.py` followed by `python tables/generate_summary_tables.py`. Historical seven-method records remain untouched. Appendix C.6 maps to `results/sota_extension/direct_rhat/C6_checkpoint_level_rhat.csv` and `C6_summary.json`. Appendix E maps to per-seed AUC, matched differences and overhead records. Appendix B maps to public preregistration, post-result hypothesis adjudication and the incomplete Nash attempt.

## Other frozen checks

- PCGrad histogram: 25,596 updates, including 9,417 conflicts.
- Figure-local CSV files are byte-identical copies of their canonical records.
- Arxiv and Reddit finalized diagnostic inputs each contain 32 rows with four
  rows in each checkpoint-by-degree-group cell.

## Configuration map

| File | Kind | Accepted by `--config_path`? | Historical standalone file retained? |
|---|---|---:|---:|
| `code/graphmae2/configs/ogbn-arxiv_upstream_default.yaml` | Flat upstream/default runtime config | Yes | Yes |
| `paper_records/ogbn-arxiv_execution_record.json` | Paper execution/provenance record | No | No |
| `paper_records/reddit_execution_record.json` | Paper execution/provenance record | No | No |

For both paper execution records, the model and optimizer fields are retained.
The exact private root schedules and original dedicated runners are excluded,
so the package does not claim one-command historical training replay. A paper
execution record is not a runnable GraphMAE2 configuration. Runtime loading is
fail-fast for the nested record schema; it never flattens a record into runtime
arguments. The original standalone paper runtime YAML files were not retained.

## Dependency map

Analysis utilities use the Python standard library plus NumPy and Matplotlib.
The attributed GraphMAE2 core additionally imports PyTorch, DGL, OGB, PyYAML,
scikit-learn, SciPy, psutil, tqdm, and tensorboardX. Weights & Biases logging and
`localgraphclustering` preprocessing are optional. Basic release verification
and processed-record analysis do not need the full GraphMAE2 dependency stack.
The separate full CLI audit runs GraphMAE2 `--help` commands and therefore
requires DGL, OGB, and the declared runtime dependencies.

## Release integrity

For an official extracted ZIP, basic verification is
`python tests/verify_release.py`; it does not require DGL or OGB. The full
GraphMAE2 CLI audit `python tests/audit_public_clis.py` additionally requires
both dependency groups. The unittest suite checks analysis CLIs and conditionally
checks PyYAML-dependent configuration behavior.
The frozen manifest detects missing, additional, and modified tracked files.
`tests/generate_manifest.py` is a maintainer-only assembly command and must not
precede official-release verification.
