# GraphMAE2 Dominance Diagnostics

This repository accompanies a study of loss-component dominance and weighting
in GraphMAE2 and MaskGAE. It publishes compact processed records, deterministic analysis
utilities, paper-figure scripts, and an attributed subset of the GraphMAE2
implementation for inspection.

## Reproducibility scope

The package supports **analysis-level reproduction from processed records**:

- rechecking the authoritative Arxiv and Reddit reference estimates;
- recomputing the released deterministic bootstrap intervals;
- rebuilding summary tables;
- redrawing the five main figures and Appendix Figure D.1; and
- inspecting the shared-target component-gradient implementation.

The package does **not** contain graph datasets, LC artifacts, model
checkpoints, optimizer states, embeddings, or complete historical training
outputs. It does not claim raw-data end-to-end reproduction of every training
trajectory. Checkpoint diagnostics require a compatible checkpoint and assets
obtained by the user. None of the analysis-only commands below trains a model.

## Layout

- `code/graphmae2/`: attributed GraphMAE2 core subset and retained configs.
- `paper_records/`: non-runnable paper execution/provenance records.
- `code/diagnostics/`: shared-target measurement, sampling, and bootstrap code.
- `code/comparisons/`: processed-record comparison utilities.
- `processed_records/`: compact canonical records used in the paper analyses.
- `figures/`: frozen reference PDFs, inputs, and isolated rerun scripts.
- `tables/`: deterministic summary-table generation.
- `protocols/`: measurement and decision rules with provenance boundaries.
- `environment/`: analysis, GraphMAE2, and optional dependency declarations.
- `tests/`: release verification, packaging, CLI audit, and regression tests.
- `historical_release/`: unchanged prior sync report and manifest, retained for provenance only.

## Added ten-seed analyses

The current manuscript additions are organized alongside the existing records:

| Experiment | Processed records | Scientific implementation / protocol | Analysis |
|---|---|---|---|
| GraphMAE2 encoder/head factorial,80runs | `processed_records/factorial/` | `experiments/factorial/`, `code/experiments/factorial_weighting.py` | `code/analysis/factorial.py` |
| GraphMAE2 G1/G2 gradient geometry,120measurements | `processed_records/gradient_geometry/` | `experiments/gradient_geometry/`, `code/measurements/gradient_geometry_components.py` | `code/analysis/gradient_geometry.py` |
| MaskGAE PubMed five weights,50runs | `processed_records/maskgae_pubmed/` | `experiments/maskgae_pubmed/`, `code/measurements/maskgae_components.py` | `code/analysis/maskgae_pubmed.py` |
| Gradient-ratio estimator reliability,ten seeds | `processed_records/estimator_reliability/` | `experiments/estimator_reliability/` | `code/analysis/estimator_reliability.py` |

After installing `environment/requirements-analysis.txt`, a single CPU-only command
recomputes the added analyses, checks manuscript claims and generates Figures1/5:

```bash
python -B reproduce_analysis.py
python -B -m unittest discover -s tests -v
```

Use a new `--output-dir` for a second run; `--skip-figures` checks only arithmetic.
Numerical reproducibility is distinct from byte identity: SciPy versions and
platforms can change the last floating-point digits of statistical tables.
The frozen reference CSVs are protected by manifests and must not be rewritten
or re-rounded to hide differences. See the numerical reproducibility policy in
`REPRODUCIBILITY.md`; the existing 84 claim-specific tolerances are unchanged.
No graph downloads, model imports, training, embedding extraction, validation
probes, GPU, or new bootstrap resampling occurs in this workflow. Stored
bootstrap interval endpoints are verified/used as stored observations, not
claimed as newly reproduced RNG histories. See `DATA_DICTIONARY.md`,
`MANUSCRIPT_CLAIM_MAP.csv` and `REPRODUCIBILITY.md` for units, sources and limits.

The reliability data retain an independently adjudicated terminal monitoring
failure: numerical data are conditionally accepted, but formal execution did
not pass. Sampling/refusal superiority and population-valid bootstrap coverage
are not demonstrated. These limitations are part of the published evidence.

## Basic release verification

After extracting the official ZIP, run this from the repository root:

```bash
python tests/verify_release.py
```

This standard-library verifier checks package structure, the frozen manifest,
privacy/path rules, canonical processed records, and frozen scientific values.
It needs no graph dataset, model checkpoint, GPU, DGL, or OGB, and it does not
train or run a model. Do not regenerate the manifest before verifying an
official release.

## Full release regression suite

Run the unittest suite, then the separate full public CLI `--help` audit.
The latter requires GraphMAE2 runtime dependencies, including DGL and OGB:

```bash
python -m pip install -r environment/requirements-analysis.txt
python -m pip install -r environment/requirements-graphmae2.txt
python -m unittest tests.test_r1_release -v
python tests/audit_public_clis.py
```

These commands perform no training, inference, dataset download, or checkpoint
loading. The unittest suite tests analysis CLIs and conditionally checks PyYAML;
the separate full CLI audit checks GraphMAE2 imports.

## Analysis-only reproduction

Install only the plotting dependencies for figures and tables:

```bash
python -m pip install -r environment/requirements-analysis.txt
```

Then run:

```bash
python code/diagnostics/bootstrap_reference_interval.py \
  --input processed_records/arxiv/reference_diagnostic_records.csv \
  --dataset ogbn-arxiv \
  --output reproduced/arxiv_reference.json
python code/diagnostics/bootstrap_reference_interval.py \
  --input processed_records/reddit/reference_diagnostic_records.csv \
  --dataset reddit \
  --output reproduced/reddit_reference.json
python tables/generate_summary_tables.py
python figures/figure_1/plot_factorial_trajectory.py
python figures/figure_5/plot_maskgae_dose_response.py
python figures/figure_2/plot_reddit_replication.py
python figures/figure_3/plot_dominance_performance.py
python figures/figure_4/plot_protocol_parameters.py
python figures/appendix_conflicts/plot_pcgrad_conflicts.py
```

Figure scripts write under each figure's `reproduced/` directory. They never
overwrite the manifest-tracked `*_reference.pdf` files. PDF binary identity is
not treated as a scientific requirement; canonical CSV values and key numeric
claims are verified instead.

## Recent baseline extension

The current release includes completed comparisons with Aligned-MTL (CVPR 2023) and FAMO (NeurIPS 2023) under the common GraphMAE2 protocol. The completed comparisons report validation trajectories and direct target-representation diagnostics. Nash-MTL was attempted but not retained as a completed comparator under the adopted preregistered integration configuration.

See `experiments/sota_extension/SOTA_BASELINE_PROVENANCE.md` for author-code attribution, the public preregistration and subsequent adjudication in that directory, and `code/sota_extension/` for the formal method integration core. Compact, path-free evidence is under `results/sota_extension/`. The public adapter is derived from the frozen formal implementation but is not a one-command replay of the private historical schedule. No checkpoint tensors or private schedules are included.

`processed_records/method_comparisons/weighting_methods.csv` and `tables/generated/weighting_method_summary.csv` are the pre-extension historical seven-method comparison. The current nine-method Table 2 is `processed_records/method_comparisons/weighting_methods_current.csv` and `tables/generated/weighting_method_summary_current.csv`. Regenerate the current record and both summaries with:

```bash
python tables/assemble_current_method_record.py
python tables/generate_summary_tables.py
```

The current mapping is Figure1: ten-seed GraphMAE2 factorial trajectory;
Figure2: Reddit coefficient response; Figure3: dominance-performance relationship;
Figure4: gamma/remasking sensitivity; Figure5: MaskGAE dose response;
AppendixD.1: PCGrad conflict histogram. Figures2/3/4/D.1 remain unchanged.
The superseded three-seed Figure1 is preserved under
`figures/historical/three_seed_trajectory/`, not reused as current Figure1.
See `FIGURE_MAP.json` and `FIGURE_NUMBERING_AUDIT.md`.

## Measurement provenance

The earlier compact diagnostic and the later finalized diagnostic are related
but not interchangeable. The finalized degree-stratified estimator uses 16
deterministically selected complete batches per checkpoint and two saved
checkpoints. Its point estimate is the pooled median over 32 records, with a
10,000-resample deterministic cell-preserving bootstrap. The package reports
the finalized estimates as the authoritative Arxiv and Reddit references.

See `protocols/measurement_rules.md` for which rules were fixed before the
corresponding measurement and which were formalized later during read-only
adjudication.

## Configuration provenance

`code/graphmae2/configs/ogbn-arxiv_upstream_default.yaml` is the unmodified
upstream default. It contains `max_epoch: 60` and `batch_size: 512`.

`paper_records/ogbn-arxiv_execution_record.json` is a reader-facing paper
execution record derived from retained protocol, schedule, and completion
evidence. It records batch size 128 and 12 actual epochs (711 updates per epoch;
8,532 total). The retained `max_epoch=60` governed the upstream scheduler
horizon; a dedicated runner stopped after 12 epochs.

`paper_records/reddit_execution_record.json` records retained effective Reddit
settings for a resource-adapted protocol ported from the verified Arxiv
model/optimizer configuration. It was not an official GraphMAE2 Reddit config.

A **runtime config** is a flat YAML accepted by `--config_path`. A **paper
execution record** is nested provenance evidence and is not a runnable
GraphMAE2 configuration. The loader rejects both released paper records rather
than silently retaining argument defaults. It does not flatten or convert them.
The original standalone Arxiv and Reddit paper runtime YAML files were not
retained.

## Dependencies

- `requirements-analysis.txt`: processed-record figures and summaries.
- `requirements-graphmae2.txt`: imports used by the attributed core.
- `requirements-optional.txt`: optional upstream logging and LC preprocessing.

Only PyTorch 2.11.0+cu128, CUDA 12.8, and DGL 2.5 were retained as exact
principal versions of the historical experiment environment. The release
acceptance environment was Python 3.9.25, PyTorch 2.1.0+cu118, DGL 2.2.1, and
OGB 1.3.6; it is not claimed to be the historical training environment. Other
historical package versions are intentionally not invented.

## Maintainer-only release assembly

These commands are for constructing a new release, not reviewer verification:

```bash
python tests/generate_manifest.py
python tests/build_release_zip.py --output ../graphmae2-dominance-diagnostics-r2.zip
python tests/verify_release.py --zip ../graphmae2-dominance-diagnostics-r2.zip
```

The ZIP builder emits POSIX `/` member paths and excludes rerun outputs.
For this expanded candidate, run `python tests/generate_sync_manifest.py` before
`python tests/generate_manifest.py`; the former creates the current
`RELEASE_CONTENT_MANIFEST.csv`. The unchanged SOTA-specific manifest in the root
is an earlier publication snapshot, not the integrity manifest for this candidate.

## Licenses

The attributed GraphMAE2 subset remains under the upstream MIT license in
`LICENSE`. Reader-facing diagnostics, plotting, tables, packaging, and
verification utilities are offered under `LICENSE-AUTHORS`. See
`THIRD_PARTY_NOTICES.md` for provenance details.
