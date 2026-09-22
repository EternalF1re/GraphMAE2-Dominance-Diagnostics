# GraphMAE2 Dominance Diagnostics

This repository accompanies a study of loss-component dominance and weighting
in GraphMAE2. It publishes compact processed records, deterministic analysis
utilities, paper-figure scripts, and an attributed subset of the GraphMAE2
implementation for inspection.

## Reproducibility scope

The package supports **analysis-level reproduction from processed records**:

- rechecking the authoritative Arxiv and Reddit reference estimates;
- recomputing the released deterministic bootstrap intervals;
- rebuilding summary tables;
- redrawing the four main figures and the appendix conflict histogram; and
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

The full suite additionally launches every public GraphMAE2 CLI with `--help`.
It therefore requires both the analysis dependencies and the GraphMAE2 runtime
dependencies, including DGL and OGB:

```bash
python -m pip install -r environment/requirements-analysis.txt
python -m pip install -r environment/requirements-graphmae2.txt
python -m unittest tests.test_r1_release -v
```

The full suite still performs no training, inference, dataset download, or
checkpoint loading. Installing only the analysis requirements is not sufficient
for all GraphMAE2 CLI checks.

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
python figures/figure_1/plot_validation_trajectory.py
python figures/figure_2/plot_dominance_performance.py
python figures/figure_3/plot_protocol_parameters.py
python figures/figure_4/plot_reddit_replication.py
python figures/appendix_conflicts/plot_pcgrad_conflicts.py
```

Figure scripts write under each figure's `reproduced/` directory. They never
overwrite the manifest-tracked `*_reference.pdf` files. PDF binary identity is
not treated as a scientific requirement; canonical CSV values and key numeric
claims are verified instead.

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

## Licenses

The attributed GraphMAE2 subset remains under the upstream MIT license in
`LICENSE`. Reader-facing diagnostics, plotting, tables, packaging, and
verification utilities are offered under `LICENSE-AUTHORS`. See
`THIRD_PARTY_NOTICES.md` for provenance details.
