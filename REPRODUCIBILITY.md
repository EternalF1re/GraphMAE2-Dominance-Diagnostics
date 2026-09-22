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
| Figure 1 | `figures/figure_1/plot_validation_trajectory.py` | `processed_records/arxiv/validation_trajectory.csv` | `figures/figure_1/figure_1_reference.pdf` | `figures/figure_1/reproduced/figure_1.pdf` |
| Figure 2 | `figures/figure_2/plot_dominance_performance.py` | Arxiv response and cross-dataset records | `figures/figure_2/figure_2_reference.pdf` | `figures/figure_2/reproduced/figure_2.pdf` |
| Figure 3 | `figures/figure_3/plot_protocol_parameters.py` | `processed_records/arxiv/protocol_parameter_response.csv` | `figures/figure_3/figure_3_reference.pdf` | `figures/figure_3/reproduced/figure_3.pdf` |
| Figure 4 | `figures/figure_4/plot_reddit_replication.py` | `processed_records/reddit/replication.csv` | `figures/figure_4/figure_4_reference.pdf` | `figures/figure_4/reproduced/figure_4.pdf` |
| Appendix conflicts | `figures/appendix_conflicts/plot_pcgrad_conflicts.py` | PCGrad histogram bins | `figures/appendix_conflicts/appendix_conflicts_reference.pdf` | `figures/appendix_conflicts/reproduced/appendix_conflicts.pdf` |

Rerun directories are excluded from `MANIFEST.sha256` and release ZIPs. The
verifier checks canonical input equality and frozen reference hashes, not PDF
binary equality across plotting backends.

## Tables

`tables/generate_summary_tables.py` deterministically creates:

- `weighting_method_summary.csv` from the released per-seed method record; and
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
The complete regression suite runs GraphMAE2 CLI `--help` commands and therefore
does require DGL, OGB, and the other declared GraphMAE2 runtime dependencies.

## Release integrity

For an official extracted ZIP, basic verification is
`python tests/verify_release.py`; it does not require DGL or OGB. The full
regression command `python -m unittest tests.test_r1_release -v` additionally
requires both dependency groups because it audits public GraphMAE2 CLI help.
The frozen manifest detects missing, additional, and modified tracked files.
`tests/generate_manifest.py` is a maintainer-only assembly command and must not
precede official-release verification.
