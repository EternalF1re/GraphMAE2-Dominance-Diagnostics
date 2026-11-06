# Third-Party Notices

## GraphMAE2

Files under `code/graphmae2/` originate from the official THUDM GraphMAE2
implementation under the MIT license in `LICENSE`. The release changes only
packaging-facing behavior in these copied files:

- `utils.py`: release-relative config lookup and optional experiment logging;
- `models/finetune.py`: optional experiment logging when `wandb` is absent;
- `datasets/localclustering.py`: lazy optional preprocessing dependency;
- `config_paths.py`: new release-relative path helper; and
- `paper_records/*_execution_record.json`: new provenance records, not upstream files.

The upstream default configuration was renamed
`ogbn-arxiv_upstream_default.yaml` without changing its contents. Original
third-party copyright notices and license terms are not replaced.

## DGL-derived sampling utility

`code/graphmae2/datasets/saint_sampler.py` is distributed as part of the
attributed GraphMAE2 source subset and contains a sampling implementation based
on DGL APIs. It remains covered by the upstream GraphMAE2 MIT distribution; DGL
itself is a separate dependency distributed under its own upstream license.

## Release-authored utilities

The diagnostics, comparisons, figures, tables, protocols, packaging, and
verification utilities prepared for this research release are distributed
under the MIT terms in `LICENSE-AUTHORS`, with copyright attributed to
`The Authors` for anonymous review.

Datasets, LC artifacts, model checkpoints, pretrained weights, and third-party
Python packages are not redistributed.

## MaskGAE analysis and inspection

The PubMed experiment used the MaskGAE implementation at
https://github.com/EdisonLeeeee/MaskGAE, pinned commit
`343ced7028681f052b7e9ea47945be7fc5166213`.
A redistribution license could not be verified in the inspected pinned source.
Consequently this candidate does NOT vendor the upstream MaskGAE model,
data-loader or third-party package files, and does not assert a redistribution
license on the authors' behalf. Obtain those separately and verify their terms.

The new scalar records, CPU analyses and study-authored inspection excerpts
under `code/analysis/`, `code/experiments/`, `code/measurements/` and the added
experiment documentation are covered by LICENSE-AUTHORS. Inspection excerpts
document the measured scientific computation, not a standalone native runtime.
See `experiments/maskgae_pubmed/README.md` for exact execution/configuration
provenance and the explicit limits of raw-data reproduction.
