# MaskGAE / PubMed Reproduction

This is CPU scalar postprocessing, not a model-training or probe launcher.
From the repository root:

```sh
python -m pip install -r experiments/maskgae_pubmed/requirements.txt
python -B code/analysis/maskgae_pubmed.py --repo-root . --outdir experiments/maskgae_pubmed/reproduced
python -B -m unittest discover -s tests -p test_maskgae_pubmed_release.py -v
```

The Python interface is `reproduce(repo_root, outdir) -> dict` in
`code/analysis/maskgae_pubmed.py`. It checks all record identities and writes
`metrics.json`, `dose_summary.csv`, `paired_contrasts.csv`, `seed_metrics.csv`
and `run_clipping.csv`. Output files are created exclusively; use a fresh
output directory for another reproduction. No Torch, graph/model loader or
label provider is imported by the analysis.

## Statistics

- Compute each seed's accuracy-trajectory AUC from the full seven-checkpoint
  trapezoidal sum divided by 485. This is not ROC-AUC. Epoch500 accuracy is
  reported separately, never replaced by a selected checkpoint.
- Compute each seed's median of its seven full-Z ratios, then the median of
  those ten seed medians. Do not pool all 70 ratios into one median.
- Use ten matched pretraining seeds for contrasts against alpha0.001.
  Difference units are percentage points. Intervals are ordinary two-sided
  95% Student-t intervals with df9, not Bonferroni intervals.
- Alpha0.5 versus native is the separate historical contrast. The three newer
  doses form a Holm3 family for AUC; a separate Holm3 family is reported for
  the exploratory secondary epoch500 endpoint. Neither family includes the
  historical half-weight contrast. The CIs are nominal, not multiplicity
  adjusted. Zero paired sample SD produces a non-estimable t test, not a
  fabricated p-value.
- All three newer doses were chosen after the historical results were visible.
  Shared historical controls are not an independent replication. No equivalence,
  plateau threshold, best-dose selection or independent causal mediation claim
  follows from these analyses.

## Inspection Code And Attribution

`code/experiments/maskgae_weighting.py` shows joint-objective and full-parameter
gradient routing and the clipping coefficient. There is no encoder-only gate.
`code/measurements/maskgae_components.py` shows full-Z component arithmetic and
the two same-forward VJP calls. These are newly authored inspection utilities,
not the executed training runner and not a bitwise numerical replacement for
FP32 autograd. They construct no models and include no graph or label loader.

The inspected upstream implementation is
[EdisonLeeeee/MaskGAE](https://github.com/EdisonLeeeee/MaskGAE), commit
`343ced7028681f052b7e9ea47945be7fc5166213`, by Jintang Li and collaborators,
associated with *What's Behind the Mask: Understanding Masked Graph Modeling
for Graph Autoencoders*. Upstream `maskgae/model.py`, `mask.py` and `loss.py`
match the executed source bindings. `executed_protocol.json` describes the
actual model, optimizer, masking, losses and evaluator configuration.

**License boundary:** no license file was found in the inspected upstream
commit's tracked file tree or in the frozen source evidence. Its redistribution
license is therefore unverified here. No upstream MaskGAE implementation is
vendored in this addition, and no license is inferred from a different project.
The existing release terms for author-written utilities apply to the new
inspection/analysis code, not to unverified third-party sources. This addition
does not claim to be a standalone licensed full-training distribution.

## Environment Evidence

Execution enforced Python3.10.12, PyTorch2.11.0+cu128 and CUDA12.8. Frozen
package-record bindings identify PyG2.7.0, torch-cluster1.6.3+pt211cu128,
torch-scatter2.1.2+pt211cu128 and torch-sparse0.6.18+pt211cu128. The historical
reference environment metadata records NumPy1.26.4 and SciPy1.15.3; these two
values are reference observations, not separate fresh version observations for
every newer trajectory. Scalar release tests use Python3.13.7 and SciPy1.16.2.
No scientific rerun is needed for the CPU analysis.
