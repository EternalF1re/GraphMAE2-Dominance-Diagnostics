# MaskGAE / PubMed Data Dictionary

The four canonical scientific CSVs contain headers and finite numeric scalar
fields, with no missing values. Accuracy is a **fraction**, not a percent; contrasts in reproduced
tables use percentage points. No row contains labels, node identifiers, graph
edges, features, embeddings, parameter tensors or per-node predictions.

## Identity And Source Conventions

- The pretraining seeds are the ten integers 82001 through 82010. Joint alpha
  is one of 0.001, 0.125, 0.25, 0.5 or 1.0. Each dose has ten matched seeds.
- The seven checkpoint epochs are 15, 42, 83, 125, 208, 333 and 500.
- The classifier seeds 54321, 54322 and 54323 are nested replicates within a
  checkpoint. They do not increase the independent sample size beyond ten.
- Source fields below refer to the saved completed role records. Safe relative
  archive selectors and exact source JSON hashes are public in
  `source_record_map.csv`; machine-specific archive locations remain private.
- Alpha0.001 and alpha0.5 training/probe records are unchanged historical
  trajectories. The other three doses are the newer exploratory trajectories.
  Native-weight ratios use the checkpoint-restored read-only measurements;
  half-weight ratios reuse their original seven-point diagnostics.

## checkpoints.csv

Exactly 350 rows. Unique key: (`pretraining_seed`, `alpha`, `epoch`).

| Column | Type / Unit | Definition And Saved Source |
| --- | --- | --- |
| `pretraining_seed` | integer | Matched pretraining RNG seed; source role `seed`. |
| `alpha` | float | Joint coefficient of degree loss, affecting encoder and degree head; diagnostic `alpha`. |
| `epoch` | integer | Completed training checkpoint epoch; diagnostic `epoch` and probe `epoch` agree. |
| `validation_accuracy` | fraction in [0,1] | Probe-role `Y_j`; arithmetic mean of the three `probes[].best_accuracy` values. |
| `preclipping_R_Z` | nonnegative ratio | Diagnostic `R`: `weighted_degree_Z_norm / rec_Z_norm`, not a parameter-gradient ratio. |
| `rec_Z_norm` | nonnegative L2 norm | Diagnostic `rec_Z_norm`: norm of `dL_edge/dZ_full`, with denominator strictly greater than 1e-12 in these completed records. |
| `weighted_degree_Z_norm` | nonnegative L2 norm | Diagnostic `weighted_degree_Z_norm`: norm of `d(alpha*L_degree)/dZ_full`. Alpha is already included once. |
| `full_z_rows` | integer, 19717 | First coordinate of diagnostic `full_z_shape`; every node row is included, not only edge endpoints or positive-degree targets. |
| `full_z_columns` | integer, 64 | Second coordinate of diagnostic `full_z_shape`; shared encoder output width, not the cat192 probe embedding width. |
| `diagnostic_seed` | integer | Saved diagnostic RNG seed, exactly `910000000 + 1000 * pretraining_seed + epoch`. |
| `probe_54321` | accuracy fraction | Saved `best_accuracy` for classifier seed 54321; matches the corresponding long-form probe row. |
| `probe_54322` | accuracy fraction | Saved `best_accuracy` for classifier seed 54322. |
| `probe_54323` | accuracy fraction | Saved `best_accuracy` for classifier seed 54323. |

Both gradient components use one unchanged diagnostic forward in native train
mode, on an isolated model clone with live state and RNG restored. FP32
autograd gradients are converted to FP64 for norm accumulation. Measurements
precede clipping and use the first edge minibatch. No degree weight is applied
a second time during scalar analysis. A denominator at or below 1e-12 would be
undefined, not zero; every exported completed diagnostic was defined.

## probes.csv

Exactly 1050 rows. Unique key: (`pretraining_seed`, `alpha`, `epoch`, `probe_seed`).

| Column | Type / Unit | Definition And Saved Source |
| --- | --- | --- |
| `pretraining_seed` | integer | Parent pretraining seed; source probe-role `seed`. |
| `alpha` | float | Parent joint-alpha policy. Historical native alpha is resolved from the frozen native run identity where its older record omitted an explicit alpha field. |
| `epoch` | integer | Parent pretraining checkpoint epoch, not a classifier training step. |
| `probe_seed` | integer | `probes[].seed`, exactly one of the three frozen classifier seeds. |
| `best_accuracy` | fraction in [0,1] | `probes[].best_accuracy`, exactly `best_correct / 500`. |
| `best_correct` | integer, 0 through 500 | `probes[].best.correct`; maximum integer validation correct-count over the saved 100-point curve. This is an aggregate count, not a label vector. |
| `best_epoch` | integer, 1 through 100 | `probes[].best.epoch`; earliest classifier epoch attaining `best_correct`. |
| `completed_steps` | integer, 100 | `probes[].completed_classifier_steps`; checked against attempted steps and the ordered 100-point curve. One classifier minibatch per step uses all 60 training examples. |

The export checked maxima and earliest ties against every original curve.
Only these aggregate best values are distributed. Validation nodes number 500;
test labels are not used. Checkpoint accuracy averages three best-validation
probe results; it is not the best pretraining checkpoint and not test accuracy.

## training_updates.csv

Exactly 25000 rows. Unique key: (`pretraining_seed`, `alpha`, `epoch`, `minibatch`).
Every completed run here has one observed native minibatch per epoch.

| Column | Type / Unit | Definition And Saved Source |
| --- | --- | --- |
| `pretraining_seed` | integer | Source training-role `seed`. |
| `alpha` | float | Executed joint coefficient, not an encoder-only routing scale. |
| `epoch` | integer, 1 through 500 | Source `epochs[].epoch`, complete and ordered for each run. |
| `minibatch` | zero-based integer | `epochs[].minibatches[].index`; observed value is zero in all these completed trajectories. |
| `preclip_norm` | nonnegative L2 norm | Saved global norm returned by native `clip_grad_norm_` over all model parameters before clipping. Not the full-Z diagnostic norm. |
| `clip_coefficient` | float in [0,1] | Saved native FP32 `min(1, 1/(preclip_norm + 1e-6))`. The reproducer allows FP32 rounding in this scalar relation. |
| `clip_triggered` | integer 0 or 1 | Exported as `int(preclip_norm > 1)`, the frozen reporting definition. It is not inferred from rounded manuscript percentages or from coefficient-below-one. |
| `edge_loss` | nonnegative scalar | Saved minibatch `edge_loss`, separately preserved before the joint scalar adds the degree objective. Positive and negative mean binary cross-entropies are summed. |
| `degree_loss_raw` | nonnegative scalar | Saved minibatch unweighted `degree_loss_raw`; full-node mean squared error against masked incoming-edge degree counts. |
| `joint_loss` | scalar | Saved minibatch `joint_loss`, checked against `edge_loss + alpha * degree_loss_raw` with native FP32 rounding tolerance. |

The epsilon can make `clip_coefficient < 1` even at a norm exactly one; that is
why clipping frequency is defined by the strict norm threshold. Updates use
Adam with coupled decay, not AdamW. No scheduler or optimizer update is
introduced by this release's postprocessing.

## run_counts.csv

Exactly 50 rows. Unique key: (`pretraining_seed`, `alpha`).

| Column | Type / Unit | Definition And Saved Source |
| --- | --- | --- |
| `pretraining_seed` | integer | Source completed training seed. |
| `alpha` | float | Source executed joint-alpha policy. |
| `completed_epochs` | integer, 500 | Training-role `completed_epochs`, verified against all 500 ordered saved epoch records. |
| `training_forwards` | integer, 500 | Training-role `counts.training_forwards`; excludes diagnostic forwards and downstream embedding extraction. |
| `backwards` | integer, 500 | Training-role `counts.backwards`; excludes the two component VJPs per diagnostic. |
| `optimizer_updates` | integer, 500 | Training-role `counts.optimizer_steps`, checked against the number of actual completed minibatches and epoch update counts. |
| `clipped_updates` | integer, 0 through 500 | Count of that run's actual minibatches with `preclip_norm > 1`; recomputed from `training_updates.csv`. |

Dose clipping percent is `100 * sum(clipped_updates) / sum(optimizer_updates)`.
Every dose has denominator 5000. The exact numerators are 0, 0, 57, 682 and
3382 in ascending alpha order, giving 0%, 0%, 1.14%, 13.64% and 67.64%.
Epoch counts are not silently substituted for optimizer-update counts.

## source_record_map.csv

This separate provenance table contains 750 rows: 50 whole-training-record
mappings, 350 checkpoint-probe mappings and 350 checkpoint-diagnostic mappings.
Its unique key is (`pretraining_seed`, `alpha`, `epoch`, `role`). There are 410
distinct original JSON members; the same training JSON contains many epochs
and also contains the non-native diagnostics, so source hashes repeat normally.
The map does not change any canonical scalar table or add scientific results.

| Column | Type / Unit | Definition |
| --- | --- | --- |
| `pretraining_seed` | integer | Scientific pretraining seed, matching the canonical records. |
| `alpha` | float | Scientific joint-alpha policy, matching the canonical records. |
| `epoch` | integer | Actual checkpoint epoch for `probe` and `diagnostic`. For `train` only, zero is a **whole-run source selector**, not a measured epoch or an extra checkpoint. |
| `role` | text | Exactly `train`, `probe` or `diagnostic`. These describe scientific record sources, not authorization or OS-policy artifacts. |
| `source_archive_member` | relative POSIX path | Exact, unrenamed member name in the source archive. Only completed train/probe/native-read-only `result.json` members are selected. No absolute paths, dated attempt paths, authorization files or OS/approval records are mapped. |
| `source_json_pointer` | text | RFC6901-style pointer within the decoded JSON. Empty means the entire JSON object for `train` or `probe`; `/diagnostics/i` selects the diagnostic object at zero-based index `i`. This empty pointer is intentional, not missing provenance. |
| `source_json_sha256` | 64-character hexadecimal text | SHA256 of the original JSON member's exact uncompressed bytes, **before** parsing or normalization. It is not a hash of the selected subobject, a reformatted copy or the CSV row. |

Join rules:

- `run_counts.csv` and every row of `training_updates.csv` join the `train`
  source by pretraining seed and alpha, using the whole-run epoch-zero selector.
  Select `epochs[epoch-1]` and its minibatch index for an individual update;
  use `counts` and `completed_epochs` for run totals.
- Each `checkpoints.csv` row joins two same-seed/same-alpha/same-epoch entries:
  `probe` supplies its three accuracies and checkpoint mean, while `diagnostic`
  supplies its ratio, primitive norms, full-Z shape and diagnostic seed.
- Each `probes.csv` row joins the corresponding `probe` mapping, then selects
  the nested `probes` object whose `seed` equals `probe_seed`.
- Native-weight diagnostic selectors address the checkpoint-restored read-only
  result, not the older training record's slightly different ratio arithmetic.
  For every other dose the selector addresses `diagnostics` in the training JSON.

`source_archive_sha256.txt` contains the 64-character SHA256 of the **outer
compressed source ZIP bytes**, followed by a newline. It binds all relative
member selectors to one exact source archive; it is separate from per-member
hashes. Check this outer digest first, then hash the selected ZIP member's
uncompressed bytes, then resolve the JSON pointer. Every mapped member digest
was also checked against that archive's existing member manifest. Only the map
and digest file are distributed here, not the operational source JSON payloads.

## executed_protocol.json

The configuration is at `experiments/maskgae_pubmed/executed_protocol.json`.
It documents the executed bindings, not an automatically runnable training
configuration. Null scheduler means no scheduler; it is not missing data.

| Key / Section | Meaning And Source |
| --- | --- |
| `model`, `dataset` | MaskGAE and PubMed, from the completed execution's frozen identity. |
| `upstream` | Official repository URL, exact commit, explicit unverified redistribution-license status, and the fact that no upstream implementation is vendored. |
| `data` | Source loader assertions: 19717 nodes, 500 FP32 feature columns, int64 edge index of shape 2 x88648, preserved row order, transductive unlabeled pretraining. No data arrays are distributed. |
| `pretraining` | Executed seed/dose grid, 500 epochs, joint objective, batch cap65536, 500 observed updates per run, coupled Adam hyperparameters, no scheduler, global clip1 with epsilon1e-6. These are executed research settings, not a claim about every upstream script's CLI defaults. |
| `pretraining.encoder` | Frozen model factory: two GCNConv layers, widths500/128/64, dropout0.5, batch normalization, ELU. |
| `pretraining.edge_decoder` | Frozen factory and upstream decoder: endpoint elementwise product, widths64/32/1, two layers, dropout0.5, ReLU hidden activation, logits for training. |
| `pretraining.degree_decoder` | Frozen factory and upstream decoder: widths64/32/1, two layers, dropout0.5, ReLU including output activation. |
| `pretraining.mask` | Frozen MaskPath constructor: p0.7, node starts, walk length3. The walk overlaps mean p is not asserted to equal an exact fraction of removed unique edges. |
| `pretraining.negative_sampler` | Upstream default PyG negative sampler, with self-loops added to the original edge index before sampling and one negative per masked edge. |
| `pretraining.edge_loss`, `pretraining.degree_loss` | Upstream `ce_loss` and full-node degree-MSE objective; the global alpha multiplies only the degree term. |
| `measurement` | Seven checkpoints, full19717x64 Z, same-forward weighted component VJPs, first minibatch, diagnostic seed formula, denominator cutoff, clone/live-RNG restoration, FP64 norm accumulation and two-level median summary. |
| `evaluation` | Frozen downstream eval/cat192 embedding and detached rowwise L2 normalization with eps1e-12; train60/valid500 only; classifier192-to3; three seeds; 100 steps each; batch cap512; dedicated CPU shuffle generator; coupled Adam .01/decay.001; mean CE; maximum-correct/earliest-tie and lowest-class-index argmax rules. |
| `executed_runtime_bindings` | Python3.10.12, PyTorch2.11.0+cu128 and CUDA12.8 enforced by worker assertions; PyG/compiled-extension versions from frozen package-record bindings; backend readback fixes cuDNN flags and one thread. |
| `reference_environment_observations` | NumPy1.26.4 and SciPy1.15.3 from saved historical environment metadata. These are not fresh observations for every newer run; the boolean records that limitation explicitly. |
| `statistics` | Independent pretraining seeds n10, paired df9, AUC divisor485, ordinary nominal95% t CIs; separate Holm3 families for the three new-dose AUC and exploratory secondary epoch500 contrasts; historical alpha0.5 excluded; post-outcome exploratory disclosure. |

## Attribution, Verification And License Gap

The upstream source is [EdisonLeeeee/MaskGAE](https://github.com/EdisonLeeeee/MaskGAE),
commit `343ced7028681f052b7e9ea47945be7fc5166213`, by Jintang Li and
collaborators. The inspected files, byte-matched to executed source bindings,
are [maskgae/model.py](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/maskgae/model.py),
[maskgae/mask.py](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/maskgae/mask.py),
and [maskgae/loss.py](https://github.com/EdisonLeeeee/MaskGAE/blob/343ced7028681f052b7e9ea47945be7fc5166213/maskgae/loss.py).
Relevant symbols are `MaskGAE.train_step`, `GNNEncoder.get_embedding`,
`EdgeDecoder`, `DegreeDecoder`, `MaskPath` and `ce_loss`.

The completed-source archive and all its member checksums were verified.
Historical training/probe bytes were independently matched to their original
completed archive members, excluding failed-execution records. Exported scalar
data were also compared with the authors' existing scalar-analysis records.

**License gap:** the inspected upstream commit's tracked tree contains no
license file, and the frozen source evidence supplies none. Redistribution
rights for the full upstream implementation are therefore not asserted. This
addition contains only newly authored inspection/analysis code and scalar
exports; it does not copy the upstream MaskGAE implementation or inherit its
license from GraphMAE2. Existing author-utility release terms govern the new
utilities, not unverified third-party sources. No network lookup was used to
fill this gap with an assumption.
