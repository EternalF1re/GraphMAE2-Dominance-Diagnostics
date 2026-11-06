# Gradient Geometry Data Dictionary

These processed records support manuscript **Table 3 (gradient geometry)**.
The dataset contains exactly 120 measurements: Protocol A, two groups, ten
pretraining seeds, three checkpoint updates, and two fixed batch positions.
One CSV row joins one identity object by `(group, seed, update, batch_position)`.
There are ten independent seed units; the 120 measurements are repeated states
and batches, not 120 independent training replicates.

## Mathematical Conventions And Units

Let `Z` be the complete online encoder representation matrix for the sampled
graph, with `full_z_rows` rows and 1024 columns. Let `T` be the ordered set of
128 training-target row indices. Define the locally measured components
`A = d L_rec / d Z` and `B = d (10 * L_lat) / d Z`. `L_rec` is the SUM of three
native remask reconstruction losses; `L_lat` is the bare auxiliary latent loss.
The local auxiliary coefficient is 10 in BOTH groups.

Full-Z norms are Frobenius norms of the complete matrices A and B. Target norms
are Frobenius norms of `A[T]` and `B[T]`, retaining ALL 128 rows including zeros.
Cosines flatten the corresponding complete matrices; they are neither means of
nodewise cosines nor cosines after filtering to shared nonzero rows. Norm units
are loss per representation-coordinate unit; no physical unit is assigned.
Ratios, cosines, fractions, and coefficients are dimensionless. Angles are in
degrees. Counts and indices are integers. All numeric scalar fields are finite
in this release; JSON hashes are lowercase, 64-character SHA-256 strings.

The measured full-Z local ratio is `R_local = ||B||_F / ||A||_F`. Training
coefficients `(E,H)` are `(10,10)` for G1 and `(.3,10)` for G2. The routed ratio
is DERIVED as `(E/H) * R_local`, separately for each group's own measured
norms: unchanged for G1 and `.03 * G2_local` for G2. It is not a newly measured
routed VJP, clipped ratio, parameter-gradient ratio, or optimizer-update ratio.
There is no direct measurement of a bare, unweighted auxiliary gradient norm.

## Measurements CSV

File: `measurements.csv`. Types are `str`, `int`, `float`, and lowercase CSV
`true`/`false` for booleans. Columns below follow the file's exact order.

| Column | Type / Units | Definition |
|---|---|---|
| `group` | str | Paper group G1 or G2; coefficients are defined above. |
| `seed` | int, identifier | Pretraining seed, 0 through 9. |
| `update` | int, optimizer-update count | Saved checkpoint after update 250, 3555, or 8532. This is not a diagnostic optimizer step. |
| `batch_position` | int, zero-based index | Fixed training-batch position 93 or 118; matched across groups and checkpoints for the same seed. No batch payload is included. |
| `loss_reconstruction` | float, loss units | Native sum of the three reconstruction criterion outputs, before an encoder/head routing intervention. |
| `loss_latent_bare` | float, loss units | Bare auxiliary latent criterion output, before multiplication by the local coefficient 10. |
| `reconstruction_gradient_norm` | float, loss / representation unit | Measured full-Z `||A||_F`. This is not a parameter-gradient norm. |
| `weighted_aux_gradient_norm` | float, weighted loss / representation unit | Measured full-Z `||B||_F`, including local coefficient 10 in both groups; not multiplied again by E. |
| `weighted_ratio` | float, dimensionless | Measured-local full-Z ratio `||B||_F / ||A||_F`, using this same row's norms. |
| `routed_weighted_ratio_derived` | float, dimensionless | Source-record derived full-Z ratio `(E/H) * weighted_ratio`; not a separate gradient measurement. |
| `routed_scale_derived` | float, dimensionless | E/H: 1 for G1, .03 for G2. |
| `cosine` | float, dimensionless, [-1,1] | Full-Z flattened-matrix cosine `sum(A*B)/(||A||_F*||B||_F)`. Zero rows remain present. |
| `angle_degrees` | float, degrees | `degrees(acos(cosine))` for the full-Z matrices, not a mean of node angles. |
| `full_z_rows` | int, row count | Number of online-Z rows in the entire sampled graph, including context and target rows. |
| `hidden_dimension` | int, coordinate count | Number of columns of Z, 1024. |
| `reconstruction_nonzero_row_fraction` | float, fraction [0,1] | Fraction of all Z rows with at least one exactly nonzero entry in A. |
| `auxiliary_nonzero_row_fraction` | float, fraction [0,1] | Fraction of all Z rows with at least one exactly nonzero entry in B. |
| `shared_nonzero_row_fraction` | float, fraction [0,1] | Fraction of all Z rows nonzero in BOTH A and B; a descriptive support statistic, not a filter used for cosine. |
| `shared_target_cosine` | float, dimensionless, [-1,1] | Supplementary flattened cosine over `A[T]` and `B[T]`, using all 128 target rows. Despite the name, it is NOT restricted to shared nonzero rows. |
| `target_count` | int, row count | Number of ordered target rows T, exactly 128, including rows whose reconstruction gradient is zero. |
| `target_reconstruction_gradient_norm` | float, loss / representation unit | Measured target-matrix `||A[T]||_F`, retaining all target rows. |
| `target_weighted_aux_gradient_norm` | float, weighted loss / representation unit | Measured target-matrix `||B[T]||_F`, including local coefficient 10; no E/H rescaling. |
| `target_reconstruction_fullZ_norm_fraction` | float, dimensionless | `||A[T]||_F / ||A||_F`; a norm ratio, NOT a squared-energy fraction or row fraction. |
| `target_aux_fullZ_norm_fraction` | float, dimensionless | `||B[T]||_F / ||B||_F`; a norm ratio. It equals 1 in these measurements because the auxiliary representation gradient has no non-target leakage. |
| `fullZ_reconstruction_nonzero_rows` | int, row count | Number of all Z rows having any nonzero coordinate in A. |
| `fullZ_aux_nonzero_rows` | int, row count | Number of all Z rows having any nonzero coordinate in B. |
| `fullZ_shared_nonzero_rows` | int, row count | Number of all Z rows nonzero in both A and B. |
| `target_reconstruction_nonzero_rows` | int, row count, 0..128 | Number of target rows nonzero in A[T]; the remaining target rows are still retained in matrix geometry. |
| `target_aux_nonzero_rows` | int, row count, 0..128 | Number of target rows nonzero in B[T]. |
| `target_shared_nonzero_rows` | int, row count, 0..128 | Number of target rows nonzero in both components; NOT the denominator of target cosine or target norms. |
| `defined` | bool | Source full-Z geometry is defined: both full-Z component norms are positive. All released records are true. |
| `target_defined` | bool | Source all-target geometry is defined: both target-matrix component norms are positive. All released records are true. |
| `model_effective_lambda` | float, dimensionless | Local auxiliary objective coefficient in the measurement forward, 10 for BOTH groups; it is not G2's routed encoder coefficient .3. |
| `training_shared_coefficient` | float, dimensionless | Training encoder/shared-path coefficient E: 10 for G1 and .3 for G2. Metadata describing the saved training policy, not an additional diagnostic multiplication. |
| `training_head_coefficient` | float, dimensionless | Training auxiliary-head coefficient H, 10 for both groups. |
| `epoch` | int, zero-based epoch index | Checkpoint epoch metadata: `(update-1)//711`, giving 0, 4, and 11. Not elapsed time. |
| `ema_updates` | int, update count | Cumulative teacher EMA updates at the checkpoint, zero for these states. |
| `preclip_only` | bool | Geometry is before clipping; no diagnostic joint backward, clipping ratio, or optimizer update was measured. |
| `parameter_gradients_unchanged` | bool | Source component-VJP purity check reported unchanged parameter .grad state. It does not mean parameter gradients were measured here. |
| `vjp_purity` | bool | Source checks reported unchanged protected state across the two component VJPs. This flag is retained evidence, not a new model-level check performed by public analysis. |
| `model_teacher_unchanged` | bool | Source checks reported unchanged model/teacher state for the measurement. The public analysis does not reload those states. |
| `paired_stochastic_identity` | bool | Source checks reported matched replayed input/RNG/mask identities across G1/G2 at the same seed/update/batch position. The released hashes also permit their equality to be checked. |

## Identity JSON

File: `identities.json`. Paths below are relative to the document root; `[]`
denotes each array item. Hashes are identities of unavailable payloads, not
payloads themselves, reconstruction instructions, or evidence of model-level
re-execution by the public analysis. Different hash recipes are not assumed
interchangeable. No individual root/node IDs or RNG states are released.

| JSON Field Path | Type / Units | Definition |
|---|---|---|
| `schema_version` | int | Public identity schema version, 1. |
| `record_count` | int, record count | Number of identity objects, 120. |
| `identities` | array of objects | Exactly one object for each measurement key. |
| `identities[].group` | str | Same G1/G2 identifier as the CSV. |
| `identities[].seed` | int, identifier | Same pretraining seed as the CSV. |
| `identities[].update` | int, update count | Same checkpoint update as the CSV. |
| `identities[].batch_position` | int, zero-based index | Same fixed batch position as the CSV. |
| `identities[].source_record_sha256` | SHA-256 | SHA of the ORIGINAL scalar measurement JSON member bytes, not the reformatted public CSV row or this sanitized object. |
| `identities[].data_fingerprints` | object | Per-batch unlabeled input identities and graph counts, without any payload. |
| `identities[].data_fingerprints.nodes_sha256` | SHA-256 | Ordered sampled graph-node tensor fingerprint. No node list is released. |
| `identities[].data_fingerprints.roots_sha256` | SHA-256 | Ordered global target/root-node tensor fingerprint. |
| `identities[].data_fingerprints.targets_sha256` | SHA-256 | Ordered local target-index tensor T fingerprint; distinct from global root-node identities. |
| `identities[].data_fingerprints.feature_sha256` | SHA-256 | Sampled feature tensor fingerprint; no features are released. |
| `identities[].data_fingerprints.graph` | object | Sampled graph before the two edge-drop views. Ordered edges are identified by hashes. |
| `identities[].data_fingerprints.graph.nodes` | int, node count | Number of nodes in the sampled graph. |
| `identities[].data_fingerprints.graph.edges` | int, edge count | Number of edges in the sampled graph. |
| `identities[].data_fingerprints.graph.src_sha256` | SHA-256 | Fingerprint of the source-index tensor in native edge order. |
| `identities[].data_fingerprints.graph.dst_sha256` | SHA-256 | Fingerprint of the destination-index tensor in the same edge order. |
| `identities[].data_fingerprints.drop_graph1` | object | First edge-drop graph view used by the native forward. |
| `identities[].data_fingerprints.drop_graph1.nodes` | int, node count | Number of nodes in the first edge-drop view. |
| `identities[].data_fingerprints.drop_graph1.edges` | int, edge count | Number of edges in the first edge-drop view. |
| `identities[].data_fingerprints.drop_graph1.src_sha256` | SHA-256 | Ordered source-index tensor fingerprint for the first view. |
| `identities[].data_fingerprints.drop_graph1.dst_sha256` | SHA-256 | Ordered destination-index tensor fingerprint for the first view. |
| `identities[].data_fingerprints.drop_graph2` | object | Second edge-drop graph view used by the native forward. |
| `identities[].data_fingerprints.drop_graph2.nodes` | int, node count | Number of nodes in the second edge-drop view. |
| `identities[].data_fingerprints.drop_graph2.edges` | int, edge count | Number of edges in the second edge-drop view. |
| `identities[].data_fingerprints.drop_graph2.src_sha256` | SHA-256 | Ordered source-index tensor fingerprint for the second view. |
| `identities[].data_fingerprints.drop_graph2.dst_sha256` | SHA-256 | Ordered destination-index tensor fingerprint for the second view. |
| `identities[].model_state_sha256` | SHA-256 | Source semantic fingerprint of the loaded model state mapping, distinct from checkpoint-file bytes. |
| `identities[].pre_forward_rng_sha256` | SHA-256 | Source fingerprint of active RNG state immediately before the measurement forward. |
| `identities[].post_forward_rng_sha256` | SHA-256 | Source fingerprint of active RNG state immediately after the measurement forward. |
| `identities[].mask_sha256` | SHA-256 | Native masked-index tensor fingerprint. |
| `identities[].remask_sha256` | array of three SHA-256 strings | Ordered fingerprints of the three native remask-index tensors. The array preserves remask-call order. |
| `identities[].checkpoint_sha256` | SHA-256 | Hash of original checkpoint FILE bytes, not merely its model state mapping. No checkpoint is distributed. |
| `identities[].teacher_state_sha256` | SHA-256 | Source semantic fingerprint of the EMA encoder/projector state subset. |
| `identities[].module_modes_sha256` | SHA-256 | Fingerprint of source module names/types/train-mode metadata. |
| `identities[].batch_hash` | SHA-256 | Frozen batch-identity digest retained from the recorded batch entry, not a CSV-row hash. |
| `identities[].root_ids_sha256` | SHA-256 | Ordered global root/target-node tensor fingerprint; agrees with data_fingerprints.roots_sha256. |
| `identities[].frozen_rng_sha256` | SHA-256 | Frozen pre-collation RNG identity from the recorded batch entry; not the same stage as pre_forward_rng_sha256. |
| `identities[].data_sha256` | SHA-256 | Source digest of the complete data_fingerprints object, not a hash of this public JSON file. |

Tensor fingerprints in the executed capture include dtype, shape and contiguous
tensor bytes. Semantic state/RNG fingerprints follow the source's canonical
state hashing. File hashes identify original file bytes. Public-table recreation
does not require any preimage, but identity hashes alone cannot re-create VJPs.

## Provenance JSON

File: `provenance.json`. Source hashes identify original archived source bytes.
They are not hashes of the newly wrapped public inspection module. The shared
release manifest, maintained separately, handles complete public-file integrity.

| JSON Field Path | Type / Units | Definition |
|---|---|---|
| `schema_version` | int | Public provenance schema version, 1. |
| `record_count` | int, record count | Complete scalar measurement count, 120. |
| `independent_seed_count` | int, seed count | Ten equally weighted pretraining seeds, not 120 independent observations. |
| `groups` | object | Group-specific training coefficients. |
| `groups.G1` | object | G1 coefficient metadata. |
| `groups.G1.encoder_coefficient` | float, dimensionless | E=10, encoder/shared auxiliary path coefficient. |
| `groups.G1.head_coefficient` | float, dimensionless | H=10, auxiliary-head coefficient. |
| `groups.G2` | object | G2 coefficient metadata. |
| `groups.G2.encoder_coefficient` | float, dimensionless | E=.3, encoder/shared auxiliary path coefficient. |
| `groups.G2.head_coefficient` | float, dimensionless | H=10, auxiliary-head coefficient. |
| `updates` | array of int, update counts | Fixed checkpoint grid [250,3555,8532]. |
| `batch_positions` | array of int, zero-based indices | Fixed batch-position grid [93,118]. |
| `target_rows_per_measurement` | int, row count | All 128 ordered target rows are used, including zeros. |
| `local_auxiliary_coefficient` | float, dimensionless | Diagnostic B is differentiated from 10 times bare auxiliary loss in both groups. |
| `scalar_record_files` | object | Public scalar input filenames mapped to their SHA-256 file-byte hashes. |
| `scalar_record_files["measurements.csv"]` | SHA-256 | Hash of the public measurements.csv bytes. |
| `scalar_record_files["identities.json"]` | SHA-256 | Hash of the public identities.json bytes. |
| `source_archive_sha256` | SHA-256 | Hash of the verified original completed measurement archive; no private archive path is published. |
| `source_archive_manifest_sha256` | SHA-256 | Hash of the original archive's member-hash manifest bytes. |
| `source_excerpts_sha256` | object | Semantic labels mapped to original executed scientific SOURCE MODULE hashes. Not hashes of individual excerpt blocks or public wrapper files. |
| `source_excerpts_sha256.component_capture` | SHA-256 | Original executed native component-capture module hash. |
| `source_excerpts_sha256.full_z_metrics` | SHA-256 | Original executed full-Z geometry module hash. |
| `source_excerpts_sha256.target_metrics` | SHA-256 | Original executed all-target geometry module hash. |
| `scope` | str | Explicit scalar/hash-only scope; excluded scientific payload categories. |
| `source_record_hashes` | str | Clarifies original measurement-member byte hashes versus reformatted public records. |
| `attribution` | str | Study-authored LICENSE-AUTHORS terms and GraphMAE2 MIT scientific context. Shared attribution documents are maintained separately. |

## Table Recreation And Interpretation

For each scalar metric, average two batches within seed/group/checkpoint, then
three checkpoint means within seed/group, then the ten seed means. Report
sample SD across those ten seed means, with `ddof=1`. A mean of per-record norm
ratios is not a ratio of mean norms. The approximately 20.32-fold routed contrast
is the ratio of the two group means, not a mean of ten paired seed ratios.

Full-Z cosine and all-target cosine are different reported quantities. Near-zero
aggregate full-Z cosine is not strict per-node orthogonality, correctly paired
population inference, a parameter-gradient statement, or a causal explanation.
No CI, p-value, bootstrap, probe, clipping measurement, or new experiment is
produced by the public reproduction code. Complete-design reproduction fails
on missing, duplicate, undefined or nonfinite records instead of selecting seeds.
