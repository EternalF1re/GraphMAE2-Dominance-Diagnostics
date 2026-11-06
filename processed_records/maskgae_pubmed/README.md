# MaskGAE / PubMed Scalar Records

These are scalar exports from completed executions, not simulated results.
No labels, node features, graph edges, model parameters, embeddings, gradients,
signatures or machine-specific paths are distributed here.
Column definitions, units, exact source fields and attribution are in
[DATA_DICTIONARY.md](DATA_DICTIONARY.md).

| File | Rows | Unit |
| --- | ---: | --- |
| `checkpoints.csv` | 350 | dose x pretraining seed x checkpoint |
| `probes.csv` | 1050 | checkpoint x nested classifier seed |
| `training_updates.csv` | 25000 | actual completed optimizer update |
| `run_counts.csv` | 50 | completed pretraining trajectory |

Five joint degree-loss weights are 0.001, 0.125, 0.25, 0.5 and 1.0. Each has
the same ten pretraining seeds, 82001 through 82010, and checkpoints
15, 42, 83, 125, 208, 333 and 500. All recorded trajectories have 500 epochs
and 500 actual Adam updates. One observed masked-edge minibatch was used per
epoch; this is an observed count, not an assumption that one epoch always
equals one update.

`validation_accuracy` is the mean of three best-validation classifier
accuracies, for seeds 54321, 54322 and 54323, each trained for exactly 100
steps. Validation uses 500 nodes; training uses 60. Each probe best value was
checked against its complete saved curve using maximum integer correct count,
with earliest epoch winning ties. Probe replicates are nested within a
pretraining seed, not independent pretraining replicates.

`preclipping_R_Z` is `weighted_degree_Z_norm / rec_Z_norm`. Both VJPs use the
same forward and all 19717 x64 shared-representation entries, before clipping.
Alpha is included once in the numerator. The diagnostic seed is
`910000000 + 1000 * pretraining_seed + epoch`. Native-weight ratios were
measured by restoring the original checkpoints read-only; half-weight ratios
are the existing completed measurements. Raw representation vectors are not
included. All exported ratios were defined with edge norm greater than 1e-12.

Clipping counts use the recorded global preclip norm **strictly greater than
1**, divided by completed optimizer updates. The native coefficient is
`min(1, 1 / (norm + 1e-6))`, computed in FP32 during execution. Its epsilon
means coefficient-below-one is not the definition of the reported trigger.

| Joint alpha | Clipped updates | Completed updates | Percent |
| ---: | ---: | ---: | ---: |
| 0.001 | 0 | 5000 | 0 |
| 0.125 | 0 | 5000 | 0 |
| 0.25 | 57 | 5000 | 1.14 |
| 0.5 | 682 | 5000 | 13.64 |
| 1.0 | 3382 | 5000 | 67.64 |

The scalar exports were verified against the authoritative completed-result
archive and its member checksums. Historical training/probe records were also
matched to their original completed-source archives. Earlier failed-execution
records are not substituted for the completed trajectories. Exact internal
archive locations and member hashes remain outside this public payload.
