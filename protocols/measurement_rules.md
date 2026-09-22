# Measurement Rules and Provenance

This file describes the rules used in the reported analyses. It does not claim
that every rule was fixed before the earliest diagnostic rows were collected.

## Shared-target dominance measurement

The reconstruction-loss and latent-loss gradients are measured with respect to
the same online-encoder representation and restricted to the same target/root
nodes. The reported dominance ratio is

`||lambda * g_latent||_2 / ||g_reconstruction||_2`.

**Provenance status:** used by the reported checkpoint diagnostics and retained
in the released diagnostic implementation.

## Later degree-stratified sampling

Candidate units are complete batches from a frozen first-epoch training
schedule. Candidate batches overlapping the earlier diagnostic target lists are
excluded. Each candidate is scored by the median full-graph in-degree of its
target/root nodes. Candidate batches are sorted by that value, with training
schedule position as the tie-breaker, and divided into four approximately
equal-count rank groups. Four complete batches per group are selected by a
deterministic SHA-256 ranking.

The groups are rank partitions, not fixed numeric degree intervals. Target/root
batch sizes are 128 for ogbn-arxiv and 64 for Reddit. The same target lists are
reused at both saved checkpoints.

**Provenance status:** target lists and sampling rule were fixed before the
corresponding expanded checkpoint measurement and stability decision.

## Stability panels

Within each degree group, the four selected batch hashes are sorted. Positions
0 and 2 form panel A; positions 1 and 3 form panel B. Panel assignment does not
affect target selection.

**Provenance status:** formalized later during read-only adjudication of the
already frozen 16-batch record universe.

## Uncertainty calculation

The point estimate is the pooled median over 32 checkpoint-by-batch dominance
ratios. The reported interval uses 10,000 deterministic bootstrap resamples,
preserving the record count in each saved-checkpoint by degree-group cell.

**Provenance status:** applied and formalized later in read-only adjudication;
the bootstrap does not alter the frozen diagnostic measurements.

## Interpretation boundary

The earlier three-batch and later 16-batch procedures changed several factors
at once. The released records support sensitivity to the combined sampling and
estimation design, not a causal claim that degree stratification alone produced
the observed estimate change.
