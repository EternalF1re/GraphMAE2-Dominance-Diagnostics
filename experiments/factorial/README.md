# GraphMAE2 encoder/head factorial experiment

ogbn-arxiv, ten matched pretraining seeds0..9, four conditions, two clipping
protocols,80 completed runs. Actual budget12epochs x711updates=8532; retained
native scheduler horizon60epochs and delayed EMA40 are different quantities.
Checkpoints250,711,1422,2133,3555,5688,8532; no final checkpoint is selected by
validation outcome. Native training batch128, validation probe batch256.

| Group | Encoder E | Auxiliary head H | Native loss coefficient | Branch backward scale |
|---|---:|---:|---:|---:|
| G1 | 10 | 10 | 10 | 1 |
| G2 | 0.3 | 10 | 10 | 0.03 |
| G3 | 0.3 | 0.3 | 0.3 | 1 |
| G4 | 10 | 0.3 | 0.3 | 10/0.3 |

`code/experiments/factorial_weighting.py` contains source-exact branch and
clipping functions. The projector-input hook has identity forward values and
scales only backward gradients by E/H. The native criterion is reconstruction
plus H times latent loss. Original AdamW.step follows native clipping.

ProtocolA globally clips model gradients to norm3. ProtocolB separately clips
disjoint encoder, reconstruction, and latent groups to norm3. Ownership is
exhaustive: enc_mask_token belongs to encoder, dec_mask_token to reconstruction;
encoder/projector EMA parameters are frozen, excluded from trainable groups.
Full ownership and handling of None gradients remain visible in copied code.

Three fixed1000epoch probes (54321,54322,54323) are averaged within each
checkpoint/pretraining seed. They are not30 independent training observations.
Normalized trajectory AUC divides trapezoidal area by8282actual updates.
Each endpoint/protocol has its own three-effect Bonferroni simultaneous95%
family (individual98.333333%, paired Student-t df9); A/B/endpoints are not pooled.
The final8532 analysis is post-hoc, not a replacement of the primary AUC analysis.
Ordinary95% intervals and within-family Holm3 p-values are separately labelled.
No missing/nonsignificant effect is interpreted as equivalence.

`configuration.json` is a source-verified execution record with private paths
removed, NOT a drop-in historical runtime configuration. The canonical original
configuration identity matches all80 completion receipts. Complete matched
initialization and Schedule hashes are in the processed run-identity table.
Private Schedules, graph assets and checkpoints are not redistributed; analysis
reproduction is complete, historical training replay is not claimed.
