# Checkpoint-Level Direct R-hat

`C6_checkpoint_level_rhat.csv` has 21 rows per completed method: three pretraining seeds times seven checkpoints. Each row records three direct diagnostic batch values and their median, so the unit for the headline method range and median is a **seed-by-checkpoint median**, not a pooled raw batch. `C6_summary.json` gives the minimum, median, and maximum of those 21 checkpoint medians separately from its explicitly labeled pooled-raw statistics.

Aligned-MTL is **pre-transformation representation-level dominance**. Its parameter-space alignment transform is not collapsed to an assumed scalar. FAMO is **realized-weight representation-level dominance** using the multipliers applied to the checkpoint-completing optimizer update. These are related but not interchangeable update semantics.
