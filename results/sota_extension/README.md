# Recent Baseline Extension Records

`aligned_mtl/` and `famo/` each contain per-seed normalized validation AUCs, seven checkpoint validation means per seed, and direct R-hat records at the same seed/checkpoint grid. The AUCs are over checkpoint updates 250 through 8532, not final-checkpoint accuracy.

`direct_rhat/` contains the complete two-method checkpoint table and its frozen summary. `overhead/` and `diagnostic_overhead/` contain separate controlled resource measurements; their timings are not historical full-run wall times. `nash_failure/` contains failure provenance only.

The AUC values were transcribed from the frozen matched-seed adjudication. The checkpoint validation rows retain the mean and standard deviation from each frozen three-probe result after omitting private checkpoint and temporary-embedding paths; their seven-point AUCs were independently checked against the frozen per-seed values. Per-method R-hat tables are row-preserving subsets of the complete frozen C.6 table. The C.6 files and ten-method overhead CSV are byte-identical copies of their frozen counterparts. No model evaluation, training, or new bootstrap was performed to prepare these records.

The full historical checkpoints, schedules, and training adapter are not part of this compact public release. Do not interpret the incomplete Nash attempt as a completed comparator or a general result about the author implementation.
