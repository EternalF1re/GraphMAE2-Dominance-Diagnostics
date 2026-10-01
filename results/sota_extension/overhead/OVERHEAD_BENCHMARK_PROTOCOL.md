# Controlled Online-Training Overhead Protocol

Ten methods were measured in a fixed order on one GPU with the same initialized GraphMAE2 model, batch size 128, frozen seed-0 batch/RNG schedule, and method-specific frozen settings. Each method used **50 warm-up** optimizer steps followed by **300 measured** steps. CUDA was synchronized around every measured step; warm-up was excluded from the reported measured-step time. Peak memory counters were reset after warm-up.

`OVERHEAD_SUMMARY.csv` reports total measured seconds, mean/median/p95 milliseconds per step, peak allocated GiB, and peak reserved GiB. These are controlled benchmark measurements, not end-to-end training durations and not downstream scores. The Nash benchmark's 300 completed steps do not make its separate frozen formal pretraining a completed comparator.
