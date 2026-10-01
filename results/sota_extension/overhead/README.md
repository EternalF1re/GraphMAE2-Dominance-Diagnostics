# Online Benchmark Results

`OVERHEAD_SUMMARY.csv` contains all ten fixed-order methods. Its `mean_ms_per_step`, `peak_allocated_gib`, and `peak_reserved_gib` columns are the resource comparisons. All rows completed 50 warm-up and 300 measured steps on the same GPU, with the same model, batch size, and seed-0 schedule. The benchmark has no validation or test probe and must not be merged with historical full-run wall times.
