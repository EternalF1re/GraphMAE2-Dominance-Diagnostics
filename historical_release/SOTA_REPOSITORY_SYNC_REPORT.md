# SOTA Repository Synchronization Report

## Added files (22)

- `.gitattributes`: preserves exact bytes of all release files across Git checkouts.
- `experiments/sota_extension/SOTA_BASELINE_PROTOCOL.md`
- `experiments/sota_extension/SOTA_BASELINE_PROVENANCE.md`
- `results/sota_extension/README.md`
- `results/sota_extension/aligned_mtl/checkpoint_rhat.csv`
- `results/sota_extension/aligned_mtl/checkpoint_validation.csv`
- `results/sota_extension/aligned_mtl/per_seed_auc.csv`
- `results/sota_extension/diagnostic_overhead/DIAGNOSTIC_OVERHEAD_PROTOCOL.md`
- `results/sota_extension/diagnostic_overhead/SUMMARY.json`
- `results/sota_extension/direct_rhat/C6_checkpoint_level_rhat.csv`
- `results/sota_extension/direct_rhat/C6_summary.json`
- `results/sota_extension/direct_rhat/README.md`
- `results/sota_extension/famo/checkpoint_rhat.csv`
- `results/sota_extension/famo/checkpoint_validation.csv`
- `results/sota_extension/famo/per_seed_auc.csv`
- `results/sota_extension/nash_failure/NASH_FAILURE_README.md`
- `results/sota_extension/nash_failure/failure_provenance.csv`
- `results/sota_extension/overhead/OVERHEAD_BENCHMARK_PROTOCOL.md`
- `results/sota_extension/overhead/OVERHEAD_SUMMARY.csv`
- `results/sota_extension/overhead/README.md`
- `SOTA_REPOSITORY_SYNC_MANIFEST.csv`
- `SOTA_REPOSITORY_SYNC_REPORT.md`

## Modified files (12)

- `README.md`: adds the recent baseline extension and its analysis-level boundary.
- `REPRODUCIBILITY.md`: maps the extension's public processed records and limitations.
- `MANIFEST.sha256`: regenerated for the extended release.
- `figures/figure_1/input/validation_trajectory.csv`
- `figures/figure_2/input/protocol_parameter_response.csv`
- `figures/figure_4/input/replication.csv`
- `processed_records/arxiv/protocol_parameter_response.csv`
- `processed_records/arxiv/reference_diagnostic_records.csv`
- `processed_records/arxiv/validation_trajectory.csv`
- `processed_records/method_comparisons/weighting_methods.csv`
- `processed_records/reddit/reference_diagnostic_records.csv`
- `processed_records/reddit/replication.csv`

The nine existing CSV files above are **Git serialization corrections only**: their prior Git blobs had LF line endings while the previously released R2 files already had CRLF. The new Git blobs will match those R2 release files byte-for-byte. No CSV field, value, row, or frozen source artifact was changed.

## Removed files

None.

## Scientific integrity

All pre-existing release working-copy scientific files and values are unchanged. The nine pre-existing CSV Git blobs listed above only receive their already-published R2 line endings so a clean checkout matches the release manifest. The released C.6 checkpoint CSV and summary JSON, and the ten-method overhead CSV, are byte-identical to their frozen sources. The 42 checkpoint-validation rows and six AUC rows are direct, path-free projections of frozen records; all six seven-checkpoint AUCs were checked against the frozen per-seed summary. Nash-MTL remains an incomplete formal comparator with three frozen failure counts, not a ranked method.

## Manuscript and availability consistency

The requested manuscript wording says that source code, processed records, and reproducibility materials are publicly available. This release contains the existing diagnostic/GraphMAE2 code, new processed extension records, and protocol/provenance documentation, so that statement is supportable **for analysis-level reproduction**. The complete historical SOTA training adapter, private schedules, checkpoints, and raw traces are not public here; the statement must not be read as a claim of end-to-end retraining. The actual manuscript TeX was not present in this workspace for verbatim checking and was not edited.

## Privacy and package checks

The public release verifier scans paths and text for local absolute paths, private usernames, secrets, internal run labels, unsafe binary artifacts, and manifest drift. The extended release and fresh-extracted ZIP passed these checks. `SOTA_REPOSITORY_SYNC_MANIFEST.csv` inventories the new extension files, `.gitattributes`, two updated documentation files, nine Git-serialization-corrected CSV files, and this report with byte size, SHA256, and category. It excludes itself and the self-excluding root `MANIFEST.sha256` to avoid circular hashes. The root manifest covers the complete release, including the synchronization report and CSV manifest. The `.gitattributes` rule keeps all release bytes stable in Git.
