# Offline Diagnostic Overhead Protocol

The finalized Arxiv measurement used two saved checkpoints with 16 deterministic degree-stratified gradient diagnostic batches per checkpoint: **32 gradient batches** total. A deterministic, cell-preserving **10,000-resample bootstrap** used the frozen 32-row diagnostic record on CPU.

`SUMMARY.json` reports 10.518766246037558 seconds of controlled compute: 10.44726499603712 seconds of synchronized gradient extraction plus 0.07150125000043772 seconds of bootstrap. This excludes model/checkpoint loading and bootstrap-input loading. No optimizer step, validation probe, or test access occurred. Bootstrap resamples are CPU postprocessing, not 10,000 additional GPU backward passes.

This is **not** end-to-end diagnostic wall time and is **not** the historical runtime of the earlier three-batch coefficient-generation procedure.
