# Paper Execution Records

These JSON files are reader-facing execution and provenance records assembled
only from retained evidence. They are not historical standalone GraphMAE2
runtime configurations.

A paper execution record is not a runnable GraphMAE2 configuration. Do not pass
these files to `--config_path`; the runtime loader rejects their nested record
schema. The original standalone Arxiv and Reddit paper runtime YAML files were
not retained.

The only runtime configuration distributed in `code/graphmae2/configs/` is the
byte-preserved upstream/default flat YAML. Use an independently retained flat
GraphMAE2 runtime YAML for any explicit `--config_path` workflow.
