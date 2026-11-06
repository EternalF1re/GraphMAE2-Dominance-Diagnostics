# Scalar Geometry Records

- `measurements.csv`: the complete 120-record grid, with measured full-Z and
  all-128-target matrix norms/cosines, support counts, coefficient metadata, and
  the source's explicitly derived routed ratio. `batch_position` is zero-based.
- `identities.json`: matching diagnostic checkpoint/model/teacher/mode hashes,
  batch/root/target/feature/topology hashes, and RNG/mask/remask hashes. No node
  lists, random states, tensors, paths, or Schedule payloads are included.
- `provenance.json`: public file hashes, source archive/member-manifest hashes,
  executed scientific-source hashes, grid definition, and attribution.
- `DATA_DICTIONARY.md`: every CSV column and nested identity/provenance JSON
  field, with units and full-Z/target/local/routed definitions for manuscript Table 3.

The gradient norms are representation-gradient norms, not parameter-gradient
norms. `weighted_aux_gradient_norm` and `target_weighted_aux_gradient_norm`
include the local coefficient 10 in both groups. Target norms/cosines include
zero rows rather than restricting to the shared nonzero subset. Raw component
norms and support counts are retained even though the summary tables use means.

`source_record_sha256` identifies the original measurement JSON bytes, not the
reformatted CSV row or sanitized identity JSON. The source archive and its
operational payload are not required for public scalar-table reproduction.
See `experiments/gradient_geometry/README.md` for definitions and limitations.
