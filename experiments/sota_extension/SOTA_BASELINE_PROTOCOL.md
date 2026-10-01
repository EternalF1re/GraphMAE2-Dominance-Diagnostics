# SOTA Baseline Comparison Protocol

- Dataset and model: `ogbn-arxiv`, GraphMAE2 with local-clustering ego-graph batches.
- Pretraining seeds: `0`, `1`, `2`; batch size `128`; intended budget `12` epochs and `8532` optimizer updates per completed method.
- Fixed checkpoint updates: `250`, `711`, `1422`, `2133`, `3555`, `5688`, `8532`.
- Validation: split-safe frozen embeddings and linear probes with probe seeds `54321`, `54322`, `54323` at every checkpoint. The checkpoint validation record is the mean and reported standard deviation over these three probes.
- Normalized validation-trajectory AUC: trapezoidal area under the seven checkpoint mean accuracies divided by `8532 - 250`. Per-seed AUCs are released, not only method means.
- Test access: no test split was accessed during this extension's training, checkpoint validation, direct-gradient diagnostics, or overhead benchmarks. The release contains no extension test scores.
- Hyperparameters: author-recommended method settings were fixed in advance; no dataset-specific search or validation-driven retuning of these new methods. Both loss inputs are unscaled reconstruction and latent components.
- Fairness: the same model/teacher initialization, optimizer and scheduler budget, root/LC batch and RNG schedules, masking/remasking rule, gradient-clipping norm, representation extraction, and probe protocol were held fixed across methods. The historical full batch schedule is not shipped.
- Direct R-hat: the ratio is measured on shared target-node representations. Aligned-MTL values are **pre-transformation** representation-level dominance; FAMO values use **realized applied weights**. They must not be described as identical parameter-space update effects.
- Nash-MTL: the formal trajectories stopped early and were excluded from completed-method AUC comparisons; their failure provenance is retained separately.

The public records support analysis-level verification of reported comparisons. They do not constitute a one-command recreation of historical model training.
