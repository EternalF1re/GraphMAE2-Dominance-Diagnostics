# Nash-MTL Attempt: Incomplete Formal Comparator

Nash-MTL was attempted under the same frozen comparison protocol, but seeds 0, 1, and 2 stopped after 707, 629, and 8 completed optimizer updates, respectively. None reached the seven-checkpoint grid, so no formal validation-trajectory AUC, direct C.6 comparison, or performance ranking is published for Nash-MTL.

The frozen local adapter raised a nonpositive Gram-product pre-solve guard. That guard is absent from the pinned author implementation; the historical failing Gram matrices were not retained. This is evidence of a **local integration-specific behavioral difference**, not evidence that Nash-MTL generally fails or is unstable. The 300-step overhead benchmark is a separate controlled run and does not repair the incomplete formal comparator. A later local engineering revision is not substituted for these frozen results.

`failure_provenance.csv` reports only the frozen seed, completed update count, error class/message, and no-test-access flags. It intentionally excludes private paths, tracebacks, checkpoint tensors, and any post-hoc successful revision.
