# SOTA Baseline Provenance

These are locally adapted comparisons, not unmodified executions of the author repositories. All three methods received the raw reconstruction and latent losses in that order. Settings were frozen before the new pretraining runs; there was no dataset-specific hyperparameter search for these methods.

| Method | Paper | Author repository and inspected commit | Local integration |
| --- | --- | --- | --- |
| Aligned-MTL | Senushkin et al., *Independent Component Alignment for Multi-Task Learning*, CVPR 2023 | <https://github.com/taohong08/MTL>, `3f577f1365101fdd16094313b4a3b0d3845ce10d` | `amtl` with minimum-eigenvalue scaling, shared GNN-encoder gradients, and the common GraphMAE2 gradient-clipping policy. Its published direct R-hat is measured before the parameter-space alignment transform. |
| FAMO | Liu et al., *FAMO: Fast Adaptive Multitask Optimization*, NeurIPS 2023 | <https://github.com/Cranial-XIX/FAMO>, `0a6ad06dbdd55fce73ec01e3a175c2498cd1e5e9` | Recommended `gamma=0.01` and auxiliary Adam learning rate `0.025`, with same-batch post-step loss replay and RNG restoration. The recorded direct R-hat uses the positive multipliers actually applied to that update. |
| Nash-MTL | Navon et al., *Multi-Task Learning as a Bargaining Game*, ICML 2022 | <https://github.com/AvivNavon/nash-mtl>, `cce18403ef5557a6b30f4ba43b896117107d6902` | An attempted two-task adapter using ECOS and shared GNN-encoder gradients. The frozen formal runs did not complete. A local pre-solve guard differs from the pinned author solver; no completed AUC or general Nash-MTL performance claim is made. |

The inspected commits identify provenance, not a guarantee that those upstream repositories reproduce this adaptation unchanged. This compact public release contains processed records and protocol documentation, not the complete historical SOTA training adapter or private batch schedules.
