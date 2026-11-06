# Current figure numbering

| Paper figure | Topic | Script | Canonical input | Reference |
|---|---|---|---|---|
| Figure1 | Ten-seed GraphMAE2 factorial trajectories | `figures/figure_1/plot_factorial_trajectory.py` | `processed_records/factorial/checkpoint_validation.csv` | `figures/figure_1/figure_1_reference.pdf` |
| Figure2 | Reddit coefficient response | `figures/figure_2/plot_reddit_replication.py` | `processed_records/reddit/replication.csv` | `figures/figure_2/figure_2_reference.pdf` |
| Figure3 | Dominance-performance relationship | `figures/figure_3/plot_dominance_performance.py` | Existing Arxiv/cross-dataset records | `figures/figure_3/figure_3_reference.pdf` |
| Figure4 | Gamma/remasking response | `figures/figure_4/plot_protocol_parameters.py` | `processed_records/arxiv/protocol_parameter_response.csv` | `figures/figure_4/figure_4_reference.pdf` |
| Figure5 | MaskGAE PubMed five-dose response | `figures/figure_5/plot_maskgae_dose_response.py` | `processed_records/maskgae_pubmed/checkpoints.csv` | `figures/figure_5/figure_5_reference.pdf` |
| AppendixD.1 | PCGrad pre-projection conflict histogram | `figures/appendix_conflicts/plot_pcgrad_conflicts.py` | Existing conflict histogram | `figures/appendix_conflicts/appendix_conflicts_reference.pdf` |

Figure2/3/4/D.1 bytes, inputs and scripts are unchanged from the exact remote
baseline. They are not renumbered based on an older local checkout.
The superseded three-seed Figure1 is preserved byte-for-byte under
`figures/historical/three_seed_trajectory/`. Its old numbering audit is retained
at `historical_release/figure_numbering_before_revision.md`.

Figure1 keeps both panels' y ticks and actual update spacing. Figure5 retains
the user's approved connecting lines; bars in both new figures are sample SD
across ten pretraining seeds, not confidence intervals. `FIGURE_MAP.json` is
the machine-readable mapping; `figures/reference_provenance.json` binds the
unchanged legacy PDFs.
