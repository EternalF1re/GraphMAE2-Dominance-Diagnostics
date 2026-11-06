# Figure numbering audit

| paper_figure | repo_directory | plot_script | input_file | reference_pdf | semantic_match | status |
|---|---|---|---|---|---|---|
| Figure 1 | `figures/figure_1` | `plot_validation_trajectory.py` | `input/validation_trajectory.csv` | `figure_1_reference.pdf` | Arxiv validation trajectory | PASS |
| Figure 2 | `figures/figure_2` | `plot_reddit_replication.py` | `input/replication.csv` | `figure_2_reference.pdf` | Reddit coefficient response | PASS |
| Figure 3 | `figures/figure_3` | `plot_dominance_performance.py` | `input/dominance_performance.csv`; `input/cross_dataset_dominance.csv` | `figure_3_reference.pdf` | Dominance-performance relationship | PASS |
| Figure 4 | `figures/figure_4` | `plot_protocol_parameters.py` | `input/protocol_parameter_response.csv` | `figure_4_reference.pdf` | Gamma and K sensitivity | PASS |

Figure 2 and Figure 4 files were moved as complete semantic sets. Only figure numbers, names and references changed; the scientific CSV bytes and PDF bytes are unchanged.
