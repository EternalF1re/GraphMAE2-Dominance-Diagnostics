# Figure 5: MaskGAE PubMed dose response

Run `python figures/figure_5/plot_maskgae_dose_response.py` from the repository root.
Output is `reproduced/figure_5.pdf` inside this directory; the frozen reference
is not overwritten. The input copy equals the canonical350-checkpoint CSV;
the complete probe/run records are validated before summarizing.
Use a new output PDF for each rerun. All protected source/evidence destinations
and existing output files are rejected before rendering begins.

Both panels display percentages: normalized validation-accuracy trajectory AUC
over epochs15..500 (area divided by485), and epoch500 validation accuracy.
Each point is the mean across ten pretraining seeds; bars show sample SD,
not confidence intervals. Alpha configurations use discrete horizontal positions.
The user's approved connecting lines are retained as visual guides only;
intermediate doses were not measured. Repeated gradient-ratio annotations are
omitted, but remain available in the associated table/data.
