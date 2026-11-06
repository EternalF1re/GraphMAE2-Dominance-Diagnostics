# Figure 1: ten-seed GraphMAE2 factorial trajectories

Run `python figures/figure_1/plot_factorial_trajectory.py` from the repository root.
The default output is `reproduced/figure_1.pdf` inside this directory; reference
PDF bytes are never overwritten. Inputs match the canonical factorial CSV.
Use a new output PDF for every rerun. Protected repository locations and all
existing destination files are rejected before rendering; temporary sidecars
are isolated from canonical evidence.

Only G1=(E10,H10) and G3=(E0.3,H0.3) are plotted, each with ten pretraining
seeds and three nested probe seeds at seven real optimizer-update distances.
Both panels show y-axis numbers, with the same range. Error bars are sample SD,
not confidence intervals. PanelA=global clip; PanelB=groupwise clip.

The superseded three-seed Figure1 was not deleted from the evidence:
its original script, PDF and inputs are retained under
`figures/historical/three_seed_trajectory/`. Do not pool its absolute AUC with
this independently executed ten-seed factorial protocol.
