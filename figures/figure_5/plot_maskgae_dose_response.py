"""Reproduce Figure5 from five-dose PubMed scalar records only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reproduced")/"figure_5.pdf")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/"code/analysis")); sys.path.insert(0, str(ROOT/"figures"))
    import revision_render
    revision_render.publication_destination(ROOT,args.output)
    import maskgae_pubmed
    groups = maskgae_pubmed.analyze(maskgae_pubmed.load_records(ROOT))["groups"]
    plot_groups = [dict(r, epoch500_mean_pct=100*r["epoch500_mean"], epoch500_sd_pp=100*r["epoch500_sd"]) for r in groups]
    revision_render.publish_figure(revision_render.mask_figure,plot_groups,ROOT,args.output)
    print("Figure5:five discrete doses; original connecting lines retained; error bars are ten-seed sample SD.")


if __name__ == "__main__": main()
