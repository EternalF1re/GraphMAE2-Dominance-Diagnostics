"""Reproduce current Figure1 from the ten-seed factorial records, not the old scan."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input")/"checkpoint_validation.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reproduced")/"figure_1.pdf")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/"code/analysis"))
    sys.path.insert(0, str(ROOT/"figures"))
    import revision_render
    revision_render.publication_destination(ROOT,args.output)
    import factorial
    rows = factorial.validate_records(factorial.read_csv(args.input))
    revision_render.publish_figure(revision_render.graph_figure,factorial.checkpoint_summary(rows),ROOT,args.output)
    print("Figure1:80 frozen runs; ten-seed mean+/-sample SD; both panels show numeric y ticks.")


if __name__ == "__main__": main()
