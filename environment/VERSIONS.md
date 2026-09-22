# Recorded Experimental Environment

The retained environment records identify these principal versions:

- PyTorch: 2.11.0+cu128
- CUDA runtime: 12.8
- DGL: 2.5

A complete historical package manifest was not retained. The requirements and
environment files therefore describe dependencies but intentionally do not
invent exact versions for packages whose versions were not preserved.

`requirements-analysis.txt` contains the dependencies for record-only table
and figure reproduction. `requirements-graphmae2.txt` contains imports used by
the attributed GraphMAE2 core. `requirements-optional.txt` lists upstream
logging and LC-preprocessing extras that are not required for release
verification or processed-record analysis. Their historical versions were not
retained.

## Release acceptance environment

The public-release acceptance run used Python 3.9.25, PyTorch 2.1.0+cu118,
DGL 2.2.1, and OGB 1.3.6. This acceptance environment is not claimed to be the
historical training environment. It was used only for static checks, CLI help,
and analysis-only reproduction; no model training or dataset evaluation was
performed.
