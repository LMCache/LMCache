#!/bin/bash

CONDA_ENV_NAME="buildkite"
PYTHON_VERSION=3.10

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

pip uninstall -y lmcache
pip uninstall -y torchac_cuda
