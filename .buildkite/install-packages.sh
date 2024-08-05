#!/bin/bash

CONDA_ENV_NAME="buildkite"
PYTHON_VERSION=3.10

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

set -xe 

pip install -r requirements.txt
pip install -r requirements-test.txt

cd third_party/torchac_cuda
pip install -e .
cd -
pip install -e .
