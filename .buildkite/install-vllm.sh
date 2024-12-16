#!/bin/bash

CONDA_ENV_NAME="buildkite"
PYTHON_VERSION=3.10

cuda_version=12.1
export CUDA_HOME=/usr/local/cuda-${cuda_version}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

git clone https://github.com/LMCache/lmcache-vllm.git lmcache-vllm
cd lmcache-vllm
pip install -e .
cd ..

