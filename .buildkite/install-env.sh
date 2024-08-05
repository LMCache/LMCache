#!/bin/bash

CONDA_ENV_NAME="buildkite"
PYTHON_VERSION=3.10

exist_env="$(conda env list | grep ${CONDA_ENV_NAME})"
if [[ -n $exist_env ]]; then
    echo "Skipping env creation"
else
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
fi
