#!/bin/bash

CONDA_ENV_NAME="buildkite"
PYTHON_VERSION=3.10

exist_env="$(conda env list | grep ${CONDA_ENV_NAME})"
if [[ -n $exist_env ]]; then
    echo "Skipping env creation"
else
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
fi

cuda_version=12.4

cuda_path="/usr/local/cuda-${cuda_version}"

if [[ -d "$cuda_path" ]]; then
    echo "Found CUDA ${cuda_version} at ${cuda_path}"
else
    echo "❌ CUDA ${cuda_version} not found at ${cuda_path}"
    exit 1
fi

export CUDA_HOME=/usr/local/cuda-${cuda_version}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

set -xe

# Clear vLLM cache
echo "Clearing vLLM cache..."
rm -rf ~/.cache/vllm/
mkdir -p ~/.cache/vllm/

# Clear Python cache files
echo "Clearing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

pip uninstall -y vllm # need to repair it
pip install -r .buildkite/mmlu_scripts/mmlu_requirements.txt

# Patch vllm api_server.py to add engine_client assignment
API_SERVER_FILE=$(python -c "import vllm.entrypoints.api_server as m; print(m.__file__)" | tail -n 1)

if [[ -f "$API_SERVER_FILE" ]]; then
    # Preload torch at the top of the vllm API server
    sed -i '1i import torch; print("✅ Preloaded torch inside vLLM api_server")' "$API_SERVER_FILE"

    # Patch the line after the usage_context=UsageContext.API_SERVER)) line
    sed -i '/usage_context=UsageContext.API_SERVER))/a \    app.state.engine_client = engine' "$API_SERVER_FILE"
    echo "✅ Patched $API_SERVER_FILE"
else
    echo "❌ Could not find vllm.api_server.py"
    exit 1
fi

set +x
echo "Current env:"
pip freeze