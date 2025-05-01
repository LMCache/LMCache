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
    sed -i '/usage_context=UsageContext.API_SERVER))/a \    app.state.engine_client = engine' "$API_SERVER_FILE"
    echo "✅ Patched $API_SERVER_FILE"
else
    echo "❌ Could not find vllm.api_server.py"
    exit 1
fi

# Completely clean and reinstall lmcache
echo "Reinstalling lmcache with C extensions..."
pip uninstall -y lmcache
rm -rf build/ dist/ *.egg-info/
VERBOSE=1 pip install -e . --no-cache-dir

# Verify C extensions are built correctly
# After installing lmcache
echo "Verifying C extensions..."
python -c "
import lmcache
import os
import sys
import importlib.util

print('lmcache path:', lmcache.__path__)
contents = os.listdir(lmcache.__path__[0])
print('lmcache directory contents:', contents)

# Check for .so file with c_ops in the name
c_ops_files = [f for f in contents if 'c_ops' in f and f.endswith('.so')]
if c_ops_files:
    print('Found C extension files:', c_ops_files)
    # Create a symlink if necessary
    if 'c_ops' not in contents:
        os.symlink(
            os.path.join(lmcache.__path__[0], c_ops_files[0]),
            os.path.join(lmcache.__path__[0], 'c_ops.so')
        )
        print('Created symlink for c_ops.so')
else:
    print('WARNING: No c_ops module found in lmcache package!', file=sys.stderr)
    sys.exit(1)
"

python -c "
import sys
import os
import importlib.util

# Force load PyTorch first to ensure its libraries are in memory
import torch
print(f'PyTorch library path: {os.path.dirname(torch.__file__)}')

# Now try to import lmcache
import lmcache
pkg_dir = lmcache.__path__[0]
so_file = os.path.join(pkg_dir, 'c_ops.so')

if os.path.exists(so_file):
    print(f'File exists: {so_file}')

    # Try importing the normal way now that PyTorch is loaded
    try:
        import lmcache.c_ops
        print('Successfully imported lmcache.c_ops after loading PyTorch')
    except Exception as e:
        print(f'Error during import: {e}')
else:
    print(f'File does not exist: {so_file}')
"

set +x
echo "Current env:"
pip freeze