#!/usr/bin/env bash
# Run AMD unit tests with kernel serialization.
set -euo pipefail

export AMD_SERIALIZE_KERNEL=1
echo "AMD kernel mode: serialized"
echo "$PWD" # for debugging

uv venv --python 3.12 ".venv-${BUILDKITE_BUILD_ID}"
# shellcheck disable=SC1090 # Buildkite generates a unique virtualenv path per build.
source ".venv-${BUILDKITE_BUILD_ID}/bin/activate"
uv pip install --upgrade pip setuptools wheel

# shellcheck disable=SC1091 # The repository script is present in every checkout.
source .buildkite/scripts/pick-free-gpu-amd.sh 18000
export PYTORCH_ROCM_ARCH="gfx942"
export TORCH_DONT_CHECK_COMPILER_ABI=1
export CXX=hipcc
export BUILD_WITH_HIP=1
uv pip install "torch==2.11.0+rocm7.2" torchvision \
    --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install -r requirements/rocm_core.txt

uv pip install -r requirements/common.txt
uv pip install -r requirements/test.txt
uv pip install -e . --no-build-isolation
uv pip freeze

LMCACHE_TRACK_USAGE="false" \
pytest --maxfail=1 --cov=lmcache \
    --cov-report term --cov-report=html:coverage-test \
    --cov-report=xml:coverage-test.xml --html=durations/test.html \
    --ignore=tests/disagg --ignore=tests/v1/test_pos_kernels.py \
    --ignore=tests/v1/test_nixl_batched_contains.py \
    --ignore=tests/v1/test_device_id_race.py \
    --ignore=tests/v1/test_nixl_multipath.py \
    --ignore=tests/skipped \
    --ignore=tests/v1/storage_backend/test_eic.py

cat << EOF | buildkite-agent annotate --style "info"
  Read the <a href="artifact://coverage-test/index.html">uploaded coverage report</a>
EOF
