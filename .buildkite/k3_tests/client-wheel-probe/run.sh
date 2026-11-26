#!/usr/bin/env bash
set -euo pipefail

backend="${1:?Expected cuda, rocm, or musa}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
script="$root/.buildkite/k3_tests/client-wheel-probe"
out="$root/client-wheel-evidence/$backend"
mkdir -p "$out"

if [[ "${2:-}" != inside ]]; then
    case "$backend" in
        rocm)
            image=vllm/vllm-openai-rocm:v0.26.0
            source "$root/.buildkite/scripts/pick-free-gpu-amd.sh" 16000 1
            devices=(--device /dev/kfd --device /dev/dri --group-add video
                     -e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
            ;;
        musa)
            image="${MUSA_CI_IMAGE:-sh-harbor.mthreads.com/ai-kv/kuae-lmcache-vllm:20260819-kuae-ssd-e2e-full-tests-musa-aiter-ipc}"
            devices=(-e MUSA_VISIBLE_DEVICES=0)
            ;;
        *) exit 2 ;;
    esac
    docker_cmd=(docker)
    if ! docker info >/dev/null 2>&1; then
        docker_cmd=(sudo -n docker)
    fi
    "${docker_cmd[@]}" pull "$image"
    "${docker_cmd[@]}" image inspect --format '{{index .RepoDigests 0}}' "$image" | tee "$out/image.txt"
    "${docker_cmd[@]}" run --rm --shm-size=4g "${devices[@]}" \
        -v "$root:/checkout:ro" -v "$out:/evidence" -w /tmp \
        --entrypoint bash "$image" /checkout/.buildkite/k3_tests/client-wheel-probe/run.sh "$backend" inside
    exit
fi

if [[ -d /evidence ]]; then
    out=/evidence
fi
exec > >(tee "$out/probe.log") 2>&1
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
unset PYTHONPATH
export LMCACHE_TRACK_USAGE=false PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO

python3 -VV
python3 -c 'import sys, torch, vllm; print("runtime", sys.executable, torch.__version__, torch.version.cuda, getattr(torch.version, "hip", None), vllm.__version__, "cxx11_abi", torch.compiled_with_cxx11_abi())'
wheel=lmcache_client-0.5.9rc102-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
curl -fLsS --retry 3 "https://github.com/maobaolong/LMCache/releases/download/client-wheel-6972-probe/$wheel" -o "$wheel"
echo "7410c6efb42cb987dca6e788f830d98746f00405d5fd49e142ec8f8f68ab734f  $wheel" | sha256sum -c
python3 -m venv --system-site-packages worker
echo 'STAGE: install original client wheel'
worker/bin/python -m pip install --no-deps "$wheel"
echo 'STAGE: resolve client dependencies using the existing vendor PyTorch'
torch_version="$(worker/bin/python -c 'import torch; print(torch.__version__)')"
echo "torch==$torch_version" > torch-constraint.txt
worker/bin/python -m pip install -c torch-constraint.txt "$wheel"
echo 'STAGE: verify client package and native import'
worker/bin/python -I - <<'PY'
from importlib import metadata, util
from pathlib import Path
import torch
import lmcache
import lmcache.lmcache_native as native
client = metadata.distribution("lmcache-client")
package = Path(client.locate_file("lmcache")).resolve()
assert Path(lmcache.__file__).resolve().parent == package
assert Path(native.__file__).resolve().parent == package
assert util.find_spec("lmcache.cuda_ops") is None
assert util.find_spec("lmcache.v1.multiprocess") is None
print("worker package:", client.version, native.__file__)
PY

echo 'STAGE: build separate full server at the #6972 revision'
curl -fLsS --retry 3 https://codeload.github.com/LMCache/LMCache/tar.gz/9625de6dcb071735e679bdccff516a3acdab4b78 -o source.tar.gz
mkdir source
tar -xzf source.tar.gz -C source --strip-components=1
python3 -m venv --system-site-packages server
server/bin/python -m pip install -c torch-constraint.txt -r source/requirements/build.txt
NO_GPU_EXT=1 MAX_JOBS=2 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.5.9rc102 \
    server/bin/python -m pip install -c torch-constraint.txt --no-build-isolation ./source
echo 'STAGE: vLLM + LMCache store, lookup hit, retrieve'
worker/bin/python -I "$script/round_trip.py" "$backend" "$tmp/server/bin/lmcache" "$out"
