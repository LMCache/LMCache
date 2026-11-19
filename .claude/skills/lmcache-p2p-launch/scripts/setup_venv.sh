#!/bin/bash
# Create a fresh uv venv with vLLM + LMCache (+ nixl) for P2P testing.
# usage: setup_venv.sh --dir DIR [--vllm 0.32.0] [--lmcache pypi|source] [--ref dev] [--repo URL] [--arch 9.0]
#   --lmcache pypi   : uv pip install "lmcache[nixl]"            (release wheels, no build)
#   --lmcache source : git clone --branch REF, editable CUDA build (~2 min on a 200-core box)
set -euo pipefail
DIR=""; VLLM=0.32.0; MODE=source; REF=dev; REPO=https://github.com/LMCache/LMCache.git; ARCH=${TORCH_CUDA_ARCH_LIST:-9.0}
while [ $# -gt 0 ]; do case "$1" in
  --dir) DIR=$2; shift 2;; --vllm) VLLM=$2; shift 2;; --lmcache) MODE=$2; shift 2;; --ref) REF=$2; shift 2;;
  --repo) REPO=$2; shift 2;; --arch) ARCH=$2; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$DIR" ] || { echo "--dir is required" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "== installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; }
mkdir -p "$DIR" && cd "$DIR"
echo "== [$(date)] venv"; uv venv --python 3.12 .venv >/dev/null; source .venv/bin/activate
echo "== [$(date)] vllm==$VLLM (pulls torch)"; uv pip install "vllm==$VLLM"
if [ "$MODE" = pypi ]; then
  echo "== [$(date)] lmcache[nixl] from PyPI"; uv pip install "lmcache[nixl]"
else
  [ -d LMCache ] || git clone --branch "$REF" "$REPO" LMCache
  (cd LMCache && git log --oneline -1)
  echo "== [$(date)] lmcache build deps"; uv pip install -r LMCache/requirements/build.txt
  echo "== [$(date)] lmcache editable CUDA build"
  MAX_JOBS=${MAX_JOBS:-32} TORCH_CUDA_ARCH_LIST="$ARCH" CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH" \
    uv pip install -e ./LMCache --no-build-isolation
  echo "== [$(date)] nixl"; uv pip install -r LMCache/requirements/nixl.txt
fi
echo "== [$(date)] sanity"
python - <<'PY'
import torch, vllm, lmcache, nixl  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__, "| lmcache", lmcache.__version__)
import lmcache.cuda_ops  # noqa: F401
print("lmcache.cuda_ops OK")
PY
echo "p2p flags in lmcache server: $(lmcache server --help | grep -c -- '--p2p')"
echo "== [$(date)] DONE: activate with 'source $DIR/.venv/bin/activate'"
