#!/usr/bin/env bash
# Per-job environment setup for jobs that DON'T need vLLM (e.g. unit tests).
# Installs LMCache from source on top of the ci-base image, which already
# has torch + requirements/cuda.txt + build.txt baked in. Much faster than
# setup-env.sh since it skips the vLLM nightly install entirely.
set -euo pipefail

trap 'echo "ERROR: setup-lmcache-only-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# ── GPU health pre-check ────────────────────────────────────
# Fail fast if GPUs are occupied by stale host processes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80
merge_pr_base_branch

echo "--- :python: Installing LMCache (no vLLM)"
if [[ -n "${XPU_WHEEL_ARTIFACT_ID:-}" ]]; then
    echo "--- :package: Installing verified XPU wheel"
    wheel_path="$(python "${REPO_ROOT}/.buildkite/k3_tests/xpu/download-wheel.py")"
    uv pip install "${wheel_path}"
    rm -f -- "${wheel_path}"
    rmdir -- "$(dirname "${wheel_path}")"
    export PYTHONSAFEPATH=1
    wheel_site_packages="$(python -P -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    # Spawned workers need repo test helpers, but must never import source lmcache.
    test_import_root="$(mktemp -d "${TMPDIR:-/tmp}/lmcache-xpu-test-imports.XXXXXX")"
    ln -s "${REPO_ROOT}/tests" "${test_import_root}/tests"
    ln -s "${REPO_ROOT}/benchmarks" "${test_import_root}/benchmarks"
    ln -s "${REPO_ROOT}/setup_extensions" "${test_import_root}/setup_extensions"
    export PYTHONPATH="${test_import_root}:${wheel_site_packages}"
    python -c 'import lmcache, pathlib, sysconfig; assert pathlib.Path(lmcache.__file__).is_relative_to(sysconfig.get_path("purelib")), lmcache.__file__; print("LMCache loaded from wheel:", lmcache.__file__)'
else
    # Skip setuptools_scm git describe; the repo carries non-PEP-440 tags
    # (nightly, nightly-cu13) that crash the newer vcs_versioning backend.
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
    uv pip install -r requirements/proto.txt
    uv pip install -e . --no-build-isolation

    # Generate ignored bindings for editable installs that skip the build_py hook.
    echo "--- :gear: Generating LMCache gRPC bindings"
    python "${REPO_ROOT}/lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py"
fi

echo "--- :white_check_mark: Environment ready (LMCache only, no vLLM)"
python -c "import lmcache; print('LMCache installed')"
