#!/usr/bin/env python3
"""Run the CacheBlend V2 blend E2E on a 2x H100 Modal worker.

This runner is intentionally thin: it clones the requested LMCache PR branch or
commit, runs the same `.buildkite/k3_tests/blend/run.sh` entrypoint used by the
Buildkite job, and copies the resulting evidence bundle into a Modal Volume.

Examples:
    modal run .buildkite/k3_tests/blend/modal_h100_e2e.py --mode smoke
    modal run .buildkite/k3_tests/blend/modal_h100_e2e.py --mode full --commit <sha>

Prerequisites:
    modal secret create hf-token HF_TOKEN=<token>

The default image must provide the same baseline expected by setup-blend-env.sh:
Python 3.12, CUDA, uv, and a default vLLM environment at /opt/venv. Override
with LMCACHE_MODAL_IMAGE if the project image name changes.

If the private Buildkite image is not pullable from Modal, use a public CUDA
image and ask Modal to bootstrap `/opt/venv` at image-build time:

    LMCACHE_MODAL_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04 \
    LMCACHE_MODAL_BOOTSTRAP_VLLM=1 \
    modal run .buildkite/k3_tests/blend/modal_h100_e2e.py --mode smoke
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Literal

import modal

APP_NAME = "lmcache-cacheblend-v2-h100-e2e"
ARTIFACT_VOLUME_NAME = "lmcache-cacheblend-v2-e2e-artifacts"
DEFAULT_REPO = "https://github.com/OCWC22/LMCache.git"
DEFAULT_BRANCH = "ocwc/cacheblend-v2-adapter-integration"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_IMAGE = os.environ.get("LMCACHE_MODAL_IMAGE", "tensormesh/cacheblend:latest")
BOOTSTRAP_VLLM = os.environ.get("LMCACHE_MODAL_BOOTSTRAP_VLLM", "0") == "1"
REQUIRE_HF_SECRET = os.environ.get("LMCACHE_MODAL_REQUIRE_HF_SECRET", "1") != "0"

Mode = Literal["smoke", "full"]

MODE_ENV: dict[Mode, dict[str, str]] = {
    "smoke": {
        "MAX_MODEL_LEN": "2048",
        "LMCACHE_L1_SIZE_GB": "20",
        "LMCACHE_CHUNK_SIZE": "64",
        "SHUFFLE_NUM_DOCUMENTS": "2",
        "SHUFFLE_DOCUMENT_LENGTH": "512",
        "SHUFFLE_OUTPUT_LEN": "64",
        "BENCHMARK_REQUEST_TIMEOUT_SEC": "1800",
        "BENCHMARK_TIMEOUT_SEC": "2400",
        # Strict/prod telemetry gate: the proxy must observe KV-store telemetry
        # before decoder handoff.  Do not enable timeout fallback in CI proof runs.
        "PROXY_TELEMETRY_TIMEOUT_SEC": "300",
        "SERVER_WAIT_TIMEOUT": "900",
    },
    "full": {
        "MAX_MODEL_LEN": "16384",
        "LMCACHE_L1_SIZE_GB": "70",
        "SHUFFLE_NUM_DOCUMENTS": "3",
        "SHUFFLE_DOCUMENT_LENGTH": "1000",
        "SHUFFLE_OUTPUT_LEN": "200",
        "BENCHMARK_REQUEST_TIMEOUT_SEC": "3600",
        "BENCHMARK_TIMEOUT_SEC": "4800",
        "PROXY_TELEMETRY_TIMEOUT_SEC": "300",
    },
}

app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
modal_secrets = [modal.Secret.from_name("hf-token")] if REQUIRE_HF_SECRET else []

image = (
    modal.Image.from_registry(DEFAULT_IMAGE, add_python="3.12")
    .apt_install(
        "bash",
        "ca-certificates",
        "curl",
        "g++",
        "git",
        "ninja-build",
        "procps",
        "tar",
    )
    .pip_install("pyyaml")
)

if BOOTSTRAP_VLLM:
    image = image.run_commands(
        "apt-get update && apt-get install -y cuda-cudart-13-0",
        "uv venv /opt/venv --python /usr/local/bin/python --seed",
        "/opt/venv/bin/python -m pip install -U pip uv",
        "VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 /opt/venv/bin/uv pip install "
        "-p /opt/venv/bin/python -U vllm --pre "
        "--extra-index-url https://wheels.vllm.ai/nightly/cu129 "
        "--extra-index-url https://download.pytorch.org/whl/cu129 "
        "--index-strategy unsafe-best-match",
    )

if not REQUIRE_HF_SECRET:
    # Keep Modal's local and remote imports consistent. Without this, the local
    # app definition can omit the secret while the remote container import sees
    # the default env and thinks the function still depends on hf-token.
    image = image.env({"LMCACHE_MODAL_REQUIRE_HF_SECRET": "0"})


def _run(
    cmd: str,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> int:
    print(f"\n$ {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        executable="/bin/bash",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {cmd}")
    return rc


def _write_versions(repo: Path, log_dir: Path, env: dict[str, str]) -> None:
    versions = log_dir / "modal-versions.txt"
    script = r"""
set -euo pipefail
{
  echo "modal_image=${LMCACHE_MODAL_IMAGE:-unknown}"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current || true)"
  echo "cuda_nvcc=$(command -v nvcc >/dev/null && nvcc --version | tail -1 || true)"
  echo "nvidia_smi_query=$(nvidia-smi \
    --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader || true)"
  for py in /opt/venv/bin/python /workspace/.venv/bin/python python3; do
    if [ -x "$py" ] || command -v "$py" >/dev/null 2>&1; then
      "$py" - <<'PY' || true
import importlib.metadata as md
import sys
print(f"python={sys.executable} {sys.version.split()[0]}")
for pkg in ("torch", "vllm", "lmcache"):
    try:
        print(f"{pkg}={md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg}=unavailable ({exc})")
PY
    fi
  done
} > "$1" 2>&1
"""
    _run(
        f"bash -lc {shlex.quote(script)} bash {shlex.quote(str(versions))}",
        cwd=str(repo),
        env=env,
    )


def _copy_artifacts(repo: Path, build_id: str, mode: str, run_rc: int) -> str:
    log_dir = repo / f"logs_{build_id}"
    target_dir = Path("/artifacts") / build_id
    target_dir.mkdir(parents=True, exist_ok=True)
    if log_dir.exists():
        _run(
            f"cp -a {shlex.quote(str(log_dir))}/. {shlex.quote(str(target_dir))}/",
            check=False,
        )
    (target_dir / "modal-result.txt").write_text(
        f"build_id={build_id}\nmode={mode}\nrun_rc={run_rc}\nlog_dir={log_dir}\n",
        encoding="utf-8",
    )
    tar_path = Path("/artifacts") / f"{build_id}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(target_dir, arcname=build_id)
    artifacts.commit()
    print("\n=== artifact inventory ===", flush=True)
    find_cmd = " ".join(
        [
            "find",
            shlex.quote(str(target_dir)),
            "-maxdepth 1 -type f -printf '%f %s bytes\\n' | sort",
        ]
    )
    _run(find_cmd, check=False)
    print(f"\nArtifact volume: {ARTIFACT_VOLUME_NAME}")
    print(f"Artifact dir: {target_dir}")
    print(f"Artifact tarball: {tar_path}")
    return str(tar_path)


@app.function(
    image=image,
    gpu="H100:2",
    cpu=16,
    memory=131072,
    ephemeral_disk=524288,
    timeout=7200,
    startup_timeout=1800,
    secrets=modal_secrets,
    volumes={"/artifacts": artifacts},
)
def run_cacheblend_e2e(
    repo_url: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    model: str = DEFAULT_MODEL,
    mode: Mode = "full",
    build_id: str = "",
) -> dict[str, str | int]:
    if mode not in MODE_ENV:
        raise ValueError(f"mode must be one of {sorted(MODE_ENV)}, got {mode!r}")
    build_id = build_id or f"modal-cacheblend-{mode}-{int(time.time())}"
    repo = Path("/workspace/LMCache")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CC": "gcc",
            "CXX": "g++",
            "CUDA_HOME": "/usr/local/cuda-12.9",
            "CUDA_PATH": "/usr/local/cuda-12.9",
            "PATH": f"/usr/local/cuda-12.9/bin:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/targets/x86_64-linux/lib:"
            "/usr/local/cuda-13.0/lib64:"
            "/usr/local/cuda-12.9/targets/x86_64-linux/lib:"
            "/usr/local/cuda-12.9/lib64:"
            f"{os.environ.get('LD_LIBRARY_PATH', '')}",
            "LMCACHE_MODAL_IMAGE": DEFAULT_IMAGE,
            "BUILDKITE_BUILD_ID": build_id,
            "MODEL": model,
            "LMCACHE_SERVER_ENTRYPOINT": "cli",
            "LMCACHE_MP_PORT": "6566",
            "LMCACHE_HTTP_PORT": "8080",
            "SERVICE_PORT": "10001",
            "PREFILLER_PORT": "8100",
            "DECODER_PORT": "8200",
            "TELEMETRY_PORT": "5768",
            "TENSOR_PARALLEL": "1",
            "GPU_MEM_UTIL": "0.5",
            "HF_HOME": "/workspace/.cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/workspace/.cache/huggingface/hub",
        }
    )
    env.update(MODE_ENV[mode])
    if "HF_TOKEN" in env:
        env.setdefault("HUGGING_FACE_HUB_TOKEN", env["HF_TOKEN"])

    _run("nvidia-smi", env=env)
    _run("rm -rf /workspace/LMCache && mkdir -p /workspace /artifacts", env=env)
    clone_cmd = " ".join(
        [
            "git clone --branch",
            shlex.quote(branch),
            shlex.quote(repo_url),
            shlex.quote(str(repo)),
        ]
    )
    _run(clone_cmd, env=env)
    if commit:
        _run(
            f"git fetch origin {shlex.quote(commit)} --depth 1", cwd=str(repo), env=env
        )
        _run(f"git checkout --detach {shlex.quote(commit)}", cwd=str(repo), env=env)
    _run("git rev-parse HEAD && git status --short --branch", cwd=str(repo), env=env)

    log_dir = repo / f"logs_{build_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    _run(
        f"nvidia-smi > {shlex.quote(str(log_dir / 'nvidia-smi.txt'))}",
        env=env,
        check=False,
    )
    _write_versions(repo, log_dir, env)

    rc = 0
    try:
        rc = _run(
            "bash .buildkite/k3_tests/blend/run.sh", cwd=str(repo), env=env, check=False
        )
        if rc != 0:
            raise RuntimeError(f"blend E2E failed with exit code {rc}")
    finally:
        tar_path = _copy_artifacts(repo, build_id, mode, rc)
        _run(
            "for f in "
            f"{shlex.quote(str(log_dir))}/build_*_blend.log "
            f"{shlex.quote(str(log_dir))}/build_*_blend_server.log "
            f"{shlex.quote(str(log_dir))}/build_*_proxy.log "
            f"{shlex.quote(str(log_dir))}/build_*_benchmark.log; do "
            '[ -f "$f" ] || continue; echo "--- $f ---"; tail -120 "$f"; done',
            env=env,
            check=False,
        )

    return {
        "status": "passed",
        "repo_url": repo_url,
        "branch": branch,
        "commit": commit,
        "model": model,
        "mode": mode,
        "build_id": build_id,
        "artifact_volume": ARTIFACT_VOLUME_NAME,
        "artifact_tarball": tar_path,
    }


@app.local_entrypoint()
def main(
    repo_url: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    model: str = DEFAULT_MODEL,
    mode: str = "full",
    build_id: str = "",
) -> None:
    if mode not in MODE_ENV:
        raise argparse.ArgumentTypeError(f"mode must be one of {sorted(MODE_ENV)}")
    result = run_cacheblend_e2e.remote(
        repo_url=repo_url,
        branch=branch,
        commit=commit,
        model=model,
        mode=mode,  # type: ignore[arg-type]
        build_id=build_id,
    )
    print("\n=== Modal CacheBlend E2E result ===")
    print(result)
