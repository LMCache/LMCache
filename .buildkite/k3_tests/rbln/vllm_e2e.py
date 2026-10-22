# SPDX-License-Identifier: Apache-2.0
"""vLLM-RBLN + LMCache multiprocess end-to-end check on a real NPU.

Starts ``lmcache server`` on the rbln backend and ``vllm serve`` (vllm-rbln)
with ``LMCacheMPConnector`` in engine-driven mode, then sends the same greedy
prompt twice. vLLM's own prefix cache is disabled, so the second request can
only skip prefill through LMCache. The check fails unless:

- the first request stored chunks into L1 and the second read them back;
- both completions are identical.

The rbln-serve image has no curl, so everything here uses the standard
library.

Usage: python vllm_e2e.py [--model MODEL] [--log-dir DIR]
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

LMCACHE_PORT = 6555
LMCACHE_HTTP_PORT = 7555
VLLM_PORT = 8555

#: LMCache chunks are whole vLLM blocks on RBLN, so both use one size.
BLOCK_SIZE = 1024
MAX_MODEL_LEN = 4096

#: A fixed prompt long enough to span several full chunks.
PROMPT = " ".join(
    f"Paragraph {i}: the lighthouse keeper climbed {i + 100} steps, logged "
    f"the wind at {i % 7} knots, and wrote the ship names in a ledger."
    for i in range(120)
)


def _get(url: str, timeout: float = 10.0) -> str:
    """Return the body of a GET request."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def _post_json(url: str, body: dict, timeout: float = 600.0) -> dict:
    """POST ``body`` as JSON and return the decoded response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _metric(name: str) -> int:
    """Sum every sample of a Prometheus counter on the LMCache server."""
    body = _get(f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/metrics")
    total = 0.0
    for line in body.splitlines():
        if line.startswith(name):
            total += float(line.rsplit(" ", 1)[1])
    return int(total)


def _wait_for_increase(name: str, base: int, timeout: float = 60.0) -> int:
    """Poll counter ``name`` until it exceeds ``base``; return the increase.

    LMCache updates its counters from a background EventBus thread, so a
    counter can lag the request that caused it. Returns the last increase
    seen (0 or less) if ``timeout`` expires first.
    """
    deadline = time.monotonic() + timeout
    while True:
        increase = _metric(name) - base
        if increase > 0 or time.monotonic() >= deadline:
            return increase
        time.sleep(0.5)


def _wait_until_ready(url: str, proc: subprocess.Popen, timeout: float) -> None:
    """Poll ``url`` until it answers, failing if ``proc`` exits first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server for {url} exited with {proc.returncode}")
        try:
            _get(url, timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    raise SystemExit(f"{url} not ready within {timeout:.0f}s")


@contextmanager
def _process(
    cmd: list[str], log: Path, env: dict[str, str]
) -> Iterator[subprocess.Popen]:
    """Run ``cmd`` with output to ``log`` and terminate it on exit."""
    with log.open("w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        try:
            yield proc
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def _complete(model: str) -> str:
    """Return the greedy completion of ``PROMPT``."""
    response = _post_json(
        f"http://127.0.0.1:{VLLM_PORT}/v1/completions",
        {"model": model, "prompt": PROMPT, "max_tokens": 32, "temperature": 0},
    )
    return response["choices"][0]["text"]


def _check(model: str) -> None:
    """Run the two requests and assert store, hit, and equal outputs."""
    writes = _metric("lmcache_mp_l1_write_chunks_total")
    first = _complete(model)
    # vLLM can answer before the store lands; waiting here also keeps the
    # second request from racing the store into a miss.
    stored = _wait_for_increase("lmcache_mp_l1_write_chunks_total", writes)
    print(f"first completion: {first!r}; L1 chunks written: {stored}")
    if stored < 1:
        raise SystemExit("first request stored nothing into LMCache")

    reads = _metric("lmcache_mp_l1_read_chunks_total")
    second = _complete(model)
    read = _wait_for_increase("lmcache_mp_l1_read_chunks_total", reads)
    print(f"second completion: {second!r}; L1 chunks read: {read}")
    if read < 1:
        raise SystemExit("second request did not hit LMCache")
    if first != second:
        raise SystemExit("completion changed after the LMCache hit")


def main() -> None:
    """Start both servers, run the check, and dump logs on failure.

    Raises:
        SystemExit: If a server fails to start or any check fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--vllm-timeout", type=float, default=2400.0)
    args = parser.parse_args()

    lmcache_log = args.log_dir / "rbln_e2e_lmcache.log"
    vllm_log = args.log_dir / "rbln_e2e_vllm.log"
    env = dict(os.environ, LMCACHE_DEVICE_BACKEND="rbln")

    # Run both CLIs through this interpreter so they import the LMCache under
    # test, whether or not its console scripts are on PATH.
    lmcache_cmd = [
        sys.executable,
        "-m",
        "lmcache.cli.main",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(LMCACHE_PORT),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(LMCACHE_HTTP_PORT),
        "--l1-size-gb",
        "2",
        "--no-l1-use-lazy",
        "--eviction-policy",
        "LRU",
        "--chunk-size",
        str(BLOCK_SIZE),
        "--supported-transfer-mode",
        "engine_driven",
    ]
    kv_transfer_config = {
        "kv_connector": "LMCacheMPConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://127.0.0.1",
            "lmcache.mp.port": LMCACHE_PORT,
        },
    }
    vllm_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(VLLM_PORT),
        # Only the vLLM-native model runner wires up KV connectors; the
        # optimum-rbln runner never hands its KV caches to LMCache.
        "--model-impl",
        "vllm",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--block-size",
        str(BLOCK_SIZE),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(BLOCK_SIZE),
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        json.dumps(kv_transfer_config),
    ]

    try:
        with _process(lmcache_cmd, lmcache_log, env) as lmcache_proc:
            _wait_until_ready(
                f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/healthcheck", lmcache_proc, 60
            )
            with _process(vllm_cmd, vllm_log, env) as vllm_proc:
                _wait_until_ready(
                    f"http://127.0.0.1:{VLLM_PORT}/v1/models",
                    vllm_proc,
                    args.vllm_timeout,
                )
                _check(args.model)
    except BaseException:
        for log in (lmcache_log, vllm_log):
            if log.exists():
                print(f"--- tail of {log}", file=sys.stderr)
                print("\n".join(log.read_text().splitlines()[-200:]), file=sys.stderr)
        raise
    print("vLLM-RBLN + LMCache e2e ok")


if __name__ == "__main__":
    main()
