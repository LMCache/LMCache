# SPDX-License-Identifier: Apache-2.0
"""Run one cold request and prove a second request retrieved its KV cache."""

# Standard
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for(proc, url, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with urlopen(url, timeout=2):
                return
        except (URLError, TimeoutError):
            time.sleep(1)
    raise RuntimeError(f"{url} did not become ready (exit={proc.poll()})")


def wait_log(path, pattern, offset=0, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        match = re.search(pattern, path.read_text(errors="replace")[offset:])
        if match:
            print(match.group(0), flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"{pattern!r} not found in {path.name}")


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def main():
    backend, server_cli, directory = sys.argv[1:]
    out = Path(directory)
    model = "Qwen/Qwen3-0.6B"
    request_port, server_http, worker_http = 16555, 18580, 18100
    worker_url = f"http://127.0.0.1:{worker_http}"
    conf = json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "grpc://127.0.0.1",
                "lmcache.mp.port": request_port,
                "lmcache.mp.mq_timeout": 60,
                "lmcache.mp.mp_transfer_mode": "engine_driven",
            },
        }
    )
    server_log = out / "server.log"
    servers = []
    try:
        with server_log.open("w") as log:
            server = subprocess.Popen(
                [
                    server_cli,
                    "server",
                    "--transport",
                    "grpc",
                    "--port",
                    str(request_port),
                    "--http-port",
                    str(server_http),
                    "--l1-size-gb",
                    "1",
                    "--eviction-policy",
                    "LRU",
                    "--max-workers",
                    "2",
                    "--supported-transfer-mode",
                    "engine_driven",
                    "--no-l1-use-lazy",
                    "--shm-name",
                    f"wheel_probe_{free_port()}",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        servers.append(server)
        wait_for(server, f"http://127.0.0.1:{server_http}/healthcheck", 120)
        with (out / "vllm.log").open("w") as log:
            worker = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from vllm.entrypoints.cli.main import main; main()",
                    "serve",
                    model,
                    "--load-format",
                    "dummy",
                    "--enforce-eager",
                    "--no-enable-prefix-caching",
                    "--max-model-len",
                    "4096",
                    "--gpu-memory-utilization",
                    "0.5",
                    "--port",
                    str(worker_http),
                    "--kv-transfer-config",
                    conf,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "LMCACHE_DEVICE_BACKEND": backend},
                start_new_session=True,
            )
        servers.append(worker)
        wait_for(worker, worker_url + "/health", 600)
        payload = json.dumps(
            {
                "model": model,
                "prompt": "The device stores a prefix for later retrieval. " * 120,
                "max_tokens": 4,
                "temperature": 0,
                "kv_transfer_params": {"cached_token_stats": True},
            }
        ).encode()
        for phase in ("cold", "warm"):
            req = Request(
                worker_url + "/v1/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=180) as response:
                body = json.load(response)
            (out / f"{phase}.json").write_text(json.dumps(body, indent=2) + "\n")
            stats = body["kv_transfer_params"]["cached_token_stats"]
            print(phase, stats, flush=True)
            assert stats["num_vllm_cached_tokens"] == 0, stats
            if phase == "cold":
                assert stats["num_lmcache_cached_tokens"] == 0, stats
                wait_log(server_log, r"Stored [1-9]\d* tokens in .*")
                warm_offset = len(server_log.read_text(errors="replace"))
                cold_output = body["choices"][0]["text"]
            else:
                assert stats["num_lmcache_cached_tokens"] > 0, stats
                wait_log(server_log, r"Retrieved [1-9]\d* tokens in .*", warm_offset)
                assert body["choices"][0]["text"] == cold_output
        print("PASS: cold store, warm LMCache lookup hit and retrieve")
    finally:
        for proc in reversed(servers):
            stop(proc)


if __name__ == "__main__":
    main()
