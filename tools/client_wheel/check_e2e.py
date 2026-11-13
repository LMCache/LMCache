# SPDX-License-Identifier: Apache-2.0
"""Run a real request from the client wheel against a server from the full wheel.

    python tools/client_wheel/check_e2e.py --server-python PYTHON
        --client-python PYTHON [--transport zmq grpc] [--report FILE] [--logs DIR]

For each transport, starts ``lmcache server`` from the full wheel's venv in an
empty directory and waits for /healthcheck. A client then runs in the client
wheel's venv with ``python -I -B`` from another empty directory: it registers a
small CPU KV cache, looks up two chunks (no hit expected), stores them, looks
them up again under a new request (every token must hit), retrieves them into
zeroed blocks and compares the bytes. The server commits a store
asynchronously, so the second lookup is repeated, under a fresh request ID,
until it hits or 10 seconds pass. Every lmcache module the client loads must
come from the client venv. The server then gets SIGTERM; it must exit and free
its ports.
"""

# Standard
from pathlib import Path
import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# Runs in the client venv. Only client, common and platform modules are used.
CLIENT = r"""
import importlib.metadata, json, sys, sysconfig, time

import torch
import zmq

from lmcache.utils import EngineType
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.client.worker_transfer import create_transfer_context
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.group_view import EngineGroupInfo

TIMEOUT, MODEL, BLOCK, LAYERS = 30.0, "client-wheel-e2e", 16, 2
context = zmq.Context()
context.setsockopt(zmq.LINGER, 0)
client = RequestClientFactory.create(sys.argv[1], context=context)
chunk = client.get_chunk_size().result(timeout=TIMEOUT)
tokens = tuple(range(1, 2 * chunk + 1))
blocks = len(tokens) // BLOCK
source, target = list(range(blocks)), list(range(blocks, 2 * blocks))
# vLLM's NHD layout: (K/V, blocks, tokens per block, heads, head size).
kv = {
    f"tensor.{i}": torch.zeros(2, 2 * blocks, BLOCK, 2, 64, dtype=torch.float16)
    for i in range(LAYERS)
}
generator = torch.Generator().manual_seed(6315)
for tensor in kv.values():
    values = torch.randint(1, 128, tensor[:, source].shape, generator=generator)
    tensor[:, source] = values.to(tensor.dtype)
transfer = create_transfer_context(
    kv, instance_id=1000, req_client=client, mode="lmcache_driven"
)
groups = [
    EngineGroupInfo(
        engine_group_id=0, layer_indices=tuple(range(LAYERS)), tokens_per_block=BLOCK
    )
]
transfer.register(
    kv, MODEL, 1, chunk // BLOCK, TIMEOUT, layout_hints={"kv_layout": "NHD"},
    engine_group_infos=groups, engine_type=EngineType.VLLM,
)


def key(request_id, worker_id):
    return IPCCacheServerKey.from_token_ids(
        model_name=MODEL, world_size=1, worker_id=worker_id, token_ids=tokens,
        start=0, end=len(tokens), request_id=request_id,
    )


def lookup(request_id):
    client.lookup(key(request_id, None), 1).result(timeout=TIMEOUT)
    for _ in range(200):
        hit = client.query_prefetch_status(request_id).result(timeout=TIMEOUT)
        if hit is not None:
            return hit
        time.sleep(0.05)
    raise TimeoutError("the prefetch status never settled")


def transfer_blocks(submit, request_id, block_ids):
    event = transfer.create_recorded_event()
    future = submit(
        request_id, key(request_id, 0), kv, [block_ids], event, chunk // BLOCK
    )
    return bool(future.result(timeout=TIMEOUT))


report = {"chunk_size": chunk, "tokens": len(tokens)}
report["cold_hit_tokens"] = lookup("first")
report["stored"] = transfer_blocks(transfer.submit_store, "first", source)
client.end_session("first").result(timeout=TIMEOUT)
deadline = time.monotonic() + 10
for attempt in range(1000):
    second = f"second-{attempt}"
    report["warm_hit_tokens"] = lookup(second)
    if report["warm_hit_tokens"] == len(tokens) or time.monotonic() > deadline:
        break
    client.end_session(second).result(timeout=TIMEOUT)
    time.sleep(0.1)
report["warm_lookups"] = attempt + 1
report["retrieved"] = transfer_blocks(transfer.submit_retrieve, second, target)
client.end_session(second).result(timeout=TIMEOUT)
report["bytes_match"] = all(
    torch.equal(tensor[:, source], tensor[:, target]) for tensor in kv.values()
)
transfer.close()
client.close()
context.term()
site = {sysconfig.get_paths()[name] for name in ("purelib", "platlib")}
loaded = {
    name: getattr(module, "__file__", None)
    for name, module in sys.modules.items()
    if name == "lmcache" or name.startswith("lmcache.")
}
version = importlib.metadata.version("lmcache-client")
report["distribution"] = f"lmcache-client {version}"
report["lmcache_modules"] = len(loaded)
report["outside_client_venv"] = sorted(
    name for name, path in loaded.items()
    if path and not any(path.startswith(prefix) for prefix in site)
)
report["server_modules"] = sorted(
    name for name in loaded
    if name.startswith(("lmcache.v1.multiprocess.", "lmcache.v1.distributed."))
)
print("REPORT " + json.dumps(report))
"""


def free_port() -> int:
    """Return a TCP port that is free on 127.0.0.1 right now."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def listening(port: int) -> bool:
    """Whether anything still accepts connections on 127.0.0.1:``port``.

    A connect test, not a bind: closed connections may hold the port in
    TIME_WAIT after the server is gone."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except OSError:
        return False
    return True


def wait_healthy(url: str, server: subprocess.Popen, timeout: float) -> None:
    """Poll ``url`` until it answers 200.

    Raises:
        RuntimeError: If the server exits or does not answer within ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"the server exited with {server.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"no answer from {url} within {timeout:.0f}s")


def run(transport: str, server_python: Path, client_python: Path, logs: Path) -> dict:
    """Serve from the full wheel, run CLIENT against it and return its report."""
    port, http = free_port(), free_port()
    command = [
        str(server_python.parent / "lmcache"),
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--http-port",
        str(http),
        "--transport",
        transport,
        "--l1-size-gb",
        "1",
        "--eviction-policy",
        "LRU",
        "--supported-transfer-mode",
        "lmcache_driven",
    ]
    url = f"{'grpc' if transport == 'grpc' else 'tcp'}://127.0.0.1:{port}"
    report: dict = {"transport": transport}
    log = logs / f"server-{transport}.log"
    with tempfile.TemporaryDirectory() as cwd, open(log, "w") as out:
        server = subprocess.Popen(
            command, cwd=cwd, stdout=out, stderr=subprocess.STDOUT
        )
        try:
            wait_healthy(f"http://127.0.0.1:{http}/healthcheck", server, 120)
            client = subprocess.run(
                [str(client_python), "-I", "-B", "-c", CLIENT, url],
                cwd=tempfile.mkdtemp(dir=cwd),
                capture_output=True,
                text=True,
                timeout=300,
            )
            (logs / f"client-{transport}.log").write_text(client.stdout + client.stderr)
            lines = client.stdout.splitlines()
            found = [line for line in lines if line.startswith("REPORT ")]
            if found:
                report.update(json.loads(found[-1][len("REPORT ") :]))
            else:
                report["error"] = client.stderr[-2000:] or f"exit {client.returncode}"
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            report["error"] = str(exc)
        finally:
            server.terminate()
            try:
                report["server_exit"] = server.wait(timeout=30)
                report["server_stopped"] = True
            except subprocess.TimeoutExpired:
                server.kill()
                report["server_stopped"] = False
    report["ports_released"] = not (listening(port) or listening(http))
    return report


def failures(report: dict) -> list[str]:
    if "error" in report:
        return [f"the request did not complete: {report['error']}"]
    checks = {
        "no hit before the first store": report["cold_hit_tokens"] == 0,
        "the store succeeded": report["stored"],
        "every token hits on the second request": (
            report["warm_hit_tokens"] == report["tokens"]
        ),
        "the retrieve succeeded": report["retrieved"],
        "retrieved bytes equal the stored ones": report["bytes_match"],
        "every lmcache module comes from the client venv": (
            not report["outside_client_venv"]
        ),
        "the server stops within 30s of SIGTERM": report["server_stopped"],
        "the server releases its ports": report["ports_released"],
    }
    return [name for name, ok in checks.items() if not ok]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--client-python", type=Path, required=True)
    parser.add_argument(
        "--transport", nargs="+", choices=("zmq", "grpc"), default=["zmq", "grpc"]
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--logs", type=Path, default=Path(tempfile.mkdtemp(prefix="e2e-"))
    )
    args = parser.parse_args()

    args.logs.mkdir(parents=True, exist_ok=True)
    reports = [
        run(transport, args.server_python, args.client_python, args.logs)
        for transport in args.transport
    ]
    failed = False
    for report in reports:
        problems = failures(report)
        failed |= bool(problems)
        summary = {k: v for k, v in report.items() if k != "server_modules"}
        print(json.dumps(summary))
        modules = report.get("server_modules", [])
        print(f"  server modules the client still loads: {len(modules)}")
        for problem in problems:
            print(f"FAIL {report['transport']}: {problem}")
    print(f"logs in {args.logs}")
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2) + "\n")
    sys.exit(int(failed))


if __name__ == "__main__":
    main()
