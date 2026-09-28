# SPDX-License-Identifier: Apache-2.0
"""Measure dispatch with the client and server in separate processes.

PING isolates tiny-request overhead. LOOKUP includes a token-key payload but
uses a synthetic handler, so neither case measures KV-cache or GPU performance.
``--handler-delay-ms`` simulates blocking work to expose lost concurrency.

Example:
    python benchmarks/microbenchmark/grpc_dispatch_benchmark.py \
        --transport grpc --request lookup --tokens 512 --workers 8 \
        --handler-delay-ms 1 --json-output dispatch.json
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal, cast
import argparse
import json
import math
import multiprocessing
import platform
import socket
import threading
import time

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

Transport = Literal["grpc", "zmq"]


@dataclass(frozen=True)
class BenchStats:
    """Latency and throughput for one scenario, excluding client startup."""

    name: str
    duration_seconds: float
    latencies_ms: list[float]

    def summary(self) -> dict[str, str | int | float]:
        """Return throughput and empirical latency percentiles."""
        samples = sorted(self.latencies_ms)
        return {
            "name": self.name,
            "calls": len(samples),
            "duration_seconds": self.duration_seconds,
            "throughput_per_second": len(samples) / self.duration_seconds,
            "avg_ms": mean(samples),
            "p50_ms": median(samples),
            "p95_ms": samples[math.ceil(0.95 * len(samples)) - 1],
            "p99_ms": samples[math.ceil(0.99 * len(samples)) - 1],
        }


class DispatchModule:
    """Synthetic normal handlers with optional blocking work."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    @request_handler(HandlerType.BLOCKING)
    def ping(self, instance_id: int | None) -> bool:
        """Return whether the instance id matches the benchmark value."""
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        return instance_id == 7

    @request_handler(HandlerType.BLOCKING)
    def lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
        """Decode a realistic lookup payload without accessing storage."""
        if self._delay_seconds:
            time.sleep(self._delay_seconds)


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serve(
    config: MPServerConfig, delay: float, ready: Connection, stop: Event
) -> None:
    server = create_request_server([cast(Any, DispatchModule(delay))], config)
    try:
        server.start()
        ready.send(True)
        stop.wait()
    finally:
        ready.close()
        server.close()


@contextmanager
def _server_url(args: argparse.Namespace) -> Iterator[str]:
    port = _unused_tcp_port()
    config = MPServerConfig(
        transport=args.transport,
        host="127.0.0.1",
        port=port,
        max_cpu_workers=args.max_cpu_workers,
        max_gpu_workers=1,
        grpc_server_workers=args.grpc_server_workers,
    )
    # Spawn before creating clients: forking an active gRPC runtime is unsafe.
    ctx = multiprocessing.get_context("spawn")
    ready, child_ready = ctx.Pipe(duplex=False)
    stop = ctx.Event()
    server = ctx.Process(
        target=_serve, args=(config, args.handler_delay_ms / 1000, child_ready, stop)
    )
    server.start()
    child_ready.close()
    try:
        if not ready.poll(30) or not ready.recv():
            raise RuntimeError("Request server did not become ready")
        scheme = "grpc" if args.transport == "grpc" else "tcp"
        yield f"{scheme}://127.0.0.1:{port}"
    finally:
        ready.close()
        stop.set()
        server.join(timeout=10)
        if server.is_alive():
            server.terminate()
            server.join(timeout=5)
        server.close()


def _call(client: RequestClient, key: IPCCacheServerKey | None) -> float:
    started = time.perf_counter_ns()
    if key is None:
        result = client.ping(7).result(timeout=30)
        if result is not True:
            raise RuntimeError(f"Unexpected ping result: {result!r}")
    else:
        lookup_result = client.lookup(key, 1).result(timeout=30)
        if lookup_result is not None:
            raise RuntimeError(f"Unexpected lookup result: {lookup_result!r}")
    return (time.perf_counter_ns() - started) / 1_000_000


def run_scenario(
    url: str, workers: int, calls: int, warmup: int, key: IPCCacheServerKey | None
) -> BenchStats:
    """Measure warmed clients with one outstanding request per worker."""
    clients = [RequestClientFactory.create(url) for _ in range(workers)]
    try:
        for client in clients:
            for _ in range(max(1, warmup)):
                _call(client, key)
        barrier = threading.Barrier(workers + 1)

        def worker(client: RequestClient) -> list[float]:
            barrier.wait()
            return [_call(client, key) for _ in range(calls)]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, client) for client in clients]
            started = time.perf_counter()
            barrier.wait()
            latencies = [sample for future in futures for sample in future.result()]
            duration = time.perf_counter() - started
        name = "sequential" if workers == 1 else f"concurrent-{workers}"
        return BenchStats(name, duration, latencies)
    finally:
        for client in clients:
            client.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> None:
    """Run independent sequential and concurrent dispatch measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["grpc", "zmq"], default="grpc")
    parser.add_argument("--request", choices=["ping", "lookup"], default="ping")
    parser.add_argument("--tokens", type=_positive_int, default=512)
    parser.add_argument("--handler-delay-ms", type=float, default=0)
    parser.add_argument("--warmup", type=_positive_int, default=100)
    parser.add_argument("--calls", type=_positive_int, default=2000)
    parser.add_argument("--workers", type=_positive_int, default=8)
    parser.add_argument("--calls-per-worker", type=_positive_int, default=500)
    parser.add_argument("--max-cpu-workers", type=_positive_int, default=8)
    parser.add_argument("--grpc-server-workers", type=_positive_int, default=32)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.handler_delay_ms) or args.handler_delay_ms < 0:
        parser.error("--handler-delay-ms must be finite and non-negative")

    key = None
    if args.request == "lookup":
        key = IPCCacheServerKey(
            model_name="benchmark",
            world_size=1,
            worker_id=None,
            token_ids=tuple(range(args.tokens)),
            start=0,
            end=args.tokens,
            request_id="dispatch-benchmark",
            cache_salt="benchmark",
            request_configs={},
            num_kv_readers=1,
        )
    with _server_url(args) as url:
        results = [
            run_scenario(url, 1, args.calls, args.warmup, key).summary(),
            run_scenario(
                url, args.workers, args.calls_per_worker, args.warmup, key
            ).summary(),
        ]
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "grpcio": version("grpcio"),
        "pyzmq": version("pyzmq"),
        "config": {k: v for k, v in vars(args).items() if k != "json_output"},
        "results": results,
    }
    output = json.dumps(report, indent=2) + "\n"
    print(output, end="")
    if args.json_output:
        args.json_output.write_text(output)


if __name__ == "__main__":
    main()
