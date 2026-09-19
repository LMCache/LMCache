# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for LMCache request-transport dispatch overhead.

The benchmark starts a minimal in-process request server with a PING handler
and measures one normal ``HandlerType.BLOCKING`` RPC. This isolates the
transport dispatch path from KV-cache hashing, storage, and transfer work.

Example:
    python benchmarks/microbenchmark/grpc_dispatch_benchmark.py \
        --transport grpc --warmup 200 --calls 2000 --workers 8 \
        --calls-per-worker 500
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import mean, median, quantiles
from typing import Any, Literal, cast
import argparse
import socket
import time

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.base import HandlerType
from lmcache.v1.multiprocess.request_handler import request_handler
from lmcache.v1.multiprocess.transport.base import RequestClient, RequestServer
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

Transport = Literal["grpc", "zmq"]


@dataclass(frozen=True)
class BenchStats:
    """Summary statistics for one benchmark scenario."""

    name: str
    calls: int
    duration_seconds: float
    latencies_ms: list[float]

    def print(self) -> None:
        """Print a compact benchmark summary."""
        sorted_latencies = sorted(self.latencies_ms)
        p95 = quantiles(sorted_latencies, n=20)[18]
        p99 = quantiles(sorted_latencies, n=100)[98]
        print(
            "%s: calls=%d duration=%.3fs throughput=%.1f/s "
            "avg=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms"
            % (
                self.name,
                self.calls,
                self.duration_seconds,
                self.calls / self.duration_seconds,
                mean(sorted_latencies),
                median(sorted_latencies),
                p95,
                p99,
            )
        )


class PingModule:
    """Minimal normal-pool blocking handler."""

    @request_handler(RequestType.PING, HandlerType.BLOCKING)
    def ping(self, instance_id: int | None) -> bool:
        """Return whether the instance id matches the benchmark value."""
        return instance_id == 7


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_url(transport: Transport, port: int) -> str:
    scheme = "grpc" if transport == "grpc" else "tcp"
    return f"{scheme}://127.0.0.1:{port}"


def _start_server(
    transport: Transport,
    *,
    max_cpu_workers: int,
    grpc_server_workers: int,
) -> tuple[RequestServer, str]:
    port = _unused_tcp_port()
    server = create_request_server(
        [cast(Any, PingModule())],
        MPServerConfig(
            transport=transport,
            host="127.0.0.1",
            port=port,
            max_cpu_workers=max_cpu_workers,
            max_gpu_workers=1,
            grpc_server_workers=grpc_server_workers,
        ),
    )
    server.start()
    return server, _request_url(transport, port)


def _create_client(target_url: str) -> RequestClient:
    return RequestClientFactory.create(target_url)


def _call_ping(client: RequestClient) -> float:
    started = time.perf_counter_ns()
    result = client.ping(7).result(timeout=5)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if result is not True:
        raise RuntimeError(f"unexpected ping result: {result!r}")
    return elapsed_ms


def run_sequential(client: RequestClient, calls: int) -> BenchStats:
    """Run one client serially."""
    latencies: list[float] = []
    started = time.perf_counter()
    for _ in range(calls):
        latencies.append(_call_ping(client))
    duration = time.perf_counter() - started
    return BenchStats("sequential", calls, duration, latencies)


def run_concurrent(
    target_url: str,
    workers: int,
    calls_per_worker: int,
) -> BenchStats:
    """Run multiple client channels in parallel."""
    started = time.perf_counter()

    def worker() -> list[float]:
        client = _create_client(target_url)
        try:
            return [_call_ping(client) for _ in range(calls_per_worker)]
        finally:
            client.close()

    latencies: list[float] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for samples in pool.map(lambda _index: worker(), range(workers)):
            latencies.extend(samples)
    duration = time.perf_counter() - started
    return BenchStats(
        f"concurrent-{workers}",
        workers * calls_per_worker,
        duration,
        latencies,
    )


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["grpc", "zmq"], default="grpc")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calls-per-worker", type=int, default=250)
    parser.add_argument("--max-cpu-workers", type=int, default=8)
    parser.add_argument("--grpc-server-workers", type=int, default=32)
    args = parser.parse_args()

    server, target_url = _start_server(
        args.transport,
        max_cpu_workers=args.max_cpu_workers,
        grpc_server_workers=args.grpc_server_workers,
    )
    client = _create_client(target_url)
    try:
        for _ in range(args.warmup):
            _call_ping(client)
        run_sequential(client, args.calls).print()
        run_concurrent(target_url, args.workers, args.calls_per_worker).print()
    finally:
        client.close()
        server.close()


if __name__ == "__main__":
    main()
