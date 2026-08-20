#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the gRPC-backed mp mode message queue.

Compares canonical gRPC endpoint and compression configurations on the exact
same ``MessageQueueClient`` / ``MessageQueueServer`` PING path, isolating RPC
overhead from cache business logic. The historical ``ipc://`` and ``tcp://``
schemes are now compatibility aliases for gRPC, so they are intentionally not
reported as separate transports. Use an equivalent PING harness against the
pre-gRPC branch when collecting a ZMQ baseline.

Run with::

    python benchmarks/microbenchmark/mq_transport_benchmark.py

    # Only a subset of transports
    python benchmarks/microbenchmark/mq_transport_benchmark.py \\
        --transports grpc-unix,grpc

    # Heavier load
    python benchmarks/microbenchmark/mq_transport_benchmark.py \\
        --requests 20000 --concurrency 16

Reported metrics per transport:

* ``rps``   sustained requests per second (higher is better)
* ``avg``   average round-trip latency (ms)
* ``p50``   median round-trip latency (ms)
* ``p99``   99th percentile round-trip latency (ms)

Each reported value is the median across ``--repeats`` complete runs.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
import argparse
import contextlib
import socket
import statistics
import tempfile
import time

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import (
    MessageQueueClient,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.protocol import get_payload_classes
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType


def _handle_ping(instance_id: int | None) -> bool:  # noqa: D401
    return True


def _pick_free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextlib.contextmanager
def _running_server(url: str) -> Iterator[None]:
    server = MessageQueueServer(url)
    server.add_handler(
        RequestType.PING,
        get_payload_classes(RequestType.PING),
        HandlerType.BLOCKING,
        _handle_ping,
    )
    server.add_normal_thread_pool([RequestType.PING], max_workers=8)
    server.start()
    time.sleep(0.05)  # let the accept loop settle
    try:
        yield
    finally:
        server.close()


def _run_client(url: str, requests: int, concurrency: int) -> dict[str, float]:
    client = MessageQueueClient(url)
    lat_ms: list[float] = []

    def one_call() -> float:
        start = time.perf_counter()
        future: MessagingFuture[bool] = client.submit_request(RequestType.PING, [None])
        future.result(timeout=10)
        return (time.perf_counter() - start) * 1000.0

    try:
        wall_start = time.perf_counter()
        if concurrency == 1:
            for _ in range(requests):
                lat_ms.append(one_call())
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for latency in pool.map(lambda _i: one_call(), range(requests)):
                    lat_ms.append(latency)
        wall = time.perf_counter() - wall_start
    finally:
        client.close()

    lat_ms.sort()
    return {
        "rps": requests / wall,
        "avg": statistics.fmean(lat_ms),
        "p50": lat_ms[len(lat_ms) // 2],
        "p99": lat_ms[max(0, int(len(lat_ms) * 0.99) - 1)],
    }


def _bench_one(
    url: str,
    requests: int,
    concurrency: int,
) -> dict[str, float]:
    with _running_server(url):
        # Warm-up so JIT / connection-establishment don't skew the numbers.
        warmup_client = MessageQueueClient(url)
        try:
            for _ in range(200):
                warmup_client.submit_request(RequestType.PING, [None]).result(
                    timeout=5,
                )
        finally:
            warmup_client.close()

        return _run_client(url, requests, concurrency)


@contextlib.contextmanager
def _transport_url(name: str) -> Iterator[str]:
    if name == "grpc-unix":
        with tempfile.TemporaryDirectory(prefix="mq-bench-") as directory:
            yield f"grpc+unix://{directory}/sock"
        return
    if name == "grpc":
        yield f"grpc://127.0.0.1:{_pick_free_port()}"
        return
    if name == "grpc-gzip":
        yield f"grpc://127.0.0.1:{_pick_free_port()}?compression=gzip"
        return
    raise ValueError(f"unknown transport: {name}")


def _median_stats(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.median(sample[key] for sample in samples)
        for key in ("rps", "avg", "p50", "p99")
    }


def _print_stats(name: str, stats: dict[str, float]) -> None:
    print(
        "{:<12} rps={:>9.0f}  avg={:>6.2f} ms  p50={:>6.2f} ms  p99={:>6.2f} ms".format(
            name,
            stats["rps"],
            stats["avg"],
            stats["p50"],
            stats["p99"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transports",
        default="grpc-unix,grpc,grpc-gzip",
        help="Comma-separated list of transports to bench "
        "(default: grpc-unix,grpc,grpc-gzip)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=5000,
        help="Total requests per transport (default: 5000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent client threads (default: 1)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Complete runs per transport; reports the median (default: 5)",
    )
    args = parser.parse_args()

    transports = [t.strip() for t in args.transports.split(",") if t.strip()]
    valid_transports = {"grpc-unix", "grpc", "grpc-gzip"}
    unknown = sorted(set(transports) - valid_transports)
    if unknown:
        parser.error("unknown transport(s): " + ", ".join(unknown))
    if args.requests <= 0 or args.concurrency <= 0 or args.repeats <= 0:
        parser.error("requests, concurrency, and repeats must all be positive")

    print(
        "mq transport benchmark: requests={}, concurrency={}, repeats={}".format(
            args.requests,
            args.concurrency,
            args.repeats,
        ),
    )
    for transport in transports:
        samples = []
        for _ in range(args.repeats):
            with _transport_url(transport) as url:
                samples.append(_bench_one(url, args.requests, args.concurrency))
        _print_stats(transport, _median_stats(samples))


if __name__ == "__main__":
    main()
