#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the mp mode message queue transport layer.

Directly compares ``zmq://`` (ipc + tcp) against ``grpc://`` on the exact
same code path (``MessageQueueClient`` / ``MessageQueueServer`` speaking
``RequestType.PING``), isolating transport overhead from cache
business logic.

Run with::

    python benchmarks/microbenchmark/mq_transport_benchmark.py

    # Only a subset of transports
    python benchmarks/microbenchmark/mq_transport_benchmark.py \\
        --transports ipc,grpc

    # Heavier load
    python benchmarks/microbenchmark/mq_transport_benchmark.py \\
        --requests 20000 --concurrency 16

Reported metrics per transport:

* ``rps``   sustained requests per second (higher is better)
* ``avg``   average round-trip latency (ms)
* ``p50``   median round-trip latency (ms)
* ``p99``   99th percentile round-trip latency (ms)
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import argparse
import contextlib
import socket
import statistics
import tempfile
import time

# First Party
from lmcache.v1.multiprocess.mq import (
    MessageQueueClient,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_payload_classes,
)


def _handle_ping(instance_id: int | None) -> bool:  # noqa: D401
    return True


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextlib.contextmanager
def _running_server(url: str):
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


def _run_client(url: str, requests: int, concurrency: int) -> dict:
    client = MessageQueueClient(url)
    lat_ms: list[float] = []

    def one_call() -> float:
        start = time.perf_counter()
        fut: object = client.submit_request(RequestType.PING, [None])
        fut.result(timeout=10)  # type: ignore[attr-defined]
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


def _bench_one(name: str, url: str, requests: int, concurrency: int) -> None:
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

        stats = _run_client(url, requests, concurrency)

    print(
        "{:<10} rps={:>9.0f}  avg={:>6.2f} ms  p50={:>6.2f} ms  p99={:>6.2f} ms".format(
            name,
            stats["rps"],
            stats["avg"],
            stats["p50"],
            stats["p99"],
        ),
    )


def _make_url_factories() -> dict[str, Callable[[], str]]:
    # Note: the ipc:// factory returns a fresh unique path every time it is
    # invoked so successive runs of the same bench do not collide.
    return {
        "ipc": lambda: "ipc://" + tempfile.mkdtemp(prefix="mq-bench-") + "/sock",
        "tcp": lambda: "tcp://127.0.0.1:" + str(_pick_free_port()),
        "grpc": lambda: "grpc://127.0.0.1:" + str(_pick_free_port()),
        "grpc-gzip": (
            lambda: "grpc://127.0.0.1:" + str(_pick_free_port()) + "?compression=gzip"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transports",
        default="ipc,tcp,grpc,grpc-gzip",
        help="Comma-separated list of transports to bench "
        "(default: ipc,tcp,grpc,grpc-gzip)",
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
    args = parser.parse_args()

    factories = _make_url_factories()
    transports = [t.strip() for t in args.transports.split(",") if t.strip()]

    print(
        "mq transport benchmark: requests={}, concurrency={}".format(
            args.requests, args.concurrency
        ),
    )
    for t in transports:
        if t not in factories:
            print("  skip unknown transport:", t)
            continue
        _bench_one(t, factories[t](), args.requests, args.concurrency)


if __name__ == "__main__":
    main()
