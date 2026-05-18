#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a small native MP concurrency stress harness.

This is intentionally independent of pytest so it can launch ThreadSanitizer
builds through wrappers such as ``setarch -R`` when the host libtsan runtime
requires that.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
import urllib.request

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.native_launcher import ensure_native_binary
from lmcache.v1.multiprocess.protocol import RequestType


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5).read()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"server did not become ready at {url}: {last_error}")


def _latency_histogram_count(status: dict[str, object]) -> int:
    metrics = status["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("status metrics must be a dict")
    histogram = metrics["request_latency_histogram"]
    if not isinstance(histogram, dict):
        raise TypeError("request_latency_histogram must be a dict")
    return sum(int(value) for value in histogram.values())


def _server_command(
    args: argparse.Namespace, zmq_port: int, http_port: int
) -> list[str]:
    binary = Path(args.binary) if args.binary else ensure_native_binary()
    command = [str(binary)]
    if args.setarch:
        command = ["setarch", platform.machine(), "-R", *command]
    command.extend(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            str(args.l1_size_gb),
            "--max-workers",
            str(args.workers),
            "--cxx-disk-path",
            args.disk_path,
        ]
    )
    return command


def _stress_clients(
    zmq_port: int,
    *,
    workers: int,
    iterations: int,
    duration_s: float | None,
) -> int:
    context = zmq.Context.instance()

    malformed = context.socket(zmq.DEALER)
    malformed.setsockopt(zmq.LINGER, 1000)
    malformed.connect(f"tcp://127.0.0.1:{zmq_port}")
    malformed.send_multipart([b"malformed-only-one-frame"])
    time.sleep(0.05)
    malformed.close()
    time.sleep(0.05)

    def submit_one(client: MessageQueueClient, client_id: int, i: int) -> None:
        if (client_id + i) % 2 == 0:
            if (
                client.submit_request(RequestType.PING, []).result(timeout=5)
                is not True
            ):
                raise RuntimeError("PING did not return True")
        elif client.submit_request(RequestType.NOOP, []).result(timeout=5) != "OK":
            raise RuntimeError("NOOP did not return OK")

    def worker(client_id: int) -> int:
        client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
        try:
            count = 0
            if duration_s is not None:
                deadline = time.monotonic() + duration_s
                while time.monotonic() < deadline or count == 0:
                    submit_one(client, client_id, count)
                    count += 1
                return count
            for i in range(iterations):
                submit_one(client, client_id, i)
                count += 1
            return count
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, client_id) for client_id in range(workers)]
        timeout = max(20.0, (duration_s or float(workers * iterations)) + 30.0)
        return sum(future.result(timeout=timeout) for future in futures)


def _validate_status(status: dict[str, object], *, workers: int, expected: int) -> None:
    metrics = status["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("status metrics must be a dict")
    checks = {
        "request_count": int(metrics["request_count"]) >= expected,
        "worker_count": metrics["worker_count"] == workers,
        "active_client_count": metrics["active_client_count"] == workers,
        "observed_client_count": metrics["observed_client_count"] == workers,
        "active_worker_count": metrics["active_worker_count"] == 0,
        "worker_queue_depth": metrics["worker_queue_depth"] == 0,
        "request_latency_count": (
            metrics["request_latency_count"] == metrics["request_count"]
        ),
        "request_latency_histogram": (
            _latency_histogram_count(status) == metrics["request_latency_count"]
        ),
        "invalid_payload_count": metrics["invalid_payload_count"] == 1,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"native stress status checks failed: {failed}; {metrics}")


def run(args: argparse.Namespace) -> dict[str, object]:
    zmq_port = _free_port()
    http_port = _free_port()
    with tempfile.TemporaryDirectory() as tempdir:
        args.disk_path = str(Path(tempdir) / "disk")
        env = os.environ.copy()
        if args.tsan:
            env.setdefault("TSAN_OPTIONS", "halt_on_error=1:second_deadlock_stack=1")
        proc = subprocess.Popen(
            _server_command(args, zmq_port, http_port),
            env=env,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_http(
                f"http://127.0.0.1:{http_port}/healthcheck",
                timeout_s=args.startup_timeout_s,
            )
            expected_requests = _stress_clients(
                zmq_port,
                workers=args.workers,
                iterations=args.iterations,
                duration_s=args.duration_s,
            )
            status = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/status",
                    timeout=5,
                ).read()
            )
            _validate_status(status, workers=args.workers, expected=expected_requests)
            metrics = status["metrics"]
            if not isinstance(metrics, dict):
                raise TypeError("status metrics must be a dict")
            return {
                "active_client_count": metrics["active_client_count"],
                "duration_s": args.duration_s,
                "expected_request_count": expected_requests,
                "request_count": metrics["request_count"],
                "invalid_payload_count": metrics["invalid_payload_count"],
                "request_latency_count": metrics["request_latency_count"],
                "worker_count": metrics["worker_count"],
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            if "ThreadSanitizer" in stderr:
                raise RuntimeError(stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Run each client until this duration elapses instead of fixed iterations.",
    )
    parser.add_argument("--l1-size-gb", type=float, default=0.001)
    parser.add_argument("--startup-timeout-s", type=float, default=20)
    parser.add_argument("--setarch", action="store_true")
    parser.add_argument("--tsan", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.duration_s is not None and args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
