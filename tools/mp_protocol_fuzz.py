#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic malformed-envelope fuzz smoke for the native MP server."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import json
import socket
import subprocess
import tempfile
import time
import urllib.request

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.multiprocess.mq import MessageQueueClient, msgspec_encode
from lmcache.v1.multiprocess.native_launcher import ensure_native_binary
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.base import RequestType as RequestTypeClass


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


def _status(http_port: int) -> dict[str, object]:
    return json.loads(
        urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/status",
            timeout=5,
        ).read()
    )


def _encoded_request_type(request_type: RequestTypeClass) -> bytes:
    return msgspec_encode(request_type, cls=RequestTypeClass)


def _fuzz_cases(iterations: int) -> list[list[bytes]]:
    cases = [
        [b"short-envelope"],
        [msgspec.msgpack.encode(1), b"\xc1"],
        [msgspec.msgpack.encode(2), msgspec.msgpack.encode(10_000)],
        [msgspec.msgpack.encode(3), _encoded_request_type(RequestType.STORE)],
        [
            msgspec.msgpack.encode(4),
            _encoded_request_type(RequestType.RETRIEVE),
            b"\xc1",
            msgspec.msgpack.encode(123),
            msgspec.msgpack.encode([1, 2]),
            b"\xc4\x02xx",
            msgspec.msgpack.encode(0),
        ],
        [
            msgspec.msgpack.encode(5),
            _encoded_request_type(RequestType.LOOKUP),
            b"\xc1",
            msgspec.msgpack.encode(1),
        ],
        [
            msgspec.msgpack.encode(6),
            _encoded_request_type(RequestType.CB_LOOKUP_PRE_COMPUTED),
            msgspec.msgpack.encode("not-an-ipc-key"),
        ],
        [
            msgspec.msgpack.encode(7),
            _encoded_request_type(RequestType.REPORT_BLOCK_ALLOCATION),
            msgspec.msgpack.encode(1),
            msgspec.msgpack.encode("facebook/opt-125m"),
            b"\xc1",
        ],
    ]
    if iterations <= len(cases):
        return cases[:iterations]
    out = list(cases)
    for index in range(len(cases), iterations):
        uid = msgspec.msgpack.encode(index + 1)
        request_type = _encoded_request_type(RequestType.RETRIEVE)
        payload_count = index % 5
        payloads = [
            bytes(((index + offset) % 256 for offset in range(size)))
            for size in range(1, payload_count + 1)
        ]
        out.append([uid, request_type, *payloads])
    return out


def _send_fuzz_cases(zmq_port: int, *, iterations: int) -> int:
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 1000)
    socket.connect(f"tcp://127.0.0.1:{zmq_port}")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    response_count = 0
    try:
        for frames in _fuzz_cases(iterations):
            socket.send_multipart(frames)
            deadline = time.time() + 0.05
            while time.time() < deadline:
                events = dict(poller.poll(5))
                if socket not in events:
                    continue
                socket.recv_multipart()
                response_count += 1
    finally:
        socket.close()
    return response_count


def _assert_ping_and_noop(zmq_port: int) -> None:
    context = zmq.Context.instance()
    client = MessageQueueClient(f"tcp://127.0.0.1:{zmq_port}", context)
    try:
        if client.submit_request(RequestType.PING, []).result(timeout=5) is not True:
            raise RuntimeError("PING failed after protocol fuzzing")
        if client.submit_request(RequestType.NOOP, []).result(timeout=5) != "OK":
            raise RuntimeError("NOOP failed after protocol fuzzing")
    finally:
        client.close()


def run(args: argparse.Namespace) -> dict[str, object]:
    binary = Path(args.binary) if args.binary else ensure_native_binary()
    zmq_port = _free_port()
    http_port = _free_port()
    with tempfile.TemporaryDirectory() as tempdir:
        proc = subprocess.Popen(
            [
                str(binary),
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
                str(Path(tempdir) / "disk"),
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_http(
                f"http://127.0.0.1:{http_port}/healthcheck",
                timeout_s=args.startup_timeout_s,
            )
            before = _status(http_port)
            responses = _send_fuzz_cases(zmq_port, iterations=args.iterations)
            _assert_ping_and_noop(zmq_port)
            after = _status(http_port)
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

    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    if not isinstance(before_metrics, dict) or not isinstance(after_metrics, dict):
        raise TypeError("status metrics must be dicts")
    before_invalid = int(before_metrics["invalid_payload_count"])
    after_invalid = int(after_metrics["invalid_payload_count"])
    invalid_delta = after_invalid - before_invalid
    if invalid_delta < args.min_invalid_payloads:
        raise RuntimeError(
            "protocol fuzzing did not produce enough invalid-payload records: "
            f"got {invalid_delta}, expected at least {args.min_invalid_payloads}"
        )
    return {
        "fuzz_cases": args.iterations,
        "invalid_payload_delta": invalid_delta,
        "request_count": after_metrics["request_count"],
        "responses": responses,
        "unsupported_count": after_metrics["unsupported_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary")
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--min-invalid-payloads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--l1-size-gb", type=float, default=0.001)
    parser.add_argument("--startup-timeout-s", type=float, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.min_invalid_payloads < 1:
        raise ValueError("--min-invalid-payloads must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
