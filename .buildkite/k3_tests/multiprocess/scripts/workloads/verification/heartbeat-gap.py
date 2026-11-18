# SPDX-License-Identifier: Apache-2.0
"""Observe real ZMQ PING replies and test restart between successful heartbeats."""

# Standard
from pathlib import Path
from typing import TypedDict
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.transport.zmq.wire import decode_operation


class PingAck(TypedDict):
    """One genuine PING reply forwarded to a client."""

    time: float
    instance_id: int | None
    healthy: bool


class ServerStatus(TypedDict):
    """The public registration field used by this LMCache-driven CPU leg."""

    cache_context_meta: dict[str, dict[str, object]]


def observe(listen_port: int, server_port: int, trace: Path) -> None:
    """Forward this unauthenticated CPU leg unchanged, recording real PING ACKs.

    Each client keeps its original ZMQ identity on its own backend socket.
    No heartbeat or response is synthesized, delayed, or suppressed.
    """
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(f"tcp://127.0.0.1:{listen_port}")
    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    backends: dict[bytes, zmq.Socket] = {}
    identities: dict[zmq.Socket, bytes] = {}
    pending: dict[tuple[zmq.Socket, bytes], int | None] = {}
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("w") as output:
        trace.with_suffix(".ready").touch()
        while True:
            for sock in dict(poller.poll()).keys():
                frames = sock.recv_multipart()
                if sock is frontend:
                    identity, uid, operation, *payloads = frames
                    if identity not in backends:
                        backend = context.socket(zmq.DEALER)
                        backend.setsockopt(zmq.IDENTITY, identity)
                        backend.connect(f"tcp://127.0.0.1:{server_port}")
                        backends[identity] = backend
                        identities[backend] = identity
                        poller.register(backend, zmq.POLLIN)
                    backend = backends[identity]
                    if operation and decode_operation(operation) == "ping":
                        pending[backend, uid] = msgspec.msgpack.decode(payloads[0])
                    backend.send_multipart(frames[1:])
                else:
                    uid = frames[0]
                    if (sock, uid) in pending:
                        instance_id = pending.pop((sock, uid))
                        healthy = msgspec.msgpack.decode(frames[-1], type=bool)
                        frontend.send_multipart([identities[sock], *frames])
                        output.write(
                            json.dumps(
                                {
                                    "time": time.monotonic(),
                                    "instance_id": instance_id,
                                    "healthy": healthy,
                                }
                            )
                            + "\n"
                        )
                        output.flush()
                    else:
                        frontend.send_multipart([identities[sock], *frames])


def read_acks(trace: Path) -> list[PingAck]:
    """Read complete JSONL records while the observer may be appending."""
    return [
        json.loads(line)
        for line in trace.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]


def status(port: int) -> ServerStatus:
    """Fetch public Server status, including registered Worker IDs."""
    with urllib.request.urlopen(
        f"http://localhost:{port}/status", timeout=5
    ) as response:
        return json.load(response)


def cache_probe(directory: Path, phase: str) -> None:
    """Reuse the existing strict store/read/engine-load/output assertions."""
    helper = Path(__file__).with_name("l1-round-trip-verification.sh")
    subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
PROMPT_FILE="$2/$3.prompt"; OUT_A="$2/$3.cold"; OUT_B="$2/$3.hit"
MAX_TOKENS=8
engine_clear_local_cache() {
    local url="http://localhost:$ENGINE_PORT/reset_prefix_cache"
    curl -fsS --max-time 60 -X POST "$url" >/dev/null
}
printf '%s %s\\n' "$BUILD_ID" "$3" > "$PROMPT_FILE"
l1_round_trip_build_prompt "$2/body.prompt" "$3" 4 || exit 1
cat "$2/body.prompt" >> "$PROMPT_FILE"
l1_round_trip_verify_cache_service "$3"
""",
            "bash",
            str(helper),
            str(directory),
            phase,
        ],
        check=True,
    )


def verify(trace: Path, interval: int, directory: Path) -> None:
    """Require a genuine heartbeat gap, then assert cache recovery normally.

    Setup errors fail independently of the expected current regression. Cache
    failures remain ordinary nonzero test failures, never xfail or success.
    """
    directory.mkdir(parents=True, exist_ok=True)
    http_port = int(os.environ["LMCACHE_HTTP_PORT"])
    cache_probe(directory, "before_restart")
    before = status(http_port)
    (directory / "status-before.json").write_text(json.dumps(before, indent=2))
    workers = set(map(int, before["cache_context_meta"]))
    if len(workers) != 1:
        raise RuntimeError("This representative CPU test requires one Worker")
    acks = read_acks(trace)
    # Scheduler PING uses None when KV-event subscriptions are disabled.
    ids = {a["instance_id"] for a in acks}
    if not workers <= ids or len(ids) != 2 or not all(a["healthy"] for a in acks):
        raise RuntimeError("Expected successful Scheduler and Worker PING replies")
    latest = {
        iid: max(a["time"] for a in acks if a["instance_id"] == iid) for iid in ids
    }
    gap_deadline = min(latest.values()) + interval
    stopped_at = time.monotonic()
    if stopped_at - min(latest.values()) > 10:
        raise RuntimeError("PING replies were not fresh enough for fault injection")
    pid_file = Path(os.environ["MP_TEST_PID_FILE"])
    pids = pid_file.read_text().splitlines()
    old_pid = int(pids[0])
    stop_helper = Path(__file__).resolve().parents[2] / "stop-processes.py"
    subprocess.run(
        [
            sys.executable,
            str(stop_helper),
            str(old_pid),
            "--ports",
            os.environ["LMCACHE_PORT"],
            str(http_port),
        ],
        check=True,
    )
    with (directory / "server-restart.log").open("w") as log:
        server = subprocess.Popen(
            ["bash", os.environ["MP_TEST_LMCACHE_LAUNCH_SCRIPT"]],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    pids[0] = str(server.pid)
    pid_file.write_text("\n".join(pids) + "\n")
    while time.monotonic() < gap_deadline - 5:
        if server.poll() is not None:
            raise RuntimeError("Restarted Server exited; see server-restart.log")
        try:
            restarted = status(http_port)
            break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    else:
        raise RuntimeError("Server did not restart inside the heartbeat gap")
    ready_at = time.monotonic()
    (directory / "status-after-restart.json").write_text(
        json.dumps(restarted, indent=2)
    )
    if any(stopped_at <= a["time"] <= ready_at for a in read_acks(trace)):
        raise RuntimeError("A heartbeat overlapped the outage; gap was not established")
    print(
        f"Server restarted in {ready_at - stopped_at:.2f}s; "
        f"{gap_deadline - ready_at:.2f}s remain before next heartbeat",
        flush=True,
    )
    deadline = ready_at + interval + 30
    while time.monotonic() < deadline:
        acks = read_acks(trace)
        if not all(a["healthy"] for a in acks):
            raise RuntimeError(
                "PING failed; this exercised ordinary unhealthy recovery"
            )
        replied = {a["instance_id"] for a in acks if a["time"] > ready_at}
        if ids <= replied:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Did not observe both successful post-restart heartbeats")
    after = status(http_port)
    (directory / "status-after-heartbeats.json").write_text(json.dumps(after, indent=2))
    evidence = {
        "old_server_pid": old_pid,
        "new_server_pid": server.pid,
        "worker_ids": sorted(workers),
        "heartbeat_ids": sorted(ids, key=str),
        "heartbeat_interval": interval,
        "last_pre_restart_acks": latest,
        "stopped_at": stopped_at,
        "ready_at": ready_at,
        "gap_deadline": gap_deadline,
        "ping_replies": acks,
        "registered_after_heartbeats": sorted(after["cache_context_meta"]),
    }
    (directory / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(
        "Both real heartbeats succeeded after restart; registrations: "
        f"{evidence['registered_after_heartbeats']}",
        flush=True,
    )
    cache_probe(directory, "after_fast_restart")


def main() -> None:
    """Run the observer or the real serving regression from the CPU runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["observe", "verify"])
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--listen-port", type=int)
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "observe":
        observe(args.listen_port, args.server_port, args.trace)
    else:
        verify(args.trace, args.interval, args.output_dir)


if __name__ == "__main__":
    main()
