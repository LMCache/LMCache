#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare Python and native MP controller-envelope latency.

This benchmark is deliberately scoped to the native binary surface currently
implemented in this branch. It does not claim KV data-path parity.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time

# Third Party
import zmq

# First Party
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.native_launcher import ensure_native_binary
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.token_hasher import TokenHasher

_CHUNK_SIZE = 256


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(server: str, port: int, l2_dir: Path | None) -> list[str]:
    if server == "native":
        argv = [
            str(ensure_native_binary()),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--l1-size-gb",
            "0.001",
            "--eviction-policy",
            "LRU",
            "--disable-http",
        ]
        if l2_dir is not None:
            argv.extend(
                [
                    "--l2-adapter",
                    json.dumps({"type": "fs", "base_path": str(l2_dir)}),
                ]
            )
        return argv

    argv = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--l1-size-gb",
        "0.001",
        "--eviction-policy",
        "LRU",
        "--disable-observability",
    ]
    if l2_dir is not None:
        argv.extend(
            [
                "--l2-adapter",
                json.dumps({"type": "fs", "base_path": str(l2_dir)}),
            ]
        )
    return argv


def _lookup_miss_key(
    request_id: str = "controller-latency-lookup-miss",
) -> IPCCacheEngineKey:
    return IPCCacheEngineKey.from_token_ids(
        model_name="lmcache-controller-benchmark",
        world_size=1,
        worker_id=None,
        token_ids=list(range(256)),
        request_id=request_id,
        cache_salt="controller-benchmark",
    )


def _lookup_fs_l2_partial_key(
    request_id: str = "controller-latency-l2-partial",
) -> IPCCacheEngineKey:
    return IPCCacheEngineKey.from_token_ids(
        model_name="lmcache-controller-benchmark",
        world_size=1,
        worker_id=None,
        token_ids=list(range(_CHUNK_SIZE * 2)),
        request_id=request_id,
        cache_salt="controller-benchmark-l2",
    )


def _fs_l2_filename(key: ObjectKey) -> str:
    model_name = key.model_name.replace("/", "-SEP-")
    kv_rank = key.kv_rank
    chunk_hash = key.chunk_hash.hex()
    cache_salt = key.cache_salt
    base = f"{model_name}@{kv_rank:#010x}@{chunk_hash}"
    if cache_salt:
        return f"{base}@{cache_salt}.data"
    return f"{base}.data"


def _seed_fs_l2_partial_hit(l2_dir: Path) -> None:
    l2_dir.mkdir(parents=True, exist_ok=True)
    lookup_key = _lookup_fs_l2_partial_key()
    chunk_hashes = TokenHasher(
        chunk_size=_CHUNK_SIZE,
        hash_algorithm="blake3",
    ).compute_chunk_hashes(list(lookup_key.token_ids))
    object_keys = ipc_key_to_object_keys(lookup_key, chunk_hashes)
    if len(object_keys) != 2:
        raise RuntimeError("expected two object keys for FS-L2 partial-hit seed")
    (l2_dir / _fs_l2_filename(object_keys[0])).write_bytes(b"l2-payload")


def _lookup_key_for_request(request: str, client_index: int) -> IPCCacheEngineKey:
    if request == "lookup-fs-l2-partial":
        return _lookup_fs_l2_partial_key(
            request_id=f"controller-latency-l2-partial-{client_index}"
        )
    return _lookup_miss_key(request_id=f"controller-latency-lookup-miss-{client_index}")


def _submit_request(
    client: MessageQueueClient,
    request: str,
    lookup_key: IPCCacheEngineKey,
) -> None:
    if request == "ping":
        assert client.submit_request(RequestType.PING, []).result(timeout=5)
        return
    if request == "noop":
        assert client.submit_request(RequestType.NOOP, []).result(timeout=5) == "OK"
        return
    if request in {"lookup-miss", "lookup-fs-l2-partial"}:
        assert (
            client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(timeout=5)
            is None
        )
        return
    raise ValueError(f"unsupported benchmark request: {request!r}")


def _wait_for_ping(port: int) -> None:
    context = zmq.Context.instance()
    client = MessageQueueClient(f"tcp://127.0.0.1:{port}", context)
    deadline = time.time() + 10
    try:
        while time.time() < deadline:
            try:
                if client.submit_request(RequestType.PING, []).result(timeout=1):
                    return
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
    finally:
        client.close()
    raise RuntimeError("server did not respond to PING")


def _clock_ticks_per_second() -> int:
    value = os.sysconf("SC_CLK_TCK")
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError("SC_CLK_TCK must be a positive integer")
    return value


def _read_sched_values(sched_text: str) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    for line in sched_text.splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", maxsplit=1)
        key = raw_key.strip()
        value = raw_value.strip()
        if key in {
            "se.sum_exec_runtime",
            "se.nr_migrations",
            "nr_switches",
            "nr_voluntary_switches",
            "nr_involuntary_switches",
        }:
            try:
                values[key] = float(value) if "." in value else int(value.split()[0])
            except ValueError:
                pass
    return values


def _read_task_sched_totals(proc_dir: Path) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "se.sum_exec_runtime": 0.0,
        "se.nr_migrations": 0,
        "nr_switches": 0,
        "nr_voluntary_switches": 0,
        "nr_involuntary_switches": 0,
    }
    task_dir = proc_dir / "task"
    try:
        task_paths = list(task_dir.iterdir())
    except OSError:
        return totals
    for task_path in task_paths:
        try:
            values = _read_sched_values(
                (task_path / "sched").read_text(encoding="utf-8")
            )
        except OSError:
            continue
        for key in totals:
            totals[key] += values.get(key, 0)
    return totals


def _read_process_resources(pid: int) -> dict[str, int | float] | None:
    proc_dir = Path("/proc") / str(pid)
    try:
        status_text = (proc_dir / "status").read_text(encoding="utf-8")
        stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
    except OSError:
        return None

    rss_bytes = 0
    rss_peak_bytes = 0
    voluntary_ctxt_switches = 0
    nonvoluntary_ctxt_switches = 0
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            rss_bytes = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            rss_peak_bytes = int(line.split()[1]) * 1024
        elif line.startswith("voluntary_ctxt_switches:"):
            voluntary_ctxt_switches = int(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            nonvoluntary_ctxt_switches = int(line.split()[1])

    try:
        sched_text = (proc_dir / "sched").read_text(encoding="utf-8")
    except OSError:
        sched_text = ""
    sched_values = _read_sched_values(sched_text)
    task_sched_values = _read_task_sched_totals(proc_dir)

    stat_tail = stat_text.rsplit(")", maxsplit=1)[1].split()
    ticks = _clock_ticks_per_second()
    user_cpu_s = int(stat_tail[11]) / ticks
    system_cpu_s = int(stat_tail[12]) / ticks
    return {
        "rss_bytes": rss_bytes,
        "rss_peak_bytes": rss_peak_bytes,
        "user_cpu_s": user_cpu_s,
        "system_cpu_s": system_cpu_s,
        "total_cpu_s": user_cpu_s + system_cpu_s,
        "thread_count": int(stat_tail[17]),
        "voluntary_ctxt_switches": voluntary_ctxt_switches,
        "nonvoluntary_ctxt_switches": nonvoluntary_ctxt_switches,
        "sched_sum_exec_runtime_ms": float(
            sched_values.get("se.sum_exec_runtime", 0.0)
        ),
        "sched_nr_migrations": int(sched_values.get("se.nr_migrations", 0)),
        "sched_nr_switches": int(sched_values.get("nr_switches", 0)),
        "sched_nr_voluntary_switches": int(
            sched_values.get("nr_voluntary_switches", 0)
        ),
        "sched_nr_involuntary_switches": int(
            sched_values.get("nr_involuntary_switches", 0)
        ),
        "task_sched_sum_exec_runtime_ms": float(
            task_sched_values.get("se.sum_exec_runtime", 0.0)
        ),
        "task_sched_nr_migrations": int(
            task_sched_values.get("se.nr_migrations", 0)
        ),
        "task_sched_nr_switches": int(task_sched_values.get("nr_switches", 0)),
        "task_sched_nr_voluntary_switches": int(
            task_sched_values.get("nr_voluntary_switches", 0)
        ),
        "task_sched_nr_involuntary_switches": int(
            task_sched_values.get("nr_involuntary_switches", 0)
        ),
    }


def _process_resource_delta(
    before: dict[str, int | float] | None,
    after: dict[str, int | float] | None,
) -> dict[str, int | float] | None:
    if before is None or after is None:
        return None
    return {
        "rss_bytes_delta": int(after["rss_bytes"]) - int(before["rss_bytes"]),
        "rss_peak_bytes": after["rss_peak_bytes"],
        "user_cpu_s_delta": float(after["user_cpu_s"]) - float(before["user_cpu_s"]),
        "system_cpu_s_delta": float(after["system_cpu_s"])
        - float(before["system_cpu_s"]),
        "total_cpu_s_delta": float(after["total_cpu_s"]) - float(before["total_cpu_s"]),
        "thread_count_end": after["thread_count"],
        "voluntary_ctxt_switches_delta": int(after["voluntary_ctxt_switches"])
        - int(before["voluntary_ctxt_switches"]),
        "nonvoluntary_ctxt_switches_delta": int(
            after["nonvoluntary_ctxt_switches"]
        )
        - int(before["nonvoluntary_ctxt_switches"]),
        "sched_sum_exec_runtime_ms_delta": float(
            after["sched_sum_exec_runtime_ms"]
        )
        - float(before["sched_sum_exec_runtime_ms"]),
        "sched_nr_migrations_delta": int(after["sched_nr_migrations"])
        - int(before["sched_nr_migrations"]),
        "sched_nr_switches_delta": int(after["sched_nr_switches"])
        - int(before["sched_nr_switches"]),
        "sched_nr_voluntary_switches_delta": int(
            after["sched_nr_voluntary_switches"]
        )
        - int(before["sched_nr_voluntary_switches"]),
        "sched_nr_involuntary_switches_delta": int(
            after["sched_nr_involuntary_switches"]
        )
        - int(before["sched_nr_involuntary_switches"]),
        "task_sched_sum_exec_runtime_ms_delta": float(
            after["task_sched_sum_exec_runtime_ms"]
        )
        - float(before["task_sched_sum_exec_runtime_ms"]),
        "task_sched_nr_migrations_delta": int(after["task_sched_nr_migrations"])
        - int(before["task_sched_nr_migrations"]),
        "task_sched_nr_switches_delta": int(after["task_sched_nr_switches"])
        - int(before["task_sched_nr_switches"]),
        "task_sched_nr_voluntary_switches_delta": int(
            after["task_sched_nr_voluntary_switches"]
        )
        - int(before["task_sched_nr_voluntary_switches"]),
        "task_sched_nr_involuntary_switches_delta": int(
            after["task_sched_nr_involuntary_switches"]
        )
        - int(before["task_sched_nr_involuntary_switches"]),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile < 0.0 or percentile > 1.0:
        raise ValueError("percentile must be in [0.0, 1.0]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    return (
        ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def _latency_summary(latencies_ms: list[float]) -> dict[str, object]:
    return {
        "count": len(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": _percentile(latencies_ms, 0.50),
        "p95_ms": _percentile(latencies_ms, 0.95),
        "p99_ms": _percentile(latencies_ms, 0.99),
        "values_ms": latencies_ms,
    }


def _throughput_summary(
    request_count: int,
    elapsed_s: float,
    clients: int,
    iterations: int,
) -> dict[str, int | float]:
    return {
        "client_count": clients,
        "iterations_per_client": iterations,
        "total_elapsed_s": elapsed_s,
        "requests_per_s": request_count / elapsed_s if elapsed_s else 0.0,
    }


def _run_client_requests(
    port: int,
    request: str,
    iterations: int,
    client_index: int,
) -> list[float]:
    context = zmq.Context.instance()
    client = MessageQueueClient(f"tcp://127.0.0.1:{port}", context)
    lookup_key = _lookup_key_for_request(request, client_index)
    latencies_ms: list[float] = []
    try:
        for _ in range(iterations):
            start = time.perf_counter()
            _submit_request(client, request, lookup_key)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        client.close()
    return latencies_ms


def _run_benchmark_requests(
    port: int,
    request: str,
    iterations: int,
    clients: int,
) -> tuple[list[float], float]:
    start = time.perf_counter()
    if clients == 1:
        return _run_client_requests(port, request, iterations, 0), (
            time.perf_counter() - start
        )

    latencies_ms: list[float] = []
    with ThreadPoolExecutor(max_workers=clients) as executor:
        futures = [
            executor.submit(_run_client_requests, port, request, iterations, index)
            for index in range(clients)
        ]
        for future in as_completed(futures):
            latencies_ms.extend(future.result())
    return latencies_ms, time.perf_counter() - start


def run_one(
    server: str,
    iterations: int,
    request: str,
    clients: int,
) -> dict[str, object]:
    port = _free_port()
    temp_dir = (
        tempfile.TemporaryDirectory() if request == "lookup-fs-l2-partial" else None
    )
    l2_dir = Path(temp_dir.name) / "l2" if temp_dir is not None else None
    if l2_dir is not None:
        _seed_fs_l2_partial_hit(l2_dir)
    proc = subprocess.Popen(
        _server_command(server, port, l2_dir),
        stderr=subprocess.DEVNULL,
    )
    resources_start = None
    resources_end = None
    latencies_ms: list[float] = []
    elapsed_s = 0.0
    try:
        _wait_for_ping(port)
        resources_start = _read_process_resources(proc.pid)
        latencies_ms, elapsed_s = _run_benchmark_requests(
            port,
            request,
            iterations,
            clients,
        )
        resources_end = _read_process_resources(proc.pid)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if temp_dir is not None:
            temp_dir.cleanup()

    report = _latency_summary(latencies_ms)
    report.update(
        _throughput_summary(
            len(latencies_ms),
            elapsed_s,
            clients,
            iterations,
        )
    )
    report["server_resources_start"] = resources_start
    report["server_resources_end"] = resources_end
    report["server_resources_delta"] = _process_resource_delta(
        resources_start,
        resources_end,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help="Number of concurrent clients. Each client sends --iterations requests.",
    )
    parser.add_argument(
        "--request",
        choices=["ping", "noop", "lookup-miss", "lookup-fs-l2-partial"],
        default="ping",
        help=(
            "MP request to measure. lookup-miss and lookup-fs-l2-partial send "
            "token-key LOOKUP requests that do not require registered CUDA KV "
            "cache."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.clients < 1:
        raise ValueError("--clients must be at least 1")

    report = {
        "scope": (
            "MP controller-envelope latency only; not KV data-path parity. "
            "server_resources_delta samples the MP server process around the "
            "measured request loop."
        ),
        "request": args.request,
        "python": run_one("python", args.iterations, args.request, args.clients),
        "native": run_one("native", args.iterations, args.request, args.clients),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
