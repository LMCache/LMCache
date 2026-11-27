# SPDX-License-Identifier: Apache-2.0
"""Compares the request transports' server cost between two LMCache trees.

Each engine sends the control RPCs a vLLM engine sends per request
(lookup, query_prefetch_status, free_lookup_locks, end_session) at a fixed
rate. Base and PR run in turns on the same machine, and the PR fails if a
transport's server CPU per request or p99 latency got clearly worse.

    python grpc_perf_check.py deps TREE OUT_DIR
    python grpc_perf_check.py compare --base TREE --base-deps DIR \
        --pr TREE --pr-deps DIR [--rounds 3] [--summary FILE]
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import threading
import time

MODES = ["zmq", "grpc", "grpc-aio", "grpc-native"]
# Eight engines at 31 req/s keep every mode below saturation on a 4-CPU runner.
ENGINES = 8
RATE = 31.0
DURATION = 15.0
NUM_TOKENS = 2048
# Same-code runs differed by up to 6% in CPU per request and 15% in p99, so
# these only catch clear regressions.
CPU_TOLERANCE = 0.25
P99_TOLERANCE = 2.0
P99_FLOOR_MS = 1.0
RUN_TIMEOUT = 300


def _pin(path: Path, name: str) -> str:
    match = re.search(rf"(?<![\w-]){re.escape(name)}(==[^;#\s]+)", path.read_text())
    if match is None:
        raise RuntimeError(f"no {name}==<version> pin in {path}")
    return name + match.group(1)


def deps(tree: Path, out: Path) -> None:
    """Installs the grpcio, grpcio-tools and pygrpc-native versions a tree pins.

    pygrpc-native is pinned in the CPU device workflow, not in requirements.

    Args:
        tree: LMCache checkout to read the pins from.
        out: Directory to install the packages into, for ``PYTHONPATH``.

    Raises:
        RuntimeError: If a pin is missing.
    """
    reqs = tree / "requirements"
    specs = [
        _pin(reqs / "build.txt", "grpcio"),
        _pin(reqs / "build.txt", "grpcio-tools"),
        _pin(reqs / "common.txt", "grpcio-observability"),
        _pin(tree / ".github/workflows/cpu_device.yml", "pygrpc-native"),
    ]
    pip = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target"]
    subprocess.run([*pip, str(out), *specs], check=True)
    print(" ".join(specs))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(mode: str, port: int) -> None:
    # First Party
    from lmcache.cli.main import main

    args = ["--transport", "zmq" if mode == "zmq" else "grpc"]
    if mode in ("grpc-aio", "grpc-native"):
        args += ["--grpc-server-mode", mode.removeprefix("grpc-")]
    args += ["--host", "127.0.0.1", "--port", str(port)]
    args += ["--l1-size-gb", "1", "--no-l1-use-lazy", "--chunk-size", "256"]
    args += ["--eviction-policy", "LRU", "--supported-transfer-mode", "engine_driven"]
    sys.argv = ["lmcache", "server", *args]
    main()


def _engines(
    url: str, engine_ids: list[int], ready: Queue, go: Event, results: Queue
) -> None:
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheServerKey,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

    clients = {}
    for engine in engine_ids:
        client = RequestClientFactory.create(url)
        for _ in range(300):
            try:
                client.noop().result(1)
                break
            except TimeoutError:
                time.sleep(0.2)
        payload = RegisterEngineDrivenContextPayload(
            instance_id=1000 + engine,
            model_name="m",
            world_size=1,
            block_size=16,
            num_layers=4,
            hidden_dim_size=128,
            dtype_str="bfloat16",
            use_mla=False,
        )
        client.register_kv_cache_engine_driven_context(payload).result(10)
        clients[engine] = client
    pool = ThreadPoolExecutor(max_workers=16 * len(engine_ids))
    futures: list[Future] = []
    latencies: list[float] = []
    lock = threading.Lock()

    def one(client: Any, engine: int, n: int) -> None:
        rng = random.Random(engine * 100003 + n)
        request_id = f"e{engine}_{n}"
        key = IPCCacheServerKey.from_token_ids(
            model_name="m",
            world_size=1,
            worker_id=0,
            token_ids=[rng.randrange(32000) for _ in range(NUM_TOKENS)],
            start=0,
            end=NUM_TOKENS,
            request_id=request_id,
        )
        start = time.perf_counter()
        client.lookup(key, 1).result(30)
        while client.query_prefetch_status(request_id).result(30) is None:
            time.sleep(0.001)
        client.free_lookup_locks(key, 1).result(30)
        client.end_session(request_id).result(30)
        with lock:
            latencies.append(time.perf_counter() - start)

    def drive(engine: int, start: float) -> None:
        rng = random.Random(engine)
        t, n = start, 0
        while True:
            t += rng.expovariate(RATE)
            if t > start + DURATION:
                return
            delay = t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(one, clients[engine], engine, n))
            n += 1

    ready.put(None)
    go.wait()
    start = time.perf_counter()
    drivers = [threading.Thread(target=drive, args=(e, start)) for e in engine_ids]
    for d in drivers:
        d.start()
    for d in drivers:
        d.join()
    pool.shutdown(wait=True)
    errors = [repr(f.exception()) for f in futures if f.exception() is not None]
    results.put((latencies, errors[:1]))
    results.close()
    results.join_thread()
    # Skip interpreter shutdown, which waits on the clients' gRPC threads.
    os._exit(0)


def measure(mode: str) -> None:
    """Runs one fixed-rate replay against a fresh server in the current tree.

    Prints one JSON line with the server's CPU per request, its p99 latency,
    and the share of the scheduled requests that finished.

    Args:
        mode: One of ``MODES``.

    Raises:
        RuntimeError: If a request failed.
    """
    # Third Party
    import psutil

    port = _free_port()
    url = f"{'tcp' if mode == 'zmq' else 'grpc'}://127.0.0.1:{port}"
    ctx = mp.get_context("spawn")
    server = ctx.Process(target=_serve, args=(mode, port), daemon=True)
    ready, go, results = ctx.Queue(), ctx.Event(), ctx.Queue()
    groups = [list(range(i, ENGINES, 4)) for i in range(4)]
    workers = [
        ctx.Process(target=_engines, args=(url, g, ready, go, results), daemon=True)
        for g in groups
    ]
    server.start()
    try:
        for w in workers:
            w.start()
        for _ in workers:
            ready.get(timeout=120)
        server_proc = psutil.Process(server.pid)
        cpu0 = sum(server_proc.cpu_times()[:2])
        go.set()
        latencies: list[float] = []
        for _ in workers:
            worker_latencies, errors = results.get(timeout=DURATION * 10)
            if errors:
                raise RuntimeError(f"a {mode} request failed: {errors[0]}")
            latencies += worker_latencies
        busy = sum(server_proc.cpu_times()[:2]) - cpu0
    finally:
        for p in [*workers, server]:
            p.kill()
            p.join()
    latencies.sort()
    n = len(latencies)
    result = {
        "mode": mode,
        "cpu_ms": busy / n * 1e3,
        "p99_ms": latencies[min(int(n * 0.99), n - 1)] * 1e3,
        "done": n / (ENGINES * RATE * DURATION),
    }
    print(json.dumps(result))


def _run(tree: Path, deps_dir: Path, mode: str) -> dict[str, Any]:
    env = dict(os.environ, PYTHONPATH=f"{deps_dir}{os.pathsep}{tree}")
    command = [sys.executable, __file__, "measure", mode]
    try:
        out = subprocess.run(
            command,
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"{mode} in {tree} timed out:\n" + str(e.stderr)) from e
    lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"{mode} in {tree} failed:\n{out.stderr[-3000:]}")
    return json.loads(lines[-1])


def compare(args: argparse.Namespace) -> int:
    """Runs base and PR in turns for every mode and judges the medians.

    Args:
        args: Parsed ``compare`` arguments.

    Returns:
        1 if any mode got worse past the tolerances, else 0.
    """
    sides = {"base": (args.base, args.base_deps), "pr": (args.pr, args.pr_deps)}
    runs: dict[tuple[str, str], list[dict[str, Any]]] = {
        (side, mode): [] for side in sides for mode in MODES
    }
    for r in range(args.rounds):
        for mode in MODES:
            for side in ["base", "pr"] if r % 2 == 0 else ["pr", "base"]:
                result = _run(*sides[side], mode)
                runs[side, mode].append(result)
                print(f"round {r} {side:4} {json.dumps(result)}", flush=True)

    def med(side: str, mode: str, key: str) -> float:
        return statistics.median(x[key] for x in runs[side, mode])

    lines = [
        "| Mode | CPU/req base | CPU/req PR | p99 base | p99 PR | Result |",
        "|---|--:|--:|--:|--:|---|",
    ]
    failed = False
    for mode in MODES:
        cpu_b, cpu_p = med("base", mode, "cpu_ms"), med("pr", mode, "cpu_ms")
        p99_b, p99_p = med("base", mode, "p99_ms"), med("pr", mode, "p99_ms")
        problems = []
        if cpu_p > cpu_b * (1 + CPU_TOLERANCE):
            problems.append(f"CPU/req +{(cpu_p / cpu_b - 1) * 100:.0f}%")
        if p99_p > p99_b * P99_TOLERANCE and p99_p - p99_b > P99_FLOOR_MS:
            problems.append(f"p99 {p99_p / p99_b:.1f}x")
        if med("pr", mode, "done") < 0.95:
            problems.append("requests left unfinished")
        failed = failed or bool(problems)
        cells = [f"{cpu_b:.3f} ms", f"{cpu_p:.3f} ms", f"{p99_b:.2f} ms"]
        cells += [f"{p99_p:.2f} ms", ", ".join(problems) or "ok"]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    print(table)
    if args.summary:
        with open(args.summary, "a") as f:
            title = f"Request transport cost, {ENGINES} engines at {RATE:g} req/s"
            f.write(f"### {title}\n\n{table}")
    return 1 if failed else 0


def main() -> None:
    """Parses the command line and runs ``deps``, ``compare`` or ``measure``."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("deps")
    p.add_argument("tree", type=Path)
    p.add_argument("out", type=Path)
    p = sub.add_parser("compare")
    for name in ("base", "base-deps", "pr", "pr-deps"):
        p.add_argument(f"--{name}", type=lambda s: Path(s).resolve(), required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--summary")
    p = sub.add_parser("measure")
    p.add_argument("mode", choices=MODES)
    args = parser.parse_args()

    if args.command == "deps":
        deps(args.tree, args.out)
    elif args.command == "measure":
        measure(args.mode)
    else:
        sys.exit(compare(args))


if __name__ == "__main__":
    main()
