# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench server`` subcommand implementation.

This module re-exports :func:`add_server_arguments` from the lightweight
arguments module and provides the execution orchestrator :func:`run_server_bench`
for the end-to-end LMCache MP cache-server sanity test.

The command exercises the full store / retrieve data path:

    For each request:
      1. LOOKUP   — submit prefix lookup (void reply)
      2. QUERY_PREFETCH_STATUS — poll by request_id until done
      3. RETRIEVE — for the hit portion (if any)
      4. STORE    — for the miss portion
      5. CHECKSUM — verify retrieved chunks in reader buffers

Usage examples::

    # GPU mode: real CUDA tensors + IPC
    lmcache bench server --rpc-url tcp://localhost:5555 \\
        --num-tokens 512 --start 0 --end 3

    # Custom KV cache shape (multi-group spec)
    lmcache bench server --rpc-url tcp://localhost:5555 \\
        --kvcache-shape-spec '(2,32,1024,8,128):float16:32'

    # Run forever starting from sequence 0
    lmcache bench server --rpc-url tcp://localhost:5555
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import argparse
import math
import os
import sys
import threading
import time

# First Party
from lmcache.cli.commands.bench.server_bench.arguments import add_server_arguments
from lmcache.cli.commands.bench.server_bench.cases.base import BenchResult
from lmcache.cli.commands.bench.server_bench.cases.baseline import (
    BaselineBenchCase,
)
from lmcache.cli.commands.bench.server_bench.cases.concurrent import (
    run_concurrently,
)
from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient
from lmcache.cli.commands.bench.server_bench.config import (
    BenchConfig,
    parse_args_to_config,
)
from lmcache.cli.profiling import (
    PY_SPY_MODES,
    FlameProfiler,
    ProfileError,
    check_profiling_deps,
    default_output_path,
    resolve_flamegraph_dir,
)

if TYPE_CHECKING:
    # Standard
    from collections.abc import Callable

    # First Party
    from lmcache.cli.commands.base import BaseCommand


__all__ = (
    "add_server_arguments",
    "run_server_bench",
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_server_profiler(
    args: argparse.Namespace,
    log: "Callable[[str], None]",
) -> "FlameProfiler | None":
    """Build a profiler attached to the server, or ``None`` if disabled.

    Validates the target pid and toolchain eagerly so a misconfigured run
    fails before any load is sent. The returned profiler is not started;
    the caller wraps the load loop with ``start`` / ``stop``.

    Args:
        args: Parsed CLI arguments for ``lmcache bench server``.
        log: Progress logger.

    Returns:
        A ready :class:`FlameProfiler` targeting ``--profile-server-pid``,
        or ``None`` when ``--flamegraph`` is off.
    """
    if getattr(args, "flamegraph", "off") != "on":
        return None

    pid = args.profile_server_pid
    if pid <= 0:
        print(
            "Error: --flamegraph on requires --profile-server-pid "
            "(the pid of the running 'lmcache server').",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"Error: no such process: --profile-server-pid {pid}", file=sys.stderr)
        sys.exit(2)
    except PermissionError:
        print(
            f"Error: server pid {pid} belongs to another user; "
            "profiling it needs root.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        check_profiling_deps(args.flamegraph_mode)
        flamegraph_dir = ""
        if args.flamegraph_mode not in PY_SPY_MODES:
            flamegraph_dir = resolve_flamegraph_dir(args.flamegraph_scripts_dir, log)
        output = args.flamegraph_output or default_output_path(
            f"server-pid{pid}", args.flamegraph_mode
        )
        return FlameProfiler(
            mode=args.flamegraph_mode,
            output=output,
            flamegraph_dir=flamegraph_dir,
            pid=pid,
            title=f"{args.flamegraph_mode} (server pid {pid})",
        )
    except ProfileError as e:
        print(
            "Error: --flamegraph on was requested but profiling is "
            f"unavailable:\n  {e}",
            file=sys.stderr,
        )
        sys.exit(2)


def _build_cases(
    config: BenchConfig,
    args: argparse.Namespace,
    stop: threading.Event,
) -> list[BaselineBenchCase]:
    """Deal the ``[--start, --end)`` sequences round-robin across clients."""
    concurrency = config.concurrency
    total = None if args.end is None else max(0, args.end - args.start)
    return [
        BaselineBenchCase(
            sequence_count=(
                None if total is None else len(range(index, total, concurrency))
            ),
            sequence_id_offset=args.start + index,
            sequence_id_stride=concurrency,
            interval_seconds=args.interval,
            num_tokens=config.num_tokens,
            stop=stop,
        )
        for index in range(concurrency)
    ]


def run_server_bench(
    command: "BaseCommand",
    args: argparse.Namespace,
) -> None:
    """Centralized orchestrator: run the server bench loop.

    Args:
        command: The owning :class:`BaseCommand` instance, used to
            obtain a configured :class:`Metrics` object via
            ``command.create_metrics``.
        args: Parsed CLI arguments for ``lmcache bench server``.
    """
    config = parse_args_to_config(args)
    stop = threading.Event()
    cases = _build_cases(config, args, stop)

    def log(msg: str) -> None:
        """Print progress messages; suppressed by --quiet."""
        if not args.quiet:
            print(msg)

    # The profiler targets the server process (--profile-server-pid), not
    # this benchmark client. Build it before opening any connection so a
    # bad pid or a missing toolchain fails immediately, not after a full
    # benchmark has already run. ``None`` when --flamegraph is off.
    profiler = _build_server_profiler(args, log)

    result = BenchResult(case_name=cases[0].name)
    clients: list[ServerBenchClient] = []
    try:
        for index in range(config.concurrency):
            client = ServerBenchClient(config, log, client_index=index)
            clients.append(client)
            client.start()

        # Record only the steady-state load, not the one-time registration.
        if profiler is not None:
            profiler.start(log)
        started_at = time.monotonic()
        if len(clients) == 1:
            result = cases[0].run(clients[0], log)
        else:
            result = run_concurrently(clients, cases, stop, log)
        result.elapsed_seconds = time.monotonic() - started_at
        if result.interrupted:
            log("\nStopping...")
    except RuntimeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        log("\nStopping...")
    finally:
        # Stop recording once load ends, before teardown
        if profiler is not None:
            profiler.stop(log)
        for client in clients:
            client.close()

    # Emit structured metrics summary.
    _emit_server_bench_metrics(
        command=command,
        args=args,
        result=result,
    )
    log("Done.")


def _emit_server_bench_metrics(
    command: "BaseCommand",
    args: argparse.Namespace,
    result: BenchResult,
) -> None:
    """Emit server bench summary using the CLI metrics system.

    Args:
        command: The owning :class:`BaseCommand` instance.
        args: Parsed CLI arguments.
        result: Structured result from the executed bench case.
    """
    if result.completed_runs == 0:
        return

    total_checksum_ok = result.passed_count("checksum_match")
    total_checksum_fail = result.failed_count("checksum_match")

    metrics = command.create_metrics("Server Bench Result", args, width=64)

    cfg_section = metrics.add_section("config", "Configuration")
    cfg_section.add("rpc_url", "RPC URL", args.rpc_url)
    cfg_section.add("mode", "Mode", args.mode)
    cfg_section.add(
        "transfer_mode", "Transfer mode", getattr(args, "transfer_mode", "auto")
    )
    cfg_section.add("num_tokens", "Tokens / request", args.num_tokens)
    cfg_section.add("interval", "Interval (s)", args.interval)
    cfg_section.add("concurrency", "Concurrency", getattr(args, "concurrency", 1))

    result_section = metrics.add_section("results", "Results")
    result_section.add("total_requests", "Total requests", result.completed_runs)
    result_section.add("checksum_ok", "Checksum OK", total_checksum_ok)
    result_section.add("checksum_fail", "Checksum FAIL", total_checksum_fail)
    server_checks = [
        check
        for kind in ("cold", "warm")
        for check in result.checks.get(f"{kind}.server_checksum_match", [])
    ]
    if server_checks:
        result_section.add(
            "server_checksum_ok", "Server checksum OK", sum(server_checks)
        )
        result_section.add(
            "server_checksum_fail", "Server checksum FAIL", server_checks.count(False)
        )
    if result.completed_runs > 0:
        pass_rate = total_checksum_ok / result.completed_runs * 100
        result_section.add("pass_rate", "Checksum pass rate (%)", round(pass_rate, 2))
    if result.elapsed_seconds > 0:
        result_section.add("elapsed_s", "Elapsed (s)", round(result.elapsed_seconds, 3))
        result_section.add(
            "requests_per_s",
            "Requests / s",
            round(result.completed_runs / result.elapsed_seconds, 2),
        )

    # Per-operation latency summary (cold pass).
    _add_latency_section(
        metrics,
        "cold_lookup",
        "Cold Lookup (ms)",
        result.latencies_ms.get("cold.lookup"),
    )
    _add_latency_section(
        metrics,
        "cold_store",
        "Cold Store (ms)",
        result.latencies_ms.get("cold.store"),
    )

    # Per-operation latency summary (warm pass).
    _add_latency_section(
        metrics,
        "warm_lookup",
        "Warm Lookup (ms)",
        result.latencies_ms.get("warm.lookup"),
    )
    _add_latency_section(
        metrics,
        "warm_retrieve",
        "Warm Retrieve (ms)",
        result.latencies_ms.get("warm.retrieve"),
    )

    metrics.emit()


def _add_latency_section(
    metrics,
    section_id: str,
    section_title: str,
    latencies: list[float] | None,
) -> None:
    """Add count and latency statistics when samples are available."""
    if not latencies:
        return

    sorted_lat = sorted(latencies)
    count = len(sorted_lat)
    mean = sum(sorted_lat) / count
    p50_idx = max(0, math.ceil(count * 0.50) - 1)
    p99_idx = max(0, math.ceil(count * 0.99) - 1)

    section = metrics.add_section(section_id, section_title)
    section.add(f"{section_id}_count", "count", count)
    section.add(f"{section_id}_mean", "mean", round(mean, 3))
    section.add(f"{section_id}_min", "min", round(sorted_lat[0], 3))
    section.add(f"{section_id}_max", "max", round(sorted_lat[-1], 3))
    section.add(f"{section_id}_p50", "p50", round(sorted_lat[p50_idx], 3))
    section.add(f"{section_id}_p99", "p99", round(sorted_lat[p99_idx], 3))
