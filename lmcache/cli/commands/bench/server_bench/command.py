# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench server`` subcommand implementation.

This module provides argument registration via :func:`add_server_arguments`
and the execution orchestrator :func:`run_server_bench` for the end-to-end
LMCache MP cache-server sanity test.

The command exercises the full store / retrieve data path:

    For each request:
      1. LOOKUP   — submit prefix lookup (void reply)
      2. QUERY_PREFETCH_STATUS — poll by request_id until done
      3. RETRIEVE — for the hit portion (if any)
      4. STORE    — for the miss portion
      5. CHECKSUM — verify KV cache integrity via HTTP API

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
from dataclasses import dataclass
from typing import TYPE_CHECKING
import argparse
import itertools
import math
import os
import sys
import time

# First Party
from lmcache.cli.commands.bench.server_bench.client import (
    LookupResult,
    ServerBenchClient,
    TransferResult,
)
from lmcache.cli.commands.bench.server_bench.config import parse_args_to_config

# Heavy imports reused by the orchestrator. ``DTYPE_MAP`` is required
# for the ``--kvcache-shape-spec`` help string at parser-registration
# time. On a slim install these symbols are placeholders; the
# ``_require_full_install`` guard inside the helpers module keeps
# orchestration safe.
from lmcache.cli.commands.bench.server_bench.helpers import (
    _DEFAULT_SHAPE_SPEC,
    DTYPE_MAP,
    _require_full_install,
)

if TYPE_CHECKING:
    # Standard
    from collections.abc import Callable

    # First Party
    from lmcache.cli.commands.base import BaseCommand
    from lmcache.cli.profiling import FlameProfiler


# Stash the original (full-install) ImportError so the parser-stub
# branch and the orchestrator branch can both surface it verbatim.
__all__ = (
    "add_server_arguments",
    "run_server_bench",
)


@dataclass
class _RequestPassResult:
    """Results for one baseline pass."""

    lookup: LookupResult
    checksums: list[str] | None = None
    retrieve: TransferResult | None = None
    store: TransferResult | None = None


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench server`` arguments to *parser*.

    Requires the full LMCache install (torch, zmq, etc.).
    Callers should check ``_IMPORT_ERROR`` before calling this.

    Args:
        parser: The ``ArgumentParser`` for the server bench subcommand.
    """

    parser.add_argument(
        "--rpc-url",
        default="tcp://localhost:5555",
        help=("ZMQ endpoint of the MP server (default: tcp://localhost:5555)"),
    )
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu"],
        default="gpu",
        help=(
            "Run mode (default: gpu). In cpu mode the client allocates "
            "POSIX-SHM-backed KV cache tensors and the server maps the "
            "same physical pages."
        ),
    )
    parser.add_argument(
        "--transfer-mode",
        choices=["auto", "engine_driven", "lmcache_driven"],
        default="auto",
        help=(
            "Transport routing for STORE/RETRIEVE (default: auto). "
            "`lmcache_driven` forces the server-driven handle path "
            "(REGISTER_KV_CACHE + STORE/RETRIEVE), which supports "
            "both CUDA IPC and CPU SHM for zero-copy transfers. "
            "`engine_driven` forces the worker-side gather/scatter "
            "data path (REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT + "
            "PREPARE/COMMIT). "
            "`auto` keeps the historical mapping: "
            "gpu->lmcache_driven, cpu->engine_driven."
        ),
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help=(
            "Simulated tensor-parallel world size (default: 1). Each "
            "rank registers its own KV cache under a distinct "
            "instance_id, and STORE / RETRIEVE fan out per rank the "
            "same way LMCacheMPWorkerAdapter routes them in a real "
            "vLLM deployment (MLA -> only rank 0 stores; non-MLA -> "
            "every rank stores; every rank always retrieves)."
        ),
    )
    parser.add_argument(
        "--use-mla",
        action="store_true",
        default=False,
        help=(
            "MLA mode: fold all TP ranks into a single kv_worker "
            "so only rank 0 writes KV and every rank retrieves "
            "the shared KV object (default: False). "
            "Also implied when --kvcache-shape-spec declares "
            "kv_size=1."
        ),
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=512,
        help="Tokens per request (default: 512)",
    )

    # -- KV cache shape --
    kv = parser.add_argument_group("KV cache shape")
    kv.add_argument(
        "--kvcache-shape-spec",
        type=str,
        default=_DEFAULT_SHAPE_SPEC,
        help=(
            "KV shape spec. Describes one or more KV layer groups "
            "separated by ';'. "
            "Grammar: "
            "'(kv_size,NB,BS,NH,HS):dtype:layers[;(...):dtype:layers...]'. "
            "Fields: kv_size=2 for classical K/V or 1 for MLA, "
            "NB=num_blocks, BS=block_size (tokens/block), "
            "NH=num_heads, HS=head_size (elements). "
            "dtype is the element dtype (supported: %s); 'uint8' "
            "is used for FP8-quantized KV. 'layers' is the number "
            "of consecutive layers sharing this group's geometry. "
            "Multi-group example (MLA + classical attention): "
            "'(1,1024,16,1,128):float16:4;"
            "(2,1024,16,8,128):float16:28'. "
            "All groups must share the same NB and BS. "
            "See lmcache.v1.kv_layer_groups.parse_kvcache_shape_spec "
            "for the authoritative parser. Default: '%s'"
            % (", ".join(DTYPE_MAP.keys()), _DEFAULT_SHAPE_SPEC)
        ),
    )
    kv.add_argument(
        "--num-blocks",
        type=int,
        default=1024,
        help="Paged blocks (default: 1024)",
    )
    kv.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per block (default: 16)",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting sequence number (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help=("Ending sequence number (exclusive). If not set, runs forever."),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help=("Seconds between requests (default: 0.5)"),
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help=("HTTP base URL for checksum API (default: http://localhost:8080)"),
    )

    prof = parser.add_argument_group(
        "server profiling",
        "Flame-graph the MP server process while this benchmark drives "
        "load into it. The server's store path (hashing, allocation, "
        "gather, D2H) runs in its own process, not in this client, so "
        "profiling attaches to --profile-server-pid rather than to the "
        "benchmark. See 'lmcache tool flamegraph' for the standalone form.",
    )
    prof.add_argument(
        "--flamegraph",
        choices=["on", "off"],
        default="off",
        help="Record a flame graph of the server during the run (default: off).",
    )
    prof.add_argument(
        "--profile-server-pid",
        type=int,
        default=0,
        metavar="PID",
        help=(
            "Server process to profile, e.g. $(pgrep -f 'lmcache server'). "
            "Required when --flamegraph on."
        ),
    )
    prof.add_argument(
        "--flamegraph-mode",
        default="gil",
        metavar="MODE[,MODE...]",
        help=(
            "What to sample in the server (default: gil). Pass several "
            "comma-separated to profile one load run per mode. Modes: on-cpu, "
            "off-cpu, wakeup, offwake (perf/bcc), wall, gil (py-spy). perf/bcc "
            "name Python functions only when the server was launched with "
            "PYTHONPERFSUPPORT=1. See the 'lmcache tool flamegraph' docs."
        ),
    )
    prof.add_argument(
        "--flamegraph-output",
        default="",
        metavar="PATH",
        help=(
            "SVG output path. Default: "
            "/tmp/lmcache_bench_flames/server-pid<PID>.<mode>.svg."
        ),
    )
    prof.add_argument(
        "--flamegraph-scripts-dir",
        default="",
        metavar="DIR",
        help=(
            "Directory with the FlameGraph scripts (flamegraph.pl, "
            "stackcollapse-perf.pl); default ~/FlameGraph (cloned there on "
            "first use). Unused by --flamegraph-mode wall / gil, which "
            "render their own SVG."
        ),
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

    # First Party
    from lmcache.cli.profiling import (
        PY_SPY_MODES,
        FlameProfiler,
        ProfileError,
        check_profiling_deps,
        default_output_path,
        resolve_flamegraph_dir,
    )

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
    _require_full_install()
    config = parse_args_to_config(args)

    def log(msg: str) -> None:
        """Print progress messages; suppressed by --quiet."""
        if not config.quiet:
            print(msg)

    # The profiler targets the server process (--profile-server-pid), not
    # this benchmark client. Build it before opening any connection so a
    # bad pid or a missing toolchain fails immediately, not after a full
    # benchmark has already run. ``None`` when --flamegraph is off.
    profiler = _build_server_profiler(args, log)

    total_requests = 0
    total_checksum_ok = 0
    total_checksum_fail = 0

    # Latency collectors: keyed by (pass_label, op_type).
    # Each entry is a list of latency values in ms.
    cold_lookup_ms: list[float] = []
    cold_store_ms: list[float] = []
    warm_lookup_ms: list[float] = []
    warm_retrieve_ms: list[float] = []

    client = ServerBenchClient(config, log)
    try:
        client.start()

        # Record only the steady-state load, not the one-time registration.
        if profiler is not None:
            profiler.start(log)

        if config.end is not None:
            seq_iter: itertools.count | range = range(config.start, config.end)
        else:
            seq_iter = itertools.count(config.start)

        for seq_no in seq_iter:
            log("=== Request seq=%d ===" % seq_no)

            # Pass 1: cold (miss -> store)
            cold_result = _run_request_pass(client, seq_no, "cold")
            if cold_result is not None:
                cold_lookup_ms.append(cold_result.lookup.latency_ms)
                if cold_result.store is not None:
                    cold_store_ms.append(cold_result.store.latency_ms)

            time.sleep(args.interval)

            # Pass 2: warm (hit -> retrieve)
            warm_result = _run_request_pass(client, seq_no, "warm")
            if warm_result is not None:
                warm_lookup_ms.append(warm_result.lookup.latency_ms)
                if warm_result.retrieve is not None:
                    warm_retrieve_ms.append(warm_result.retrieve.latency_ms)

            # Compare checksums
            total_requests += 1
            cold_checksums = cold_result.checksums if cold_result else None
            warm_checksums = warm_result.checksums if warm_result else None
            if cold_checksums and warm_checksums:
                if cold_checksums == warm_checksums:
                    total_checksum_ok += 1
                    log("  [seq %d] CHECKSUM MATCH OK" % seq_no)
                else:
                    total_checksum_fail += 1
                    log("  [seq %d] CHECKSUM MISMATCH!" % seq_no)
                    for i, (c, w) in enumerate(
                        zip(
                            cold_checksums,
                            warm_checksums,
                            strict=False,
                        )
                    ):
                        log(
                            "    chunk %d: cold=%s warm=%s %s"
                            % (
                                i,
                                c[:12],
                                w[:12],
                                ("OK" if c == w else "FAIL"),
                            )
                        )

            log("")
            time.sleep(args.interval)
    except RuntimeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        log("\nStopping...")
    finally:
        # Stop recording once load ends, before teardown
        if profiler is not None:
            profiler.stop(log)
        client.close()

    # Emit structured metrics summary.
    _emit_server_bench_metrics(
        command=command,
        args=args,
        total_requests=total_requests,
        total_checksum_ok=total_checksum_ok,
        total_checksum_fail=total_checksum_fail,
        cold_lookup_ms=cold_lookup_ms,
        cold_store_ms=cold_store_ms,
        warm_lookup_ms=warm_lookup_ms,
        warm_retrieve_ms=warm_retrieve_ms,
    )
    log("Done.")


def _run_request_pass(
    client: ServerBenchClient,
    seq_no: int,
    pass_label: str,
) -> _RequestPassResult | None:
    """Run one cold or warm baseline pass."""
    if pass_label not in ("cold", "warm"):
        raise ValueError("unsupported baseline pass: %s" % pass_label)

    request = client.create_request(
        seq_no,
        request_id="req-%d-%s" % (seq_no, pass_label),
        label=pass_label,
    )
    if request is None:
        return None

    lookup = client.lookup(request)
    if not lookup.succeeded:
        return None

    hit_tokens = lookup.hit_chunks * request.chunk_size
    miss_tokens = request.num_full_tokens - hit_tokens

    checksums: list[str] | None = None
    if pass_label == "cold" and miss_tokens > 0:
        checksums = client.compute_checksums(
            request,
            start_token=hit_tokens,
            token_count=miss_tokens,
        )
    if pass_label == "warm" and hit_tokens > 0:
        client.zero_destination(
            request,
            start_token=0,
            token_count=hit_tokens,
        )

    retrieve = client.retrieve(
        request,
        start_token=0,
        token_count=hit_tokens,
    )
    store = client.store(
        request,
        start_token=hit_tokens,
        token_count=miss_tokens,
    )

    if pass_label == "warm" and hit_tokens > 0:
        checksums = client.compute_checksums(
            request,
            start_token=0,
            token_count=hit_tokens,
        )

    client.end_session(request)
    return _RequestPassResult(
        lookup=lookup,
        checksums=checksums,
        retrieve=retrieve,
        store=store,
    )


def _emit_server_bench_metrics(
    command: "BaseCommand",
    args: argparse.Namespace,
    total_requests: int,
    total_checksum_ok: int,
    total_checksum_fail: int,
    cold_lookup_ms: list[float] | None = None,
    cold_store_ms: list[float] | None = None,
    warm_lookup_ms: list[float] | None = None,
    warm_retrieve_ms: list[float] | None = None,
) -> None:
    """Emit server bench summary using the CLI metrics system.

    Args:
        command: The owning :class:`BaseCommand` instance.
        args: Parsed CLI arguments.
        total_requests: Total number of request pairs processed.
        total_checksum_ok: Number of requests with matching checksums.
        total_checksum_fail: Number of requests with mismatched checksums.
        cold_lookup_ms: Per-request cold lookup latencies (ms).
        cold_store_ms: Per-request cold store latencies (ms).
        warm_lookup_ms: Per-request warm lookup latencies (ms).
        warm_retrieve_ms: Per-request warm retrieve latencies (ms).
    """
    if total_requests == 0:
        return

    metrics = command.create_metrics("Server Bench Result", args, width=64)

    cfg_section = metrics.add_section("config", "Configuration")
    cfg_section.add("rpc_url", "RPC URL", args.rpc_url)
    cfg_section.add("mode", "Mode", args.mode)
    cfg_section.add(
        "transfer_mode", "Transfer mode", getattr(args, "transfer_mode", "auto")
    )
    cfg_section.add("num_tokens", "Tokens / request", args.num_tokens)
    cfg_section.add("interval", "Interval (s)", args.interval)

    result_section = metrics.add_section("results", "Results")
    result_section.add("total_requests", "Total requests", total_requests)
    result_section.add("checksum_ok", "Checksum OK", total_checksum_ok)
    result_section.add("checksum_fail", "Checksum FAIL", total_checksum_fail)
    if total_requests > 0:
        pass_rate = total_checksum_ok / total_requests * 100
        result_section.add("pass_rate", "Pass rate (%)", round(pass_rate, 2))

    # Per-operation latency summary (cold pass).
    _add_latency_section(metrics, "cold_lookup", "Cold Lookup (ms)", cold_lookup_ms)
    _add_latency_section(metrics, "cold_store", "Cold Store (ms)", cold_store_ms)

    # Per-operation latency summary (warm pass).
    _add_latency_section(metrics, "warm_lookup", "Warm Lookup (ms)", warm_lookup_ms)
    _add_latency_section(
        metrics, "warm_retrieve", "Warm Retrieve (ms)", warm_retrieve_ms
    )

    metrics.emit()


def _add_latency_section(
    metrics,
    section_id: str,
    section_title: str,
    latencies: list[float] | None,
) -> None:
    """Add a latency summary section to the metrics report.

    Computes count, mean, min, max, p50, and p99 from the raw
    latency list. Skipped if the list is empty or None.

    Args:
        metrics: The :class:`Metrics` instance.
        section_id: Unique section identifier.
        section_title: Human-readable section title.
        latencies: Raw latency values in milliseconds.
    """
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
