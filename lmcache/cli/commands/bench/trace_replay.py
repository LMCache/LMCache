# SPDX-License-Identifier: Apache-2.0

"""``lmcache bench trace-replay`` — benchmark flavor of trace replay.

Wraps the same
:class:`~lmcache.tools.trace_replay.driver.StorageReplayDriver`
used by ``lmcache trace replay``, adding:

* CSV + JSON export of per-qualname stats, in the output directory
  convention the rest of ``lmcache bench`` uses.
* Metrics-system summary using
  :class:`~lmcache.cli.metrics.Metrics` so the result table matches
  ``lmcache bench engine``.

Kept deliberately small: no workload generation, no network, no vLLM
dependency.  The entire driver runs in-process against a local
StorageManager.
"""

# Future
from __future__ import annotations

# Standard
import argparse
import os
import sys

# First Party
from lmcache.cli.metrics import Metrics, StreamHandler, get_formatter
from lmcache.logging import init_logger
from lmcache.tools.trace_replay.driver import (
    ReplayPace,
    ReplayResult,
    StorageReplayDriver,
)
from lmcache.tools.trace_replay.stats import ReplayStatsCollector
from lmcache.v1.distributed.config import (
    add_storage_manager_args,
    parse_args_to_config,
)

logger = init_logger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Wire the ``trace-replay`` arguments onto *parser*.

    Called from :class:`lmcache.cli.commands.bench.BenchCommand` during
    ``register()``.  Split into a module-level function so tests can
    construct an isolated parser without instantiating the full bench
    command.

    Args:
        parser: The argparse parser for the ``trace-replay`` sub-target.
    """
    parser.add_argument(
        "trace_path",
        metavar="FILE",
        help=(
            "Path to a .lct trace file produced by "
            "'lmcache server --trace-level storage'."
        ),
    )
    parser.add_argument(
        "--pace",
        choices=[p.value for p in ReplayPace],
        default=ReplayPace.ASAP.value,
        help=(
            "Replay pacing.  'asap' (default) dispatches as fast as "
            "possible; 'realtime' honors recorded inter-call intervals."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for CSV/JSON output (default: current).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV export.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also export JSON summary.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the terminal summary (files are still written).",
    )
    add_storage_manager_args(parser)


def run(args: argparse.Namespace) -> None:
    """Execute the bench ``trace-replay`` subcommand.

    Args:
        args: Parsed CLI arguments.
    """
    sm_config = parse_args_to_config(args)
    pace = ReplayPace(args.pace)

    logger.info(
        "trace replay (bench): file=%s pace=%s",
        args.trace_path,
        pace.value,
    )

    with StorageReplayDriver(sm_config, args.trace_path) as driver:
        result = driver.run(pace=pace)

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_csv:
        csv_path = os.path.join(args.output_dir, "trace_replay_ops.csv")
        result.stats.export_csv(csv_path)
        logger.info("CSV written to %s", csv_path)
    if args.json:
        json_path = os.path.join(args.output_dir, "trace_replay_summary.json")
        result.stats.export_json(json_path)
        logger.info("JSON written to %s", json_path)

    if not args.quiet:
        _emit_terminal_summary(result.stats, result)

    if result.records_failed > 0:
        sys.exit(1)


def _emit_terminal_summary(
    stats: ReplayStatsCollector,
    result: ReplayResult,
) -> None:
    """Print the bench-style summary using :class:`Metrics`.

    Args:
        stats: The stats collector populated during replay.
        result: The full :class:`ReplayResult` — used for the
            replayed/skipped/failed totals and digest comparison.
    """
    metrics = Metrics(title="Trace Replay Result")
    metrics.add_handler(StreamHandler(get_formatter("terminal", width=64)))

    overall = metrics.add_section("overall", "Overall")
    overall.add("replayed", "Records replayed", result.records_replayed)
    overall.add("skipped", "Records skipped", result.records_skipped)
    overall.add("failed", "Records failed", result.records_failed)
    overall.add(
        "duration",
        "Replay duration (s)",
        round(stats.total_duration_s(), 3),
    )
    header_digest = result.header_digest
    replay_digest = result.replay_config_digest
    if header_digest and replay_digest and header_digest != replay_digest:
        overall.add(
            "digest",
            "Config digest",
            f"MISMATCH (rec={header_digest[:8]}, run={replay_digest[:8]})",
        )
    elif header_digest:
        overall.add("digest", "Config digest", f"match ({header_digest[:8]})")

    summary = stats.summary()
    if summary:
        ops_section = metrics.add_section("ops", "Per-Op Latency (ms)")
        for qn in sorted(summary):
            s = summary[qn]
            short = qn.split(".")[-1]
            ops_section.add(
                f"{short}_count",
                f"{short} count",
                s.count,
            )
            ops_section.add(
                f"{short}_p50",
                f"{short} p50",
                round(s.p50_ms, 3),
            )
            ops_section.add(
                f"{short}_p90",
                f"{short} p90",
                round(s.p90_ms, 3),
            )
            ops_section.add(
                f"{short}_p99",
                f"{short} p99",
                round(s.p99_ms, 3),
            )

    metrics.emit()
