# SPDX-License-Identifier: Apache-2.0

"""``lmcache trace`` — inspect and replay storage-level trace files.

Subcommands:

* ``info FILE`` — print a summary (header metadata + per-qualname
  record counts).
* ``replay FILE ...`` — reissue every recorded call against a fresh
  StorageManager.  Takes the standard storage-manager CLI flags (see
  :func:`lmcache.v1.distributed.config.add_storage_manager_args`),
  plus pacing (``--pace``), per-record output (``--verbose`` /
  ``--jsonl-out``), aggregated CSV/JSON summary export
  (``--output-dir`` / ``--no-csv`` / ``--json``), and a terminal
  metrics table (suppressible with ``-q``).
* ``record`` — v1 stub.  Prints the equivalent
  ``lmcache server --trace-level storage …`` invocation and exits.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from typing import Callable
import argparse
import json
import os
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.metrics import Metrics, StreamHandler, get_formatter
from lmcache.logging import init_logger
from lmcache.tools.trace_replay.driver import (
    ReplayPace,
    ReplayResult,
    StorageReplayDriver,
)
from lmcache.tools.trace_replay.stats import ReplayStatsCollector
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.mp_observability.trace.reader import TraceReader

logger = init_logger(__name__)


class TraceCommand(BaseCommand):
    """Subcommand group for trace inspection and replay."""

    def name(self) -> str:
        return "trace"

    def help(self) -> str:
        return "Inspect and replay LMCache storage-level trace files."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        # Empty; all args live under the per-subcommand parsers added
        # in :meth:`register`.
        pass

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register ``trace`` with the root parser.

        Overrides :meth:`BaseCommand.register` because ``trace`` has
        its own nested subparsers (``info``, ``replay``, ``record``).
        The base-class ``--format``/``--output``/``--quiet`` flags do
        not apply uniformly across the subcommands — ``replay`` has
        its own ``--jsonl-out`` output channel — so they are added
        only to ``info``.
        """
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description=self.help(),
        )
        inner = parser.add_subparsers(
            dest="trace_target",
            required=True,
            metavar="{info,replay,record}",
        )
        self._register_info(inner)
        self._register_replay(inner)
        self._register_record(inner)

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch the parsed subcommand."""
        handlers: dict[str, Callable[[argparse.Namespace], None]] = {
            "info": self._run_info,
            "replay": self._run_replay,
            "record": self._run_record,
        }
        handler = handlers.get(args.trace_target)
        if handler is None:
            # ``required=True`` on the subparser makes this unreachable
            # in practice; branch is kept for defensive logging.
            print(f"Unknown trace target: {args.trace_target}", file=sys.stderr)
            sys.exit(1)
        handler(args)

    # ------------------------------------------------------------------
    # ``info``
    # ------------------------------------------------------------------

    def _register_info(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "info",
            help="Print a summary of a trace file.",
        )
        parser.add_argument(
            "trace_path",
            metavar="FILE",
            help="Path to a .lct trace file.",
        )
        parser.set_defaults(func=self.execute)

    def _run_info(self, args: argparse.Namespace) -> None:
        """Read a trace file and print a one-screen summary."""
        with TraceReader(args.trace_path) as r:
            header = r.header
            counts: Counter[str] = Counter()
            max_mono = 0.0
            for record in r.records():
                counts[record.qualname] += 1
                if record.t_mono > max_mono:
                    max_mono = record.t_mono

        print(f"Trace file: {args.trace_path}")
        print(f"  level:                {header.level}")
        print(f"  format_version:       {header.format_version}")
        print(f"  trace_schema_version: {header.trace_schema_version}")
        print(f"  duration:             {max_mono:.3f}s")
        print(f"  sm_config_digest:     {header.sm_config_digest or '(none)'}")
        print(f"  total_records:        {sum(counts.values())}")
        if counts:
            print("  ops:")
            for qn in sorted(counts):
                print(f"    {qn}: {counts[qn]}")
        else:
            print("  ops: (none)")

    # ------------------------------------------------------------------
    # ``replay``
    # ------------------------------------------------------------------

    def _register_replay(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "replay",
            help="Replay a trace file against a fresh StorageManager.",
            description=(
                "Replay a trace file against a fresh StorageManager.  "
                "Accepts the standard storage-manager config flags "
                "(--l1-size-gb, --eviction-policy, --l2-…); see "
                "'lmcache server --help' for the full list."
            ),
        )
        parser.add_argument(
            "trace_path",
            metavar="FILE",
            help="Path to a .lct trace file.",
        )
        parser.add_argument(
            "--pace",
            choices=[p.value for p in ReplayPace],
            default=ReplayPace.ASAP.value,
            help="Pacing strategy (default: asap).",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Print one line per replayed record.",
        )
        parser.add_argument(
            "--jsonl-out",
            default=None,
            metavar="PATH",
            help=(
                "Write one JSON object per replayed record to PATH "
                "(qualname, latency_ms, failed).  Useful for post-hoc "
                "analysis."
            ),
        )
        parser.add_argument(
            "--output-dir",
            default=".",
            help=(
                "Directory for aggregated CSV/JSON summary output "
                "(default: current directory)."
            ),
        )
        parser.add_argument(
            "--no-csv",
            action="store_true",
            help="Skip the aggregated CSV summary export.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Also export an aggregated JSON summary.",
        )
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="Suppress the terminal metrics table (files are still written).",
        )
        add_storage_manager_args(parser)
        parser.set_defaults(func=self.execute)

    def _run_replay(self, args: argparse.Namespace) -> None:
        """Construct a StorageManager from *args* and drive replay.

        Produces three kinds of output:

        * Per-record stream: ``--verbose`` (stdout) and ``--jsonl-out``
          (one JSON object per line).  Useful for post-hoc analysis.
        * Aggregated per-qualname summary: CSV (unless ``--no-csv``)
          and JSON (with ``--json``) written under ``--output-dir``.
        * Terminal metrics table (unless ``--quiet``) using the shared
          :class:`~lmcache.cli.metrics.Metrics` renderer.
        """
        sm_config: StorageManagerConfig = parse_args_to_config(args)
        pace = ReplayPace(args.pace)

        jsonl_fh = open(args.jsonl_out, "w") if args.jsonl_out else None
        verbose = args.verbose

        def _on_record(qualname: str, latency_s: float, failed: bool) -> None:
            if verbose:
                status = "FAIL" if failed else "OK  "
                print(f"  {status}  {latency_s * 1000:8.3f}ms  {qualname}")
            if jsonl_fh is not None:
                jsonl_fh.write(
                    json.dumps(
                        {
                            "qualname": qualname,
                            "latency_ms": latency_s * 1000.0,
                            "failed": failed,
                        }
                    )
                    + "\n"
                )

        logger.info(
            "trace replay: file=%s pace=%s",
            args.trace_path,
            pace.value,
        )

        try:
            with StorageReplayDriver(sm_config, args.trace_path) as driver:
                result = driver.run(pace=pace, on_record=_on_record)
        finally:
            if jsonl_fh is not None:
                jsonl_fh.close()

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
            self._emit_replay_metrics(result.stats, result)

        if result.records_failed > 0:
            sys.exit(1)

    @staticmethod
    def _emit_replay_metrics(
        stats: ReplayStatsCollector,
        result: ReplayResult,
    ) -> None:
        """Print the replay summary using the shared :class:`Metrics` renderer.

        Args:
            stats: The stats collector populated during replay.
            result: The full :class:`ReplayResult` — used for the
                replayed/skipped/failed totals and digest comparison.
        """
        metrics = Metrics(title="Trace Replay Result")
        metrics.add_handler(StreamHandler(get_formatter("terminal", width=64)))

        overall = metrics.add_section("overall", "Overall")
        overall.add("level", "Trace level", result.header_level)
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
                ops_section.add(f"{short}_count", f"{short} count", s.count)
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

    # ------------------------------------------------------------------
    # ``record`` (stub)
    # ------------------------------------------------------------------

    def _register_record(self, subparsers: argparse._SubParsersAction) -> None:
        subparsers.add_parser(
            "record",
            help=(
                "(v1 stub) print the equivalent 'lmcache server "
                "--trace-level storage' invocation."
            ),
        ).set_defaults(func=self.execute)

    def _run_record(self, _args: argparse.Namespace) -> None:
        """Print the server-flag equivalent and exit.

        Kept as a visible subcommand so future runtime-capture work
        lands at a discoverable name.  Exits with status 2 to
        distinguish "intentional no-op" from "successful run".
        """
        print(
            "'lmcache trace record' is a v1 stub.  "
            "To capture a trace, restart the server with:",
        )
        print(
            "  lmcache server --trace-level storage "
            "[--trace-output /path/to/trace.lct] ...",
        )
        print(
            "Runtime start/stop of tracing from the CLI is planned for "
            "a follow-up release.",
        )
        sys.exit(2)
