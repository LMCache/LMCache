# SPDX-License-Identifier: Apache-2.0

"""``lmcache trace`` — inspect and replay storage-level trace files.

Subcommands:

* ``info FILE`` — print a summary (header metadata + per-qualname
  record counts).
* ``replay FILE ...`` — reissue every recorded call against a fresh
  StorageManager, honoring the recorded inter-call timings.  Takes
  the standard storage-manager CLI flags (see
  :func:`lmcache.v1.distributed.config.add_storage_manager_args`),
  plus per-record output (``--verbose`` / ``--jsonl-out``),
  aggregated CSV/JSON summary export (``--output-dir`` / ``--no-csv``
  / ``--json``), and a terminal metrics table (suppressible with
  ``-q``).

Trace *capture* is not a ``trace`` subcommand — recording is bound to
the live process via ``lmcache server --trace-level storage
[--trace-output ...]``.  Surfacing a CLI stub here would only
duplicate that flag while leaving the user wondering why it cannot
start a recorder against an already-running server.
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

logger = init_logger(__name__)

# ``lmcache trace`` drives a real StorageManager and decodes a binary
# trace file — both pulled from the full LMCache runtime
# (``lmcache.v1.*``, torch kernels, native ops).  Users who installed
# the thin ``lmcache-cli`` shell lack those modules, so importing them
# unconditionally at the top of this file would kill the *entire*
# ``lmcache`` CLI with an opaque ImportError the first time
# ``lmcache/cli/commands/__init__.py`` loads the command registry.
#
# Wrap the heavy imports and remember the error so each subcommand
# handler can bail out with an actionable install hint.  ``record`` is
# a stub that needs no runtime, so it keeps working on a CLI-only
# install.
_IMPORT_ERROR: ImportError | None = None
try:
    # First Party
    from lmcache.cli.commands.trace.driver import (
        ReplayResult,
        StorageReplayDriver,
    )
    from lmcache.cli.commands.trace.stats import ReplayStatsCollector
    from lmcache.v1.distributed.config import (
        StorageManagerConfig,
        add_storage_manager_args,
        parse_args_to_config,
    )
    from lmcache.v1.mp_observability.config import (
        add_observability_args,
        parse_args_to_observability_config,
    )
    from lmcache.v1.mp_observability.trace.reader import TraceReader
except ImportError as _exc:
    _IMPORT_ERROR = _exc


def _require_full_install() -> None:
    """Exit with an install hint if the full LMCache runtime is missing.

    ``lmcache trace info`` and ``lmcache trace replay`` both need
    ``lmcache.v1.*`` (StorageManager, trace codecs, TraceReader).
    When those imports failed at module load — almost always because
    the user installed ``lmcache-cli`` instead of the full package —
    this helper prints the shortest actionable message to stderr and
    exits with status ``2`` so scripts can detect the install gap
    programmatically.

    Writes directly to :data:`sys.stderr` rather than going through
    :mod:`logging` so the message reaches the user even when the
    lmcache logger has been suppressed or its handlers redirected.

    No-op when imports succeeded, so it is safe to call
    unconditionally at the top of every trace handler.
    """
    if _IMPORT_ERROR is None:
        return
    print(
        "ERROR: `lmcache trace` needs the full LMCache package "
        "(StorageManager, trace codecs, etc.), but only the `lmcache-cli` "
        "shell appears to be installed.\n"
        "  Install the full package with `pip install lmcache` and try "
        "again.\n"
        f"  Original import error: {_IMPORT_ERROR}",
        file=sys.stderr,
    )
    sys.exit(2)


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
        its own nested subparsers (``info`` and ``replay``).  The
        base-class ``--format``/``--output``/``--quiet`` flags do not
        apply uniformly across the subcommands — ``replay`` has its
        own ``--jsonl-out`` output channel — so they are added only
        to ``info``.
        """
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description=self.help(),
        )
        inner = parser.add_subparsers(
            dest="trace_target",
            required=True,
            metavar="{info,replay}",
        )
        self._register_info(inner)
        self._register_replay(inner)

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch the parsed subcommand."""
        handlers: dict[str, Callable[[argparse.Namespace], None]] = {
            "info": self._run_info,
            "replay": self._run_replay,
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
        _require_full_install()
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
        # ``add_storage_manager_args`` and ``add_observability_args``
        # live in the full LMCache runtime (``lmcache.v1.*``) and are
        # unavailable in the CLI-only install.  When those imports
        # failed at module load, register ``replay`` with only the
        # CLI-local flags above so ``--help`` still works; the actual
        # execute path bails via :func:`_require_full_install` before
        # it would try to parse the missing-flag namespace.
        #
        # When the full runtime *is* present we share the whole
        # observability surface with ``lmcache server``.
        # ``--trace-level`` / ``--trace-output`` configure *recording*
        # and have no effect during replay; :meth:`_run_replay`
        # overrides them to ``None`` before constructing the
        # observability config rather than duplicating the argparse
        # registration to strip them.
        if _IMPORT_ERROR is None:
            add_storage_manager_args(parser)
            add_observability_args(parser)
        parser.set_defaults(func=self.execute)

    def _run_replay(self, args: argparse.Namespace) -> None:
        """Construct a StorageManager from *args* and drive replay.

        Produces three kinds of output:

        * Per-record stream: every dispatch is logged at INFO with its
          progress (``[N/total]``), qualname, and latency.
          ``--verbose`` additionally mirrors each record to stdout,
          and ``--jsonl-out PATH`` writes one JSON object per record
          to ``PATH`` for post-hoc analysis.
        * Aggregated per-qualname summary: CSV (unless ``--no-csv``)
          and JSON (with ``--json``) written under ``--output-dir``.
        * Terminal metrics table (unless ``--quiet``) using the shared
          :class:`~lmcache.cli.metrics.Metrics` renderer.
        """
        _require_full_install()
        sm_config: StorageManagerConfig = parse_args_to_config(args)

        # ``--trace-level`` / ``--trace-output`` belong to the recording
        # surface.  They are still registered on the parser (see
        # :meth:`_register_replay`) so the flag set stays in lock-step
        # with ``lmcache server``, but they have no meaning here — any
        # value a caller passes is silently clobbered to ``None`` so
        # the replay-side ``ObservabilityConfig`` never tries to start
        # a recorder.
        args.trace_level = None
        args.trace_output = None
        obs_config = parse_args_to_observability_config(args)

        # Create output directories *before* replay starts.  A replay
        # can run for minutes; surfacing a bad ``--output-dir`` or
        # unwritable ``--jsonl-out`` parent now avoids silently losing
        # the summary/stream after the work has already happened.
        os.makedirs(args.output_dir, exist_ok=True)
        if args.jsonl_out:
            jsonl_parent = os.path.dirname(os.path.abspath(args.jsonl_out))
            if jsonl_parent:
                os.makedirs(jsonl_parent, exist_ok=True)

        # ANSI: bold + yellow for the banner text, reset at the end.
        # The lmcache log formatter only colors the WARNING prefix;
        # these codes highlight the message body too.  Writing them
        # into a file via shell redirection leaves the escape bytes
        # visible but still readable.
        bold = "\033[1;33m"
        reset = "\033[0m"
        bar = "=" * 78
        logger.warning(
            "\n%s%s\n"
            "  !! REPLAY ENVIRONMENT MISMATCH MAY CAUSE RETRIEVE MISSES !!\n"
            "%s%s\n"
            "  * Replay uses the *replay-side* StorageManager config, which\n"
            "    may differ from the config recorded in the trace.\n"
            "  * Replay runs on a host whose performance may differ from\n"
            "    the recording host.\n"
            "  * StorageManager reads/writes are async — an L2 load that\n"
            "    had finished at record time may not have finished yet at\n"
            "    replay time, so the matching retrieve can miss.\n"
            "\n"
            "  Treat retrieve-miss counts as a signal about the replay\n"
            "  environment, not as a defect in the trace.\n"
            "%s%s",
            bold,
            bar,
            bar,
            reset,
            bar,
            reset,
        )

        # Pre-scan to count total records so progress logs can show
        # [N/total].  The reader streams frames, so counting is cheap
        # relative to replay (which actually dispatches StorageManager
        # calls).
        with TraceReader(args.trace_path) as r:
            total_records = sum(1 for _ in r.records())
        logger.info(
            "trace replay: file=%s records=%d",
            args.trace_path,
            total_records,
        )

        jsonl_fh = open(args.jsonl_out, "w") if args.jsonl_out else None
        verbose = args.verbose
        counter = {"n": 0}

        def _on_record(qualname: str, latency_s: float, failed: bool) -> None:
            counter["n"] += 1
            status = "FAIL" if failed else "OK"
            logger.info(
                "[%d/%d] %s %s (%.3fms)",
                counter["n"],
                total_records,
                status,
                qualname,
                latency_s * 1000.0,
            )
            if verbose:
                print(
                    f"  [{counter['n']}/{total_records}]  "
                    f"{status:<4}  {latency_s * 1000:8.3f}ms  {qualname}"
                )
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

        try:
            with StorageReplayDriver(
                sm_config, args.trace_path, obs_config=obs_config
            ) as driver:
                result = driver.run(on_record=_on_record)
        finally:
            if jsonl_fh is not None:
                jsonl_fh.close()

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
                short = _short_op_name(qn)
                ops_section.add(f"{short}_count", f"{short} count", s.count)
                ops_section.add(
                    f"{short}_mean",
                    f"{short} mean",
                    round(s.mean_ms, 3),
                )
                ops_section.add(
                    f"{short}_p50",
                    f"{short} p50",
                    round(s.p50_ms, 3),
                )
                ops_section.add(
                    f"{short}_p99",
                    f"{short} p99",
                    round(s.p99_ms, 3),
                )

        metrics.emit()


def _short_op_name(qualname: str) -> str:
    """Return a compact, human-readable label for a traced qualname.

    Plain methods collapse to the method name: the table has limited
    column width and the fully-qualified path is verbose.

    Context-manager handlers (``__enter__`` / ``__exit__``) instead
    collapse to ``<owning_method>.enter`` / ``<owning_method>.exit``,
    so the reader can tell *which* context manager the pair belongs
    to — the bare ``__enter__`` / ``__exit__`` label is useless when
    multiple context-manager-returning methods are traced.

    Args:
        qualname: Dotted qualname recorded by the tracer, e.g.
            ``lmcache.v1.distributed.storage_manager.StorageManager.read_prefetched_results.__enter__``.

    Returns:
        A short label suitable as a metrics row prefix.
    """
    parts = qualname.split(".")
    last = parts[-1]
    if last in ("__enter__", "__exit__") and len(parts) >= 2:
        return f"{parts[-2]}.{last.strip('_')}"
    return last
