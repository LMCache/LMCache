# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool flamegraph`` — profile any already-running process.

Attaches a recorder to a live process -- an MP cache server, a vLLM worker,
or any Python process -- for a set duration (or until interrupted) and
renders a flame graph. Unlike the ``--flamegraph`` option on ``lmcache
bench l2`` / ``lmcache bench server``, it drives no synthetic load: it
profiles the target under whatever real work it is already doing.

The recorder, modes, and rendering all live in the shared
:mod:`lmcache.cli.profiling`; this module only maps CLI arguments onto
it. :class:`FlamegraphCommand` is auto-discovered by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand

# Seconds to record when ``--duration`` is not given.
_DEFAULT_DURATION_SEC = 30.0


class FlamegraphCommand(BaseCommand):
    """CLI command that flame-graphs an already-running process."""

    def name(self) -> str:
        return "flamegraph"

    def help(self) -> str:
        return "Record a flame graph of a running LMCache process."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add flame-graph arguments to *parser*.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        parser.add_argument(
            "--pid",
            type=int,
            required=True,
            metavar="PID",
            help="Process to profile, e.g. the pid of 'lmcache server'.",
        )
        parser.add_argument(
            "--mode",
            choices=["on-cpu", "off-cpu", "offwake", "wakeup", "wall", "gil"],
            default="gil",
            help=(
                "What to sample (default: gil). 'gil' / 'wall' use py-spy "
                "(Python/GIL, CPython only, no target change); 'on-cpu' / "
                "'off-cpu' / 'offwake' / 'wakeup' use perf/bcc (CPU/IO/kernel, "
                "any process, Python names need PYTHONPERFSUPPORT=1). See the "
                "'lmcache tool flamegraph' docs for what each shows."
            ),
        )
        parser.add_argument(
            "--duration",
            type=float,
            default=_DEFAULT_DURATION_SEC,
            metavar="SECONDS",
            help=(
                f"Seconds to record (default: {_DEFAULT_DURATION_SEC:g}). "
                "Pass 0 to record until interrupted with Ctrl-C."
            ),
        )
        parser.add_argument(
            "--output",
            default="",
            metavar="PATH",
            help=(
                "SVG output path. Default: "
                "/tmp/lmcache_bench_flames/pid<PID>.<mode>.svg."
            ),
        )
        parser.add_argument(
            "--flamegraph-scripts-dir",
            default="",
            metavar="DIR",
            help=(
                "Directory with the FlameGraph scripts (perf/bcc modes only). "
                "If omitted, resolved from $FLAMEGRAPH_DIR / ~/FlameGraph or "
                "auto-cloned to a temp directory."
            ),
        )

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register without the common ``--format``/``--output`` flags.

        This command owns ``--output`` (the SVG path) and emits no
        metrics, so the generic output flags would collide and confuse.

        Args:
            subparsers: The subparsers action from the parent parser.
        """
        parser = subparsers.add_parser(self.name(), help=self.help())
        self.add_arguments(parser)
        parser.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        """Record the flame graph, or exit non-zero with the reason.

        Args:
            args: Parsed CLI arguments.
        """
        # Deferred so ``lmcache -h`` does not pay for the import.
        # First Party
        from lmcache.cli.profiling import (
            PY_SPY_MODES,
            ProfileError,
            check_profiling_deps,
            default_output_path,
            record_attached,
            resolve_flamegraph_dir,
        )

        def log(message: str) -> None:
            print(message, flush=True)

        output = args.output or default_output_path(f"pid{args.pid}", args.mode)
        try:
            check_profiling_deps(args.mode)
            # py-spy renders its own SVG; only the perf/bcc modes need the
            # FlameGraph scripts, so do not clone them otherwise.
            flamegraph_dir = ""
            if args.mode not in PY_SPY_MODES:
                flamegraph_dir = resolve_flamegraph_dir(
                    args.flamegraph_scripts_dir, log
                )
            record_attached(
                pid=args.pid,
                mode=args.mode,
                output=output,
                flamegraph_dir=flamegraph_dir,
                duration=args.duration,
                log=log,
            )
        except ProfileError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)
