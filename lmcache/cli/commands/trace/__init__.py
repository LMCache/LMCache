# SPDX-License-Identifier: Apache-2.0
"""``lmcache trace`` command — inspect and replay storage-level trace files.

Subcommands:

* ``info FILE`` — print a summary (header metadata + per-qualname
  record counts).
* ``replay FILE ...`` — reissue every recorded call against a fresh
  StorageManager, honoring the recorded inter-call timings.

Trace *capture* is not a ``trace`` subcommand — recording is bound to
the live process via ``lmcache server --trace-level storage
[--trace-output ...]``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Callable
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.trace.info_command import register_info_parser, run_trace_info
from lmcache.cli.commands.trace.replay_command import (
    register_replay_parser,
    run_trace_replay,
)

# The full LMCache runtime (``lmcache.v1.*``, torch kernels, native
# ops) is required for ``trace info`` and ``trace replay``.  Users who
# installed the thin ``lmcache-cli`` shell lack those modules.  Wrap
# the heavy imports and remember the error so each subcommand handler
# can bail out with an actionable install hint.
_IMPORT_ERROR: ImportError | None = None
try:
    # First Party
    from lmcache.cli.commands.trace.driver import (  # noqa: F401
        ReplayResult,
        StorageReplayDriver,
    )
    from lmcache.cli.commands.trace.stats import ReplayStatsCollector  # noqa: F401
    from lmcache.v1.distributed.config import (  # noqa: F401
        StorageManagerConfig,
        add_storage_manager_args,
        parse_args_to_config,
    )
    from lmcache.v1.mp_observability.config import (  # noqa: F401
        add_observability_args,
        parse_args_to_observability_config,
    )
    from lmcache.v1.mp_observability.trace.reader import TraceReader  # noqa: F401
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
        pass  # args registered in register() via subparsers

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register ``trace`` with the root parser.

        Overrides :meth:`BaseCommand.register` because ``trace`` has
        its own nested subparsers (``info`` and ``replay``).
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
        register_info_parser(inner, self.execute)
        register_replay_parser(inner, self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch to the appropriate trace subcommand handler.

        Args:
            args: Parsed CLI arguments containing ``trace_target``.
        """
        _require_full_install()
        handlers: dict[str, Callable[[argparse.Namespace], None]] = {
            "info": lambda a: run_trace_info(self, a),
            "replay": lambda a: run_trace_replay(self, a),
        }
        handler = handlers.get(args.trace_target)
        if handler is None:
            print(
                f"Unknown trace target: {args.trace_target}",
                file=sys.stderr,
            )
            sys.exit(1)
        handler(args)
