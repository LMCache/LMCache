# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench`` command — sustained performance benchmarking."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.bench import test_cache as _test_cache_mod
from lmcache.cli.commands.bench.engine_bench.command import (
    register_engine_parser,
    run_engine_bench,
)
from lmcache.cli.commands.bench.l2_adapter_bench.command import (
    register_l2_parser,
    run_l2_adapter_bench,
)
from lmcache.cli.commands.bench.test_cache import TestCacheCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


class BenchCommand(BaseCommand):
    """CLI command for sustained performance benchmarking."""

    def __init__(self) -> None:
        super().__init__()
        # None on slim install; _register_kvcache registers a stub instead.
        self._kvcache_delegate = (
            TestCacheCommand() if _test_cache_mod._IMPORT_ERROR is None else None
        )

    def name(self) -> str:
        return "bench"

    def help(self) -> str:
        return "Run sustained performance benchmarks."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        pass  # args registered in register() via subparsers

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description="Run sustained performance benchmarks.",
        )
        inner = parser.add_subparsers(
            dest="bench_target",
            required=True,
            metavar="{engine,kvcache,l2}",
        )
        # TODO(chunxiaozheng): move kvcache to its own sub module too
        register_engine_parser(inner, self.execute)
        self._register_kvcache(inner)
        register_l2_parser(inner, self.execute)

    # ------------------------------------------------------------------
    # kvcache bench target — end-to-end MP cache sanity test
    # ------------------------------------------------------------------

    def _register_kvcache(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        """Register ``lmcache bench kvcache``. Delegates to
        :class:`TestCacheCommand`, or registers a stub on slim install.
        """
        if _test_cache_mod._IMPORT_ERROR is not None:
            subparsers.add_parser(
                "kvcache",
                help="(requires full lmcache install)",
                description=(
                    "End-to-end sanity test for the LMCache MP cache server. "
                    "Requires the full `lmcache` package; not available in "
                    "the `lmcache-cli` install."
                ),
            ).set_defaults(func=self.execute)
            return
        assert self._kvcache_delegate is not None
        parser = subparsers.add_parser(
            "kvcache",
            help=self._kvcache_delegate.help(),
            description=(
                "End-to-end sanity test for the LMCache MP cache server: "
                "runs LOOKUP / STORE / RETRIEVE against a live MP server "
                "and verifies KV cache checksums."
            ),
        )
        assert self._kvcache_delegate is not None
        self._kvcache_delegate.add_arguments(parser)
        parser.set_defaults(func=self.execute)

    def _bench_kvcache(self, args: argparse.Namespace) -> None:
        """Dispatch ``lmcache bench kvcache`` to ``TestCacheCommand``."""
        if _test_cache_mod._IMPORT_ERROR is not None:
            print(
                "ERROR: `lmcache bench kvcache` needs the full LMCache "
                "package (torch, zmq, MP runtime), but only the "
                "`lmcache-cli` shell appears to be installed.\n"
                "  Install the full package with `pip install lmcache` "
                "and try again.\n"
                f"  Original import error: {_test_cache_mod._IMPORT_ERROR}",
                file=sys.stderr,
            )
            sys.exit(1)
        assert self._kvcache_delegate is not None
        self._kvcache_delegate.execute(args)

    def execute(self, args: argparse.Namespace) -> None:
        handlers = {
            "engine": lambda a: run_engine_bench(self, a),
            "kvcache": self._bench_kvcache,
            "l2": lambda a: run_l2_adapter_bench(self, a),
        }
        handler = handlers.get(args.bench_target)
        if handler is None:
            print(
                f"Unknown bench target: {args.bench_target}",
                file=sys.stderr,
            )
            sys.exit(1)
        handler(args)
