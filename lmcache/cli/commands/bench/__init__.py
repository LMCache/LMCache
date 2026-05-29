# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench`` command — sustained performance benchmarking."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.bench.engine_bench.command import (
    register_engine_parser,
    run_engine_bench,
)
from lmcache.cli.commands.bench.l2_adapter_bench.command import (
    register_l2_parser,
    run_l2_adapter_bench,
)
from lmcache.cli.commands.bench.server_bench.command import (
    register_server_parser,
    run_server_bench,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)


class BenchCommand(BaseCommand):
    """CLI command for sustained performance benchmarking."""

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
            metavar="{engine,server,l2}",
        )
        register_engine_parser(inner, self.execute)
        register_server_parser(inner, self.execute)
        register_l2_parser(inner, self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        handlers = {
            "engine": lambda a: run_engine_bench(self, a),
            "server": lambda a: run_server_bench(self, a),
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
