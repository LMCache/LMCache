# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench engine`` subpackage.

Exposes :class:`EngineBenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class EngineBenchCommand(BaseCommand):
    """Benchmark an inference engine."""

    def name(self) -> str:
        return "engine"

    def help(self) -> str:
        return "Benchmark an inference engine."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.cli.commands.bench.engine_bench.command import (
            add_engine_arguments,
        )

        add_engine_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.cli.commands.bench.engine_bench.command import (
            run_engine_bench,
        )

        run_engine_bench(self, args)
