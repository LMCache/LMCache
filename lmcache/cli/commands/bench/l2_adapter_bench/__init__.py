# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench l2`` subpackage.

Exposes :class:`L2AdapterBenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class L2AdapterBenchCommand(BaseCommand):
    """Benchmark an L2 adapter (store / lookup / load)."""

    def name(self) -> str:
        return "l2"

    def help(self) -> str:
        return "Benchmark an L2 adapter (store / lookup / load)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.cli.commands.bench.l2_adapter_bench.command import (
            add_l2_arguments,
        )

        add_l2_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.cli.commands.bench.l2_adapter_bench.command import (
            run_l2_adapter_bench,
        )

        run_l2_adapter_bench(self, args)
