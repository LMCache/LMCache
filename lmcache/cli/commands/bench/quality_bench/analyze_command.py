# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench quality analyze`` subcommand."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class QualityAnalyzeCommand(BaseCommand):
    """Score saved arm answers and reach the quality verdict."""

    def name(self) -> str:
        return "analyze"

    def help(self) -> str:
        return "Score the saved answers in a results directory and print a verdict."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.cli.commands.bench.quality_bench.core.analyze_stage import (
            add_analyze_arguments,
        )

        add_analyze_arguments(parser)

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register without the common output args (--format/--output).

        Results are written into the results directory, so the generic
        output flags do not apply.
        """
        parser = subparsers.add_parser(self.name(), help=self.help())
        self.add_arguments(parser)
        parser.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.cli.commands.bench.quality_bench.core.analyze_stage import (
            run_quality_analyze,
        )

        run_quality_analyze(args)
