# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench quality run`` subcommand."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class QualityRunCommand(BaseCommand):
    """Send the quality plan to one or more arms and save their answers."""

    def name(self) -> str:
        return "run"

    def help(self) -> str:
        return "Send the quality prompts to one or more arms and save the answers."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # The runtime imports the engine-bench stack, which needs optional
        # dependencies; keep `lmcache bench quality -h` working without them.
        # First Party
        from lmcache.cli.commands.bench.quality_bench.core.run_stage import (
            add_run_arguments,
        )

        add_run_arguments(parser)

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register without the common output args (--format/--output).

        Results go to ``--output-dir`` as files, so the generic output
        flags do not apply.
        """
        parser = subparsers.add_parser(self.name(), help=self.help())
        self.add_arguments(parser)
        parser.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.cli.commands.bench.quality_bench.core.run_stage import (
            run_quality_run,
        )

        run_quality_run(args)
