# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench quality`` subpackage.

Measures whether CacheBlend changes the model's answers: runs a full
recompute reference, optionally vLLM prefix caching, and CacheBlend on
byte-identical prompts, and reports per-question rows and a verdict.

Exposes :class:`QualityBenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class QualityBenchCommand(BaseCommand):
    """Measure CacheBlend answer quality against full recompute."""

    def name(self) -> str:
        return "quality"

    def help(self) -> str:
        return "Measure CacheBlend answer quality against full recompute."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # The command module imports the engine-bench stack, which needs
        # optional dependencies. Keep ``lmcache --help`` working without them.
        # First Party
        from lmcache.cli.commands.bench.quality_bench.command import (
            add_quality_arguments,
        )

        add_quality_arguments(parser)

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
        from lmcache.cli.commands.bench.quality_bench.command import (
            run_quality_bench,
        )

        run_quality_bench(args)
