# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench trace-analyze`` subpackage.

Exposes :class:`TraceAnalyzeCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class TraceAnalyzeCommand(BaseCommand):
    """Offline cache hit analysis of an MP server request trace."""

    def name(self) -> str:
        return "trace-analyze"

    def help(self) -> str:
        return (
            "Replay an --enable-token-trace request trace (OTLP JSON) through a "
            "simulated cache and report prefix and CacheBlend hit tokens."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # First Party
        from lmcache.tools.cache_simulator.trace_analyze import (
            add_trace_analyze_arguments,
        )

        add_trace_analyze_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.tools.cache_simulator.trace_analyze import run_trace_analyze

        run_trace_analyze(args, self.create_metrics("Trace Analysis", args, width=96))
