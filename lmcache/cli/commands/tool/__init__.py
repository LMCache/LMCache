# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool`` command — offline analysis utilities."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand


class ToolCommand(BaseCommand):
    """CLI command for offline analysis tools bundled with LMCache."""

    def name(self) -> str:
        """Return the subcommand name."""
        return "tool"

    def help(self) -> str:
        """Return short help text shown by ``lmcache -h``."""
        return "Run offline analysis tools."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        """No top-level arguments; all args are registered in register()."""

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register ``lmcache tool`` with nested sub-subcommands.

        Sub-subcommands
        ---------------
        cache-simulator simulate
            Replay lookup-hash JSONL logs at a fixed cache capacity and print
            a full text report plus a 7-panel statistics PNG.
        cache-simulator sweep
            Sweep across a log-spaced range of cache capacities and save a
            hit-rate vs capacity PNG.

        Args:
            subparsers: The subparsers action from the root parser.
        """
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description="Run offline analysis tools bundled with LMCache.",
        )
        inner = parser.add_subparsers(
            dest="tool_name",
            required=True,
            metavar="{cache-simulator}",
        )
        self._register_cache_simulator(inner)

    # ------------------------------------------------------------------
    # cache-simulator sub-subcommands
    # ------------------------------------------------------------------

    def _register_cache_simulator(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        """Register ``lmcache tool cache-simulator`` with simulate/sweep actions.

        Flag definitions are imported from the simulator modules so there is
        a single source of truth:
        - :func:`~lmcache.tools.cache_simulator.simulator.add_simulate_arguments`
        - :func:`~lmcache.tools.cache_simulator.plot_hit_rate.add_sweep_arguments`

        Args:
            subparsers: The inner subparsers action from the ``tool`` parser.
        """
        # Lazy imports — keeps CLI startup fast (avoids loading matplotlib)
        # First Party
        from lmcache.tools.cache_simulator.plot_hit_rate import add_sweep_arguments
        from lmcache.tools.cache_simulator.simulator import add_simulate_arguments

        cs_parser = subparsers.add_parser(
            "cache-simulator",
            help="Simulate KV-cache token hit rate from lookup-hash JSONL logs.",
            description=(
                "Replay LMCache lookup-hash JSONL logs through an LRU cache "
                "to measure token hit rate."
            ),
        )
        cs_sub = cs_parser.add_subparsers(
            dest="cs_action",
            required=True,
            metavar="{simulate,sweep}",
        )

        sim_parser = cs_sub.add_parser(
            "simulate",
            help=(
                "Replay logs at a fixed cache capacity; print a text report "
                "and save a 7-panel statistics PNG."
            ),
        )
        add_simulate_arguments(sim_parser)
        sim_parser.set_defaults(func=self.execute)

        sweep_parser = cs_sub.add_parser(
            "sweep",
            help=(
                "Sweep across a range of cache capacities and save a "
                "hit-rate vs capacity PNG."
            ),
        )
        add_sweep_arguments(sweep_parser)
        sweep_parser.set_defaults(func=self.execute)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(self, args: argparse.Namespace) -> None:
        """Dispatch to the appropriate tool handler.

        Args:
            args: Parsed CLI arguments.
        """
        dispatch = {
            "cache-simulator": self._run_cache_simulator,
        }
        handler = dispatch.get(args.tool_name)
        if handler is None:
            print(f"Unknown tool: {args.tool_name}", file=sys.stderr)
            sys.exit(1)
        handler(args)

    @staticmethod
    def _run_cache_simulator(args: argparse.Namespace) -> None:
        """Dispatch to the correct cache-simulator action.

        Args:
            args: Parsed CLI arguments (includes ``cs_action``).
        """
        # Lazy imports — keeps CLI startup fast (avoids loading matplotlib)
        # First Party
        from lmcache.tools.cache_simulator.plot_hit_rate import run_sweep
        from lmcache.tools.cache_simulator.simulator import run_simulate

        if args.cs_action == "simulate":
            run_simulate(args)
        elif args.cs_action == "sweep":
            run_sweep(args)
        else:
            print(f"Unknown cache-simulator action: {args.cs_action}", file=sys.stderr)
            sys.exit(1)
