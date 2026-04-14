# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool`` command — offline analysis utilities."""

# Standard
from pathlib import Path
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand

_GIB = 2**30


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

        Args:
            subparsers: The inner subparsers action from the ``tool`` parser.
        """
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
        self._register_cs_simulate(cs_sub)
        self._register_cs_sweep(cs_sub)

    @staticmethod
    def _add_common_args(parser: argparse.ArgumentParser) -> None:
        """Add arguments shared by both simulate and sweep.

        Args:
            parser: The ArgumentParser to add the flags to.
        """
        parser.add_argument(
            "-i",
            "--input",
            nargs="+",
            required=True,
            metavar="PATH",
            help="Lookup-hash JSONL files or directories to load",
        )
        parser.add_argument(
            "-n",
            "--max-samples",
            type=int,
            default=None,
            metavar="N",
            help="Maximum number of events to process (default: all)",
        )
        parser.add_argument(
            "--model",
            default=None,
            metavar="NAME",
            help="Filter events by model_name (exact match)",
        )
        parser.add_argument(
            "--kv-bytes-per-chunk",
            type=int,
            default=None,
            metavar="BYTES",
            help=(
                "Bytes consumed by one cached chunk. "
                "Auto-computed from the first event's shapes/dtypes if omitted."
            ),
        )

    def _register_cs_simulate(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        """Register the ``simulate`` sub-subcommand.

        Runs the simulator at a single fixed cache capacity, prints a text
        report, and saves a 7-panel statistics PNG.

        Args:
            subparsers: The subparsers action from the ``cache-simulator`` parser.
        """
        parser = subparsers.add_parser(
            "simulate",
            help=(
                "Replay logs at a fixed cache capacity; print a text report "
                "and save a 7-panel statistics PNG."
            ),
        )
        self._add_common_args(parser)
        parser.add_argument(
            "--cache-capacity-gib",
            type=float,
            required=True,
            metavar="GiB",
            help="Cache capacity in gibibytes",
        )
        parser.add_argument(
            "-o",
            "--output",
            default="cache_stats.png",
            metavar="FILE",
            help="Output image path (default: cache_stats.png)",
        )
        parser.set_defaults(func=self.execute)

    def _register_cs_sweep(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        """Register the ``sweep`` sub-subcommand.

        Sweeps across a log-spaced range of cache capacities and saves a
        hit-rate vs capacity PNG.

        Args:
            subparsers: The subparsers action from the ``cache-simulator`` parser.
        """
        parser = subparsers.add_parser(
            "sweep",
            help=(
                "Sweep across a range of cache capacities and save a "
                "hit-rate vs capacity PNG."
            ),
        )
        self._add_common_args(parser)
        parser.add_argument(
            "--min-capacity-gib",
            type=float,
            default=0.5,
            metavar="GiB",
            help="Minimum cache capacity to sweep (default: 0.5 GiB)",
        )
        parser.add_argument(
            "--max-capacity-gib",
            type=float,
            default=500.0,
            metavar="GiB",
            help="Maximum cache capacity to sweep (default: 500 GiB)",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=30,
            metavar="N",
            help="Number of log-spaced capacity samples (default: 30)",
        )
        parser.add_argument(
            "--linear",
            action="store_true",
            help="Use a linear x-axis instead of log scale",
        )
        parser.add_argument(
            "-o",
            "--output",
            default="hit_rate_vs_capacity.png",
            metavar="FILE",
            help="Output image path (default: hit_rate_vs_capacity.png)",
        )
        parser.set_defaults(func=self.execute)

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

    def _run_cache_simulator(self, args: argparse.Namespace) -> None:
        """Run the requested cache-simulator action.

        Args:
            args: Parsed CLI arguments (includes ``cs_action``).
        """
        # Lazy import to avoid pulling in matplotlib at CLI startup
        # First Party
        from lmcache.tools.cache_simulator.simulator import (
            compute_kv_bytes_per_chunk,
            load_lookup_events,
        )

        paths = [Path(p) for p in args.input]
        print(f"Loading lookup events from {[str(p) for p in paths]} …")
        events = load_lookup_events(
            paths, model=args.model, max_samples=args.max_samples
        )
        print(f"Loaded {len(events):,} event(s)")

        if not events:
            print("No events to process.")
            sys.exit(0)

        kv_bpc = args.kv_bytes_per_chunk
        if kv_bpc is None:
            kv_bpc = compute_kv_bytes_per_chunk(events[0])
            if kv_bpc == 0:
                print(
                    "Error: could not determine kv_bytes_per_chunk from the first "
                    "event (shapes/dtypes are empty). "
                    "Pass --kv-bytes-per-chunk explicitly.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"Auto-detected kv_bytes_per_chunk = {kv_bpc:,} bytes")

        if args.cs_action == "simulate":
            self._cs_simulate(args, events, kv_bpc)
        elif args.cs_action == "sweep":
            self._cs_sweep(args, events, kv_bpc)
        else:
            print(f"Unknown cache-simulator action: {args.cs_action}", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def _cs_simulate(
        args: argparse.Namespace,
        events: list,
        kv_bpc: int,
    ) -> None:
        """Execute the simulate action: single-capacity report + chart.

        Args:
            args: Parsed CLI arguments.
            events: Loaded lookup events.
            kv_bpc: KV bytes per chunk.
        """
        # First Party
        from lmcache.tools.cache_simulator.simulator import (
            plot_statistics,
            print_statistics,
            simulate,
        )

        capacity_bytes = int(args.cache_capacity_gib * _GIB)
        chunk_sz = events[0].get("chunk_size", "?")
        print("\nSimulation parameters:")
        print(
            f"  Cache capacity     : {args.cache_capacity_gib:.2f} GiB "
            f"({capacity_bytes:,} bytes)"
        )
        print(f"  KV bytes/chunk     : {kv_bpc:,}")
        print(f"  Chunk size         : {chunk_sz} tokens")
        if args.model:
            print(f"  Model filter       : {args.model}")
        print()

        results = simulate(events, capacity_bytes, kv_bpc)
        print_statistics(results)
        plot_statistics(results, events, args.output)

    @staticmethod
    def _cs_sweep(
        args: argparse.Namespace,
        events: list,
        kv_bpc: int,
    ) -> None:
        """Execute the sweep action: capacity range scan + hit-rate plot.

        Args:
            args: Parsed CLI arguments.
            events: Loaded lookup events.
            kv_bpc: KV bytes per chunk.
        """
        # Standard
        import math

        # Third Party
        import matplotlib.pyplot as plt

        # First Party
        from lmcache.tools.cache_simulator.simulator import simulate

        chunk_size = events[0].get("chunk_size", "?")
        model_label = args.model or "all models"

        # Build log-spaced capacity list
        log_min = math.log10(args.min_capacity_gib * _GIB)
        log_max = math.log10(args.max_capacity_gib * _GIB)
        step = (log_max - log_min) / max(args.points - 1, 1)
        capacities_bytes = sorted(
            {round(10 ** (log_min + i * step)) for i in range(args.points)}
        )

        scale_label = "linear" if args.linear else "log"
        print(
            f"Sweeping {len(capacities_bytes)} capacity values "
            f"({args.min_capacity_gib:.2f} – {args.max_capacity_gib:.2f} GiB), "
            f"chunk_size = {chunk_size} tokens, model = {model_label}\n"
        )
        print(f"{'Capacity (GiB)':>18}  {'Hit rate':>10}")
        print("-" * 32)

        hit_rates: list[float] = []
        for cap_bytes in capacities_bytes:
            cap_gib = cap_bytes / _GIB
            res = simulate(events, cap_bytes, kv_bpc, fast=True)
            rate = res["token_hit_rate"]
            hit_rates.append(rate)
            print(f"{cap_gib:>18.3f}  {rate:>9.2%}")

        x_values = [c / _GIB for c in capacities_bytes]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            x_values,
            [r * 100 for r in hit_rates],
            marker="o",
            linewidth=2,
            markersize=4,
        )
        if not args.linear:
            ax.set_xscale("log")
        ax.set_xlabel("Cache capacity (GiB)", fontsize=12)
        ax.set_ylabel("Token hit rate (%)", fontsize=12)
        ax.set_title(
            f"Token cache hit rate vs capacity\n"
            f"(chunk_size = {chunk_size} tokens, {len(events):,} requests, "
            f"model = {model_label}, {scale_label} scale)",
            fontsize=11,
        )
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        fig.tight_layout()
        fig.savefig(args.output, dpi=150)
        print(f"\nPlot saved to '{args.output}'")
