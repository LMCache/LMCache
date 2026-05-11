# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool cache-simulator`` sub-subcommand wiring.

Registers the ``simulate``, ``sweep``, and ``gen-dataset`` actions under
``lmcache tool cache-simulator``.  Flag definitions and execution logic
live entirely in the simulator modules:

* :func:`~lmcache.tools.cache_simulator.simulator.add_simulate_arguments`
* :func:`~lmcache.tools.cache_simulator.simulator.run_simulate`
* :func:`~lmcache.tools.cache_simulator.plot_hit_rate.add_sweep_arguments`
* :func:`~lmcache.tools.cache_simulator.plot_hit_rate.run_sweep`
* :func:`~lmcache.tools.cache_simulator.gen_bench_dataset.add_gen_dataset_arguments`
* :func:`~lmcache.tools.cache_simulator.gen_bench_dataset.run_gen_dataset`

To add a new action, add ``add_*_arguments`` / ``run_*`` functions to the
appropriate simulator module and wire them up in :func:`register`.
"""

# Standard
import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``cache-simulator`` and its ``simulate``/``sweep`` actions.

    Imports are lazy so that matplotlib is not loaded at CLI startup.

    Args:
        subparsers: The subparsers action from the ``lmcache tool`` parser.
    """
    # Lazy imports — keeps CLI startup fast and makes cache-simulator optional.
    # Missing sortedcontainers/matplotlib (plot extra) skips the subcommand.
    try:
        # First Party
        from lmcache.tools.cache_simulator.gen_bench_dataset import (
            add_gen_dataset_arguments,
        )
        from lmcache.tools.cache_simulator.plot_hit_rate import add_sweep_arguments
        from lmcache.tools.cache_simulator.simulator import add_simulate_arguments
    except ImportError:
        return

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
        metavar="{simulate,sweep,gen-dataset}",
    )

    sim_parser = cs_sub.add_parser(
        "simulate",
        help=(
            "Replay logs at a fixed cache capacity; print a text report "
            "and save a 7-panel statistics PNG."
        ),
    )
    add_simulate_arguments(sim_parser)
    sim_parser.set_defaults(func=execute)

    sweep_parser = cs_sub.add_parser(
        "sweep",
        help=(
            "Sweep across a range of cache capacities and save a "
            "hit-rate vs capacity PNG."
        ),
    )
    add_sweep_arguments(sweep_parser)
    sweep_parser.set_defaults(func=execute)

    gen_parser = cs_sub.add_parser(
        "gen-dataset",
        help=(
            "Generate a vllm bench serve custom dataset (JSONL) from "
            "lookup-hash JSONL logs, preserving prefix-sharing structure."
        ),
    )
    add_gen_dataset_arguments(gen_parser)
    gen_parser.set_defaults(func=execute)


def execute(args: argparse.Namespace) -> None:
    """Dispatch to the correct cache-simulator action.

    Args:
        args: Parsed CLI arguments (includes ``cs_action``).
    """
    # Standard
    import sys

    # First Party
    # Lazy imports — keeps CLI startup fast (avoids loading matplotlib)
    from lmcache.tools.cache_simulator.plot_hit_rate import run_sweep
    from lmcache.tools.cache_simulator.simulator import run_simulate

    if args.cs_action == "simulate":
        run_simulate(args)
    elif args.cs_action == "sweep":
        run_sweep(args)
    elif args.cs_action == "gen-dataset":
        # First Party
        from lmcache.tools.cache_simulator.gen_bench_dataset import run_gen_dataset

        run_gen_dataset(args)
    else:
        print(f"Unknown cache-simulator action: {args.cs_action}", file=sys.stderr)
        sys.exit(1)
