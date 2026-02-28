# SPDX-License-Identifier: Apache-2.0
"""``lmcache server`` subcommand — starts the ZMQ multiprocess cache server."""

# Standard
import argparse

# First Party
from lmcache.v1.distributed.config import parse_args_to_config
from lmcache.v1.multiprocess.server import add_server_args, run_cache_server


def _handle_server(args: argparse.Namespace) -> None:
    """Start the ZMQ cache server from parsed CLI arguments.

    Converts the flat argparse namespace into a
    :class:`StorageManagerConfig` and delegates to
    :func:`run_cache_server`.

    Args:
        args: Parsed CLI arguments containing server and storage-manager
            options.
    """
    storage_manager_config = parse_args_to_config(args)
    run_cache_server(
        storage_manager_config=storage_manager_config,
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        hash_algorithm=args.hash_algorithm,
    )


def register_server_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``server`` subcommand on *subparsers*.

    Adds a ``server`` sub-parser with all server and storage-manager
    arguments and wires it to :func:`_handle_server`.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers action
            returned by :meth:`argparse.ArgumentParser.add_subparsers`.
    """
    server_parser = subparsers.add_parser(
        "server",
        help="Start the LMCache ZMQ cache server",
    )
    add_server_args(server_parser)
    server_parser.set_defaults(func=_handle_server)
