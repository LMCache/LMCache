# SPDX-License-Identifier: Apache-2.0
"""``lmcache server`` subcommand — starts the ZMQ multiprocess cache server."""

# Standard
import argparse

# First Party
from lmcache.v1.multiprocess.server import (
    add_server_args,
    run_server_from_args,
)


def register_server_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``server`` subcommand on *subparsers*.

    Adds a ``server`` sub-parser with all server and storage-manager
    arguments and wires it to :func:`run_server_from_args`.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers action
            returned by :meth:`argparse.ArgumentParser.add_subparsers`.
    """
    server_parser = subparsers.add_parser(
        "server",
        help="Start the LMCache ZMQ cache server",
    )
    add_server_args(server_parser)
    server_parser.set_defaults(func=run_server_from_args)
