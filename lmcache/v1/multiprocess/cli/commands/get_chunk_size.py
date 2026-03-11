# SPDX-License-Identifier: Apache-2.0

"""
``get-chunk-size`` sub-command — query the server chunk size.

Sends a :attr:`RequestType.GET_CHUNK_SIZE` request over ZMQ
and prints the configured chunk size.
"""

# Standard
import argparse

# First Party
from lmcache.v1.multiprocess.cli.base import (
    add_connection_args,
    send_request,
)
from lmcache.v1.multiprocess.protocols.base import RequestType


def _run(args: argparse.Namespace) -> None:
    """Entry-point called by the CLI dispatcher."""
    chunk_size = send_request(args, RequestType.GET_CHUNK_SIZE)
    print(chunk_size)


def register_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``get-chunk-size`` sub-command."""
    parser = subparsers.add_parser(
        "get-chunk-size",
        help="Query the server's chunk size",
        description=(
            "Send a GET_CHUNK_SIZE request and print the configured chunk size."
        ),
    )
    add_connection_args(parser)
    parser.set_defaults(func=_run)
