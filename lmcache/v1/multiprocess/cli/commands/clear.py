# SPDX-License-Identifier: Apache-2.0

"""
``clear`` sub-command — clear all cached KV data.

Sends a :attr:`RequestType.CLEAR` request over ZMQ.
"""

# Standard
import argparse

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.cli.base import (
    add_connection_args,
    send_request,
)
from lmcache.v1.multiprocess.protocols.base import RequestType

logger = init_logger(__name__)


def _run(args: argparse.Namespace) -> None:
    """Entry-point called by the CLI dispatcher."""
    send_request(args, RequestType.CLEAR)
    logger.info("Cache cleared successfully.")


def register_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``clear`` sub-command."""
    parser = subparsers.add_parser(
        "clear",
        help="Clear all cached KV data",
        description=(
            "Send a CLEAR request to remove all stored KV cache data from the server."
        ),
    )
    add_connection_args(parser)
    parser.set_defaults(func=_run)
