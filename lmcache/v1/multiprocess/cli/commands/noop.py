# SPDX-License-Identifier: Apache-2.0

"""
``noop`` sub-command — test connectivity to a running server.

Sends a :attr:`RequestType.NOOP` request over ZMQ and prints
the response string returned by the engine's ``debug()`` method.
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
    resp = send_request(args, RequestType.NOOP)
    print(resp)


def register_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``noop`` sub-command."""
    parser = subparsers.add_parser(
        "noop",
        help="Ping the server (NOOP/heartbeat)",
        description=(
            "Send a NOOP request to verify connectivity with the LMCache server."
        ),
    )
    add_connection_args(parser)
    parser.set_defaults(func=_run)
