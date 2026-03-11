# SPDX-License-Identifier: Apache-2.0

"""
``end-session`` sub-command — end a server-side session.

Sends a :attr:`RequestType.END_SESSION` request over ZMQ
to remove the session state for a given request ID.
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
    send_request(
        args,
        RequestType.END_SESSION,
        payloads=[args.request_id],
    )
    logger.info(
        "Session %s ended successfully.",
        args.request_id,
    )


def register_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``end-session`` sub-command."""
    parser = subparsers.add_parser(
        "end-session",
        help="End a server-side session",
        description=(
            "Send an END_SESSION request to remove "
            "session state for a given request ID."
        ),
    )
    add_connection_args(parser)
    parser.add_argument(
        "request_id",
        type=str,
        help="The request ID of the session to end.",
    )
    parser.set_defaults(func=_run)
