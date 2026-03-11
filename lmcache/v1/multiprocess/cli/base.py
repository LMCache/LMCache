# SPDX-License-Identifier: Apache-2.0

"""
Base utilities for CLI sub-commands.

Every module under ``cli/commands/`` must define a module-level
``register_command`` function that matches :class:`CommandRegistrar`.
The CLI entry-point discovers and calls these registrars
automatically, so adding a new command never requires touching
existing files.

:func:`send_request` is a lightweight helper shared by every
command that communicates with the ZMQ server.
"""

# Standard
from typing import Any, Protocol
import argparse
import sys

# Third Party
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType

logger = init_logger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5555
DEFAULT_TIMEOUT = 5.0  # seconds


class CommandRegistrar(Protocol):
    """Protocol that every command module must satisfy."""

    def __call__(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        """Register one sub-command on *subparsers*."""
        ...


def add_connection_args(
    parser: argparse.ArgumentParser,
) -> None:
    """Add ``--host`` and ``--port`` args for server connection.

    Shared by every command that talks to a running server.
    """
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="ZMQ server host. Default: %s." % DEFAULT_HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="ZMQ server port. Default: %d." % DEFAULT_PORT,
    )


def send_request(
    args: argparse.Namespace,
    request_type: RequestType,
    payloads: list[Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Send a single ZMQ request and return the response.

    Creates a temporary :class:`MessageQueueClient`, submits
    the request, waits for the response and tears down the
    connection.

    Args:
        args: Parsed CLI args (must contain ``host`` and
            ``port``).
        request_type: The protocol request type to send.
        payloads: Positional payloads matching the protocol
            definition.  Defaults to an empty list.
        timeout: Seconds to wait before giving up.

    Returns:
        The decoded response from the server, or *None*
        if the protocol defines no response.
    """
    if payloads is None:
        payloads = []

    url = "tcp://%s:%d" % (args.host, args.port)
    ctx = zmq.Context()
    ctx.linger = 0
    client = MessageQueueClient(url, ctx)

    try:
        future: MessagingFuture[Any] = client.submit_request(request_type, payloads)
        if not future.wait(timeout):
            logger.error(
                "Request %s timed out after %.1fs",
                request_type.name,
                timeout,
            )
            sys.exit(1)
        return future.result()
    finally:
        client.close()
        # NOTE: skip ctx.term() — MessageQueueClient.close()
        # does not close internal inproc sockets
        # (task_notifier / task_waiter), so ctx.term()
        # blocks forever.  The CLI process exits right
        # after, letting the OS reclaim all resources.
