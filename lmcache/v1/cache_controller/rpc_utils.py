# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
import socket

# Third Party
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def get_zmq_context():
    return zmq.asyncio.Context.instance()


def get_zmq_socket(
    context, socket_path: str, protocol: str, role, bind_or_connect: str
):
    """
    Create a ZeroMQ socket with the specified protocol and role.
    """
    socket_addr = f"{protocol}://{socket_path}"
    socket = context.socket(role)
    if bind_or_connect == "bind":
        socket.bind(socket_addr)
    elif bind_or_connect == "connect":
        socket.connect(socket_addr)
    else:
        raise ValueError(f"Invalid bind_or_connect: {bind_or_connect}")

    return socket


def close_zmq_socket(socket: zmq.asyncio.Socket, linger: int = 0) -> None:
    """
    Close a ZeroMQ socket cleanly.

    :param socket: The zmq.Socket to be closed.
    :param linger: LINGER period (in milliseconds).
    Default is 0 (drop immediately).
    """
    try:
        socket.setsockopt(zmq.LINGER, linger)  # type: ignore[attr-defined]
        socket.close()
    except Exception as e:
        logger.error(f"Warning: Failed to close socket cleanly: {e}")


def get_ip():
    """
    Get the local IP address of the machine.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # "Connect" to a public IP — just to determine local IP
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        logger.warning(
            "Failed to get local IP address. Falling back to loopback address."
        )
        return "127.0.0.1"  # Fallback to loopback
    finally:
        s.close()
