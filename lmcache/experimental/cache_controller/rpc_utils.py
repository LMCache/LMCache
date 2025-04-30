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

import zmq
import zmq.asyncio

from lmcache.logging import init_logger

logger = init_logger(__name__)


def get_zmq_context():
    return zmq.asyncio.Context.instance()


def get_zmq_socket(context, socket_path: str, protocol: str, role):
    """
    Create a ZeroMQ socket with the specified protocol and role.
    """
    socket_addr = f"{protocol}://{socket_path}"
    socket = context.socket(role)
    if role in [zmq.PUB, zmq.PUSH, zmq.REP]:  # type: ignore[attr-defined]
        socket.bind(socket_addr)
    elif role in [zmq.SUB, zmq.PULL, zmq.REQ]:  # type: ignore[attr-defined]
        socket.connect(socket_addr)
    else:
        raise ValueError(f"Invalid role: {role}")

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
