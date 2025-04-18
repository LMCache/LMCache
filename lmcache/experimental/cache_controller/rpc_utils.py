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
    if role in [zmq.PUB, zmq.PUSH, zmq.REP]:
        socket.bind(socket_addr)
    elif role in [zmq.SUB, zmq.PULL, zmq.REQ]:
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
        socket.setsockopt(zmq.LINGER, linger)
        socket.close()
    except Exception as e:
        logger.error(f"Warning: Failed to close socket cleanly: {e}")
