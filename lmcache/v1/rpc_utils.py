# SPDX-License-Identifier: Apache-2.0
# Standard
import socket

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def get_ip():
    """
    Get the local IP address of the machine.

    Kept here because it is used by the multiprocess coordinator; the ZMQ
    helpers and IPC socket-path builder moved to
    ``lmcache.inprocess.v1.rpc_utils``.
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
