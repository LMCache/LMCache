# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Literal, Optional
import hashlib
import os
import socket

# Third Party
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

ServiceKind = Literal["lookup", "offload", "lookup_worker", "lookup_scheduler"]

# ``sockaddr_un.sun_path`` is 108 bytes on Linux (107 usable chars + NUL), and
# ZMQ rejects IPC paths longer than that. A long base directory (e.g. a
# supercomputer scratch filesystem assigned via ``TMPDIR``) plus the
# UUID-heavy descriptive socket name can easily exceed it, so the generated
# path must be kept within this limit. See
# https://github.com/LMCache/LMCache/issues/3529.
IPC_SOCKET_PATH_MAX_LEN = 107

# Default timeout constants for socket operations (in milliseconds)
DEFAULT_SOCKET_RECV_TIMEOUT_MS = 30000
DEFAULT_SOCKET_SEND_TIMEOUT_MS = 10000


def get_zmq_context(use_asyncio: bool = True):
    if use_asyncio:
        return zmq.asyncio.Context.instance()
    else:
        return zmq.Context.instance()


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


def get_zmq_socket_with_timeout(
    context,
    socket_path: str,
    protocol: str,
    role,
    bind_or_connect: str,
    recv_timeout_ms: int,
    send_timeout_ms: int,
) -> zmq.asyncio.Socket:
    """
    Create a ZeroMQ socket with timeout settings.
    """
    socket = get_zmq_socket(
        context,
        socket_path,
        protocol,
        role,
        bind_or_connect,
    )
    # Only set RCVTIMEO for client role connect sockets
    if bind_or_connect == "connect":
        socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, send_timeout_ms)
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
        logger.error("Warning: Failed to close socket cleanly: %s", e)


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


def get_zmq_rpc_path_lmcache(
    engine_id: str,
    service_name: ServiceKind = "lookup",
    rpc_port: int = 0,
    rank: int = 0,
    base_url: Optional[str] = None,
) -> str:
    """Get the ZMQ RPC path for LMCache lookup and offload communication.

    Args:
        engine_id: The engine ID for the RPC path.
        service_name: The service name, one of 'lookup', 'offload',
            'lookup_worker', 'lookup_scheduler'.
        rpc_port: The RPC port number.
        rank: The rank of the worker.
        base_url: Optional base URL for the socket path. If not provided,
            will try to use VLLM_RPC_BASE_PATH or fallback to /tmp/vllm_rpc.

    Returns:
        The ZMQ socket path string.
    """
    if base_url is None:
        # Try to import vllm.envs, fallback to default if not available
        try:
            # Third Party
            import vllm.envs as envs

            base_url = envs.VLLM_RPC_BASE_PATH
        except (ImportError, ModuleNotFoundError):
            # Fallback for testing environments without vllm
            base_url = "/tmp/vllm_rpc"
            logger.debug("vllm not available, using default base_url: %s", base_url)
            # Ensure the directory exists for IPC socket
            os.makedirs(base_url, exist_ok=True)

    if service_name not in {"lookup", "offload", "lookup_worker", "lookup_scheduler"}:
        raise ValueError(
            f"service_name must be 'lookup' or 'offload', got {service_name!r}"
        )

    if isinstance(rpc_port, str):
        rpc_port = rpc_port + str(rank)
    else:
        rpc_port += rank

    logger.debug(
        "Base URL: %s, Engine: %s, Service Name: %s, RPC Port: %s",
        base_url,
        engine_id,
        service_name,
        rpc_port,
    )

    socket_path = (
        f"{base_url}/engine_{engine_id}_service_{service_name}_"
        f"lmcache_rpc_port_{rpc_port}"
    )

    return _enforce_ipc_path_limit(socket_path, base_url)


def _ipc_path_nbytes(path: str) -> int:
    """Length of ``path`` as the kernel sees it: ``sun_path`` is byte-sized,
    not character-sized, so a non-ASCII base directory must be measured after
    encoding (the ``ipc://`` scheme prefix is not part of ``sun_path``)."""
    return len(os.fsencode(path))


def _enforce_ipc_path_limit(socket_path: str, base_url: str) -> str:
    """Keep an IPC socket path within the ``sockaddr_un`` length limit.

    The descriptive path is preferred for readability/debuggability and is
    returned unchanged when it already fits. When it would exceed
    :data:`IPC_SOCKET_PATH_MAX_LEN` (long ``base_url`` on shared filesystems),
    the descriptive filename is replaced by a short, deterministic digest of
    the full path so that every caller — both the binding server and the
    connecting client — derives the *same* shortened path from the same
    inputs. The digest is taken over the full descriptive path, so distinct
    ``(engine_id, service_name, rpc_port, rank)`` tuples keep distinct sockets.

    Length is measured in bytes (``sun_path`` is byte-sized), so non-ASCII
    base directories are handled correctly.

    Args:
        socket_path: The descriptive ``{base_url}/engine_..._rpc_port_N`` path.
        base_url: The base directory the socket should live under.

    Returns:
        ``socket_path`` if it already fits, otherwise a shortened path under
        ``base_url`` that is within the limit.

    Raises:
        ValueError: If ``base_url`` is itself so long that even a shortened
            name cannot fit. Falling back to a different directory is *not*
            done on purpose: the binding server and connecting client may be
            separate processes with different ``TMPDIR``, so a temp-dir
            fallback could resolve differently on each end and silently break
            IPC. Raising here is deterministic on both ends; the user must
            point ``VLLM_RPC_BASE_PATH`` / ``TMPDIR`` at a shorter directory.
    """
    if _ipc_path_nbytes(socket_path) <= IPC_SOCKET_PATH_MAX_LEN:
        return socket_path

    digest = hashlib.sha256(os.fsencode(socket_path)).hexdigest()[:16]
    shortened = os.path.join(base_url, f"lmcache_rpc_{digest}")

    if _ipc_path_nbytes(shortened) > IPC_SOCKET_PATH_MAX_LEN:
        raise ValueError(
            f"Cannot construct an IPC socket path within the "
            f"{IPC_SOCKET_PATH_MAX_LEN}-byte sockaddr_un limit: base directory "
            f"{base_url!r} is too long to host even a shortened socket name. "
            f"Set VLLM_RPC_BASE_PATH (or TMPDIR) to a shorter directory."
        )

    logger.warning(
        "IPC socket path (%d bytes) exceeds the %d-byte sockaddr_un limit; "
        "using shortened path %r instead of %r.",
        _ipc_path_nbytes(socket_path),
        IPC_SOCKET_PATH_MAX_LEN,
        shortened,
        socket_path,
    )
    return shortened
