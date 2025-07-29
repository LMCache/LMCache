# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Literal, Optional
import socket

# Third Party
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)

ServiceKind = Literal["lookup", "offload"]


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


def get_zmq_rpc_path_lmcache(
    vllm_config: Optional["VllmConfig"] = None,
    service_name: ServiceKind = "lookup",
    rpc_port: int = 0,
    tp_rank: int = 0,
) -> str:
    """Get the ZMQ RPC path for LMCache lookup and offload communication."""
    # Third Party
    import vllm.envs as envs

    if vllm_config is None or vllm_config.kv_transfer_config is None:
        raise ValueError("A valid kv_transfer_config with engine_id is required.")

    if service_name not in {"lookup", "offload"}:
        raise ValueError(
            f"service_name must be 'lookup' or 'offload', got {service_name!r}"
        )

    base_url = envs.VLLM_RPC_BASE_PATH

    engine_id = vllm_config.kv_transfer_config.engine_id

    if isinstance(rpc_port, str):
        rpc_port = rpc_port + str(tp_rank)
    else:
        rpc_port += tp_rank

    logger.debug(
        "Base URL: %s, Engine: %s, Service Name: %s, RPC Port: %s",
        base_url,
        engine_id,
        service_name,
        rpc_port,
    )

    socket_path = (
        f"ipc://{base_url}/engine_{engine_id}_service_{service_name}_"
        f"lmcache_rpc_port_{rpc_port}"
    )

    return socket_path
