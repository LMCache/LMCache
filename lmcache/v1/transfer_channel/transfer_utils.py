# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum
from typing import Any, Union

# Third Party
import msgspec


def get_correct_device(device: str, worker_id: int) -> str:
    """
    Get the correct device based on the given device string.

    Args:
        device (str): The device string, could be cpu or cuda.
        worker_id (int): The worker id to determine the cuda device.

    Returns:
        str: The correct device string with device id.
    """
    if device == "cpu":
        return "cpu"
    elif device.startswith("cuda"):
        return f"cuda:{worker_id}"
    else:
        raise ValueError(f"Invalid device: {device}")


def maybe_transpose(transfer_spec: Any = None) -> bool:
    """Check if we need to transpose the kv based on transfer spec"""
    if transfer_spec is None:
        return False
    if not getattr(transfer_spec, "receiver_init_ports", None) or not getattr(
        transfer_spec, "sender_tp_size", None
    ):
        return False

    return len(transfer_spec.receiver_init_ports) > transfer_spec.sender_tp_size


class SideMsgBase(msgspec.Struct, tag=True):
    """Base class for all side-related messages during initialization"""

    pass


# Side messages during initialization
class InitSideMsgBase(SideMsgBase):
    """Base class for all side-related messages during initialization"""

    pass


class P2PInitSideMsg(InitSideMsgBase):
    """P2P specific initialization message"""

    pass


# Side return messages during initialization
class InitSideRetMsgBase(SideMsgBase):
    """Base class for all side-related messages during initialization"""

    pass


class P2PInitSideRetMsg(InitSideRetMsgBase):
    """P2P specific initialization return message"""

    peer_lookup_url: str


class TransferRole(Enum):
    SENDER = "sender"
    RECEIVER = "receiver"
    # TODO(novahow): for role switch
    BOTH = "both"


SideMsg = Union[
    P2PInitSideMsg,
    P2PInitSideRetMsg,
]
