# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Union


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


class InitSideMsgBase(msgspec.Struct, tag=True):
    """Base class for all side-related messages during initialization"""

    pass


class P2PInitSideMsg(InitSideMsgBase):
    """P2P specific initialization message"""

    pass


class P2PInitSideRetMsg(InitSideMsgBase):
    """P2P specific initialization return message"""

    peer_lookup_url: str


InitSideMsg = Union[
    P2PInitSideMsg,
    P2PInitSideRetMsg,
]
