# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Union

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


SideMsg = Union[
    P2PInitSideMsg,
    P2PInitSideRetMsg,
]


def handle_p2p_init_side_msg(
    req: InitSideMsgBase,
    peer_lookup_url: str,
) -> InitSideRetMsgBase:
    """
    Handle P2P initialization side messages.

    This is a utility function that can be used by transfer channel
    implementations to handle P2PInitSideMsg.

    :param req: The initialization-related side message from the peer.
    :param peer_lookup_url: The peer lookup URL to include in the response.
    :return: A side message to be sent back to the peer.
    :raises ValueError: If the message type is not supported or peer_lookup_url is None.
    """
    if isinstance(req, P2PInitSideMsg):
        if peer_lookup_url is None:
            raise ValueError("P2PInitSideMsg requires peer_lookup_url to be configured")
        return P2PInitSideRetMsg(peer_lookup_url=peer_lookup_url)
    else:
        raise ValueError(f"Unsupported InitSideMsg type: {type(req)}")
