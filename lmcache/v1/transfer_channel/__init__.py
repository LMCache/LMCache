# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.nixl_channel import NixlChannel


# TODO(Jiayi): Refactor this function when we support more channels.
def CreateTransferChannel(
    channel_type: str,
    async_mode: bool,
    role: str,
    buffer_ptr: int,
    buffer_size: int,
    align_bytes: int,
    tp_rank: int,
    peer_init_url: str,
    **kwargs,
) -> BaseTransferChannel:
    """
    Create a transfer channel based on the specified channel type.
    Currently, only "nixl" channel type is supported.

    :param channel_type: Type of the transfer channel (e.g., "nixl").
    :param async_mode: Whether to operate in asynchronous mode.
    :param role: Role of the channel (e.g., "both", "sender" or "receiver").
    :param buffer_ptr: Pointer to the pre-allocated buffer.
    :param buffer_size: Size of the pre-allocated buffer in bytes.
    :param align_bytes: Alignment requirement in bytes.
    :param tp_rank: Tensor parallel rank of the current process.
    :param peer_init_url: Initialization URL for the peer.
    :kwargs: Additional keyword arguments specific to the channel type.

    :return: An instance of the specified transfer channel.
    """

    assert channel_type in ["nixl"], f"Unsupported channel type: {channel_type}"

    assert "backends" in kwargs, (
        "`backends` must be provided to create nixl transfer channel."
    )
    transfer_channel = NixlChannel(
        async_mode=async_mode,
        role=role,
        buffer_ptr=buffer_ptr,
        buffer_size=buffer_size,
        align_bytes=align_bytes,
        tp_rank=tp_rank,
        peer_init_url=peer_init_url,
        **kwargs,
    )
    return transfer_channel
