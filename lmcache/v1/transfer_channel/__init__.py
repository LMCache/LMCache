# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.nixl_channel import NixlChannel


# TODO(Jiayi): Refactor this function when we support more channels.
def CreateTransferChannel(
    channel_type: str,
    role: str,
    buffer_ptr: int,
    buffer_size: int,
    align_bytes: int,
    tp_rank: int,
    peer_init_url: str,
    **kwargs,
) -> BaseTransferChannel:
    assert channel_type in ["nixl"], f"Unsupported channel type: {channel_type}"

    assert "backends" in kwargs, (
        "`backends` must be provided to create nixl transfer channel."
    )
    transfer_channel = NixlChannel(
        role=role,
        buffer_ptr=buffer_ptr,
        buffer_size=buffer_size,
        align_bytes=align_bytes,
        tp_rank=tp_rank,
        peer_init_url=peer_init_url,
        **kwargs,
    )
    return transfer_channel
