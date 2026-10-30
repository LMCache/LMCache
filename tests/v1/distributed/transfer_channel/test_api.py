# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the transfer channel data types.

Tests are written against the public contracts documented in
``lmcache/v1/common/transfer_channel/api.py`` and exercise only the
public interface.
"""

# Standard
import dataclasses
import subprocess
import sys
import textwrap

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.common.transfer_channel.api import CxlArenaDescriptor
from lmcache.v1.distributed.p2p import P2PClient
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelAddress,
    TransferChannelTransferResult,
)
from lmcache.v1.transport.grpc.method_registry import get_method_codec_registry


# =========================================================
# TransferChannelAddress
# =========================================================
def test_address_stores_offset_and_size():
    addr = TransferChannelAddress(offset=128, size=64)
    assert addr.offset == 128
    assert addr.size == 64


def test_address_is_immutable():
    addr = TransferChannelAddress(offset=0, size=16)
    with pytest.raises(dataclasses.FrozenInstanceError):
        addr.offset = 32  # type: ignore[misc]


def test_addresses_with_same_fields_are_equal():
    a = TransferChannelAddress(offset=10, size=20)
    b = TransferChannelAddress(offset=10, size=20)
    c = TransferChannelAddress(offset=10, size=21)
    assert a == b
    assert a != c


@pytest.mark.parametrize("cxl", [False, True])
def test_address_carries_its_cxl_arena_over_both_transports(cxl: bool) -> None:
    """The slab identity of a CXL address survives msgpack and protobuf; an
    address without one is unchanged for copying peers."""
    arena = CxlArenaDescriptor("pool", 0, 8192, 4096, "owner") if cxl else None
    address = TransferChannelAddress(4096, 4096, arena)
    wire = msgspec.msgpack.encode(address)
    assert msgspec.msgpack.decode(wire, type=TransferChannelAddress) == address
    codec = get_method_codec_registry(P2PClient).by_full_name[
        "lmcache.mp.P2PService.P2PQueryLookupResults"
    ]
    message = codec.response_encoder([address])
    assert message.addresses.addresses[0].HasField("cxl_arena") is cxl
    assert codec.response_decoder(message) == [address]


# =========================================================
# TransferChannelTransferResult
# =========================================================
def test_transfer_result_succeeded_mask_defaults_to_empty_list():
    result = TransferChannelTransferResult(finished=False)
    assert result.succeeded_mask == []


def test_transfer_result_is_finished_reflects_finished_flag():
    in_flight = TransferChannelTransferResult(finished=False)
    done = TransferChannelTransferResult(finished=True)
    assert in_flight.is_finished() is False
    assert done.is_finished() is True


def test_transfer_result_succeeded_mask_returns_flags():
    result = TransferChannelTransferResult(finished=True, succeeded_mask=[True, False])
    assert result.succeeded_mask == [True, False]


def test_transfer_result_default_succeeded_masks_are_independent():
    a = TransferChannelTransferResult(finished=False)
    b = TransferChannelTransferResult(finished=False)
    a.succeeded_mask.append(True)
    assert b.succeeded_mask == []


# =========================================================
# Import boundary
# =========================================================
def test_data_types_do_not_load_client_or_distributed_modules():
    script = textwrap.dedent(
        """
        import sys

        import lmcache.v1.common.transfer_channel.api

        endpoints = ("lmcache.v1.client", "lmcache.v1.distributed")
        loaded = sorted(name for name in sys.modules if name.startswith(endpoints))
        assert loaded == [], loaded
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
