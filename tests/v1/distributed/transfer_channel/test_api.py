# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the transfer channel data types.

Tests are written against the public contracts documented in
``lmcache/v1/distributed/transfer_channel/api.py`` and exercise only the
public interface.
"""

# Standard
import dataclasses

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.distributed.internal_api import CxlArenaDescriptor
from lmcache.v1.distributed.transfer_channel import (
    MemoryRegionAddress,
    TransferChannelAddress,
    TransferChannelReadResult,
)
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_codec_registry,
)


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
def test_region_address_preserves_lookup_wire_contract(cxl: bool) -> None:
    """Neutral addresses retain legacy ZMQ fields and protobuf compatibility."""
    arena = CxlArenaDescriptor("pool", 0, 8192, 4096, "owner") if cxl else None
    address = MemoryRegionAddress(4096, 4096, arena, 30)
    wire = msgspec.msgpack.encode(address)
    assert msgspec.msgpack.decode(wire)["cxl_ttl_seconds"] == 30
    legacy = msgspec.msgpack.decode(wire, type=TransferChannelAddress)
    assert legacy == address
    assert legacy.read_ttl_seconds == 30
    codec = get_method_codec_registry().by_full_name[
        "lmcache.mp.P2PService.P2PQueryLookupResults"
    ]
    message = codec.response_encoder([address])
    assert message.addresses.addresses[0].cxl_ttl_seconds == 30
    assert codec.response_decoder(message) == [address]


# =========================================================
# TransferChannelReadResult
# =========================================================
def test_read_result_succeeded_mask_defaults_to_empty_list():
    result = TransferChannelReadResult(finished=False)
    assert result.succeeded_mask == []


def test_read_result_is_finished_reflects_finished_flag():
    in_flight = TransferChannelReadResult(finished=False)
    done = TransferChannelReadResult(finished=True)
    assert in_flight.is_finished() is False
    assert done.is_finished() is True


def test_read_result_succeeded_mask_returns_flags():
    result = TransferChannelReadResult(finished=True, succeeded_mask=[True, False])
    assert result.succeeded_mask == [True, False]


def test_read_result_default_succeeded_masks_are_independent():
    a = TransferChannelReadResult(finished=False)
    b = TransferChannelReadResult(finished=False)
    a.succeeded_mask.append(True)
    assert b.succeeded_mask == []
