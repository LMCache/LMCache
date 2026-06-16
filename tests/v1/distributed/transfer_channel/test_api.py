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
import pytest

# First Party
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelAddress,
    TransferChannelReadResult,
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


# =========================================================
# TransferChannelReadResult
# =========================================================
def test_read_result_succeeded_defaults_to_empty_list():
    result = TransferChannelReadResult(finished=False)
    assert result.succeeded == []
    assert result.succeed_addresses() == []


def test_read_result_is_finished_reflects_finished_flag():
    in_flight = TransferChannelReadResult(finished=False)
    done = TransferChannelReadResult(finished=True)
    assert in_flight.is_finished() is False
    assert done.is_finished() is True


def test_read_result_succeed_addresses_returns_succeeded():
    addrs = [
        TransferChannelAddress(offset=0, size=16),
        TransferChannelAddress(offset=16, size=16),
    ]
    result = TransferChannelReadResult(finished=True, succeeded=addrs)
    assert result.succeed_addresses() == addrs


def test_read_result_default_succeeded_lists_are_independent():
    a = TransferChannelReadResult(finished=False)
    b = TransferChannelReadResult(finished=False)
    a.succeeded.append(TransferChannelAddress(offset=0, size=8))
    assert b.succeeded == []
