# SPDX-License-Identifier: Apache-2.0
"""MooncakestoreConnector.batched_get_non_blocking() goes through batched_get().

No Mooncake build needed: the connector is created without __init__ and
batched_get() is stubbed.
"""

# Standard
from unittest.mock import MagicMock
import asyncio

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)


def _connector(results):
    conn = object.__new__(MooncakestoreConnector)
    calls = []

    async def batched_get(keys):
        calls.append(list(keys))
        return results

    conn.batched_get = batched_get
    return conn, calls


def _obj():
    return MagicMock(spec=MemoryObj)


def test_one_batched_get_and_full_hit():
    objs = [_obj(), _obj(), _obj()]
    conn, calls = _connector(objs)
    keys = ["k0", "k1", "k2"]
    got = asyncio.run(conn.batched_get_non_blocking("lookup", keys))
    assert got == objs
    assert calls == [keys]
    for o in objs:
        o.ref_count_down.assert_not_called()


def test_prefix_until_first_miss_and_rest_released():
    a, b, c = _obj(), _obj(), _obj()
    conn, _ = _connector([a, b, None, c])
    got = asyncio.run(conn.batched_get_non_blocking("lookup", ["k0", "k1", "k2", "k3"]))
    assert got == [a, b]
    c.ref_count_down.assert_called_once()
    a.ref_count_down.assert_not_called()


def test_first_key_missing_returns_empty():
    b = _obj()
    conn, _ = _connector([None, b])
    got = asyncio.run(conn.batched_get_non_blocking("lookup", ["k0", "k1"]))
    assert got == []
    b.ref_count_down.assert_called_once()
