# SPDX-License-Identifier: Apache-2.0
"""Cache-line flushes keep the bytes of any range intact."""

# Standard
import ctypes

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import cache_flush_range, cache_flush_supported


@pytest.mark.skipif(not cache_flush_supported(), reason="requires CLFLUSHOPT")
def test_flush_preserves_bytes() -> None:
    data = bytes(range(256)) * 256
    buffer = ctypes.create_string_buffer(data)
    address = ctypes.addressof(buffer)
    cache_flush_range(address + 3, 61)  # unaligned start and length
    cache_flush_range(address, len(data))
    cache_flush_range(address, 0)
    assert buffer.raw[: len(data)] == data
