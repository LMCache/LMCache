# SPDX-License-Identifier: Apache-2.0
"""require_num_kv_readers: exact count only; keys without it are rejected."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


def _key(num_kv_readers: int) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        num_kv_readers=num_kv_readers,
        worker_id=0,
        token_ids=(1, 2, 3),
        start=0,
        end=3,
        request_id="req",
    )


def test_reader_count_passes_through_exactly():
    assert _key(1).require_num_kv_readers() == 1
    assert _key(4).require_num_kv_readers() == 4


@pytest.mark.parametrize("missing", [0, -1])
def test_keys_without_reader_count_are_rejected(missing):
    with pytest.raises(ValueError, match="num_kv_readers"):
        _key(missing).require_num_kv_readers()
