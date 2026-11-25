# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`RequestIndex` contracts the HTTP tests in
``test_requests_api.py`` do not reach."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.views.request_index import RequestIndex


def _store(
    seq: int,
    chunk_hash: bytes,
    offset: int,
    parent: bytes | None,
    request_ids: list[str],
    event_type: CacheEventType = CacheEventType.STORE,
) -> CacheEventBatch:
    key = ObjectKey(chunk_hash=chunk_hash, model_name="m", kv_rank=0)
    return CacheEventBatch(
        instance_id="mp-0",
        incarnation=1,
        seq=seq,
        event_type=event_type,
        tier=Tier.L1,
        backend="dram",
        entries=[
            CacheEventEntry(
                key=key.to_encoded_object_key(),
                token_ids=[offset, offset + 1],
                token_offset=offset,
                parent_hash_hex=parent.hex() if parent is not None else "",
                request_ids=request_ids,
            )
        ],
    )


@pytest.mark.parametrize("bound", [0, -1])
def test_non_positive_bound_is_rejected(bound):
    with pytest.raises(ValueError):
        RequestIndex(max_chunks=bound)


@pytest.mark.parametrize("bound", [0, "many"])
def test_invalid_configured_bound_is_rejected(bound):
    config = MPCoordinatorConfig(extra_config={"request_index_max_chunks": bound})
    with pytest.raises(ValueError):
        RequestIndex.from_config(config, views=None)  # type: ignore[arg-type]


def test_configured_bound_is_applied():
    config = MPCoordinatorConfig(extra_config={"request_index_max_chunks": 1})
    index = RequestIndex.from_config(config, views=None)  # type: ignore[arg-type]
    index.consume(_store(1, b"\x01", 0, None, ["req-1"]))
    index.consume(_store(2, b"\x02", 2, b"\x01", ["req-1"]))

    trace = index.get_request("mp-0", "req-1")

    assert trace is not None
    assert [c.chunk_hash for c in trace.chunks] == [b"\x02"]


def test_fencing_an_instance_keeps_its_requests():
    index = RequestIndex()
    index.consume(_store(1, b"\x01", 0, None, ["req-1"]))

    index.fence_instance("mp-0")

    assert index.get_request("mp-0", "req-1") is not None


def test_non_store_batches_record_nothing():
    index = RequestIndex()
    index.consume(_store(1, b"\x01", 0, None, ["req-1"], CacheEventType.ACCESS))

    assert index.get_request("mp-0", "req-1") is None


def test_oldest_request_is_forgotten_past_the_bound():
    index = RequestIndex(max_requests=2)
    index.consume(_store(1, b"\x01", 0, None, ["req-1", "req-2", "req-3"]))

    assert index.get_request("mp-0", "req-1") is None
    assert index.get_request("mp-0", "req-3") is not None


def test_a_parent_cycle_ends_the_walk():
    """Corrupt parent links must not hang the request handler."""
    index = RequestIndex()
    index.consume(_store(1, b"\x01", 2, b"\x02", ["req-1"]))
    index.consume(_store(2, b"\x02", 4, b"\x01", ["req-1"]))

    trace = index.get_request("mp-0", "req-1")

    assert trace is not None
    assert [c.chunk_hash for c in trace.chunks] == [b"\x01", b"\x02"]
    assert trace.complete is False


def test_a_forgotten_request_leaves_its_chunks_request_list():
    index = RequestIndex(max_requests=2)
    index.consume(_store(1, b"\x01", 0, None, ["req-1"]))
    index.consume(_store(2, b"\x01", 0, None, ["req-2", "req-3"]))

    refs = index.get_chunk_requests(b"\x01")

    assert refs is not None
    assert [r.request_id for r in refs] == ["req-2", "req-3"]


def test_requests_have_their_own_bound():
    """Every request with a hit takes a request slot, so requests are not
    held to the chunk bound."""
    index = RequestIndex(max_chunks=1, max_requests=3)
    index.consume(_store(1, b"\x01", 0, None, ["req-1", "req-2", "req-3"]))

    assert all(
        index.get_request("mp-0", rid) is not None
        for rid in ("req-1", "req-2", "req-3")
    )


def test_configured_request_bound_is_applied():
    config = MPCoordinatorConfig(extra_config={"request_index_max_requests": 1})
    index = RequestIndex.from_config(config, views=None)  # type: ignore[arg-type]
    index.consume(_store(1, b"\x01", 0, None, ["req-1", "req-2"]))

    assert index.get_request("mp-0", "req-1") is None
    assert index.get_request("mp-0", "req-2") is not None


@pytest.mark.parametrize("bound", [0, "many"])
def test_invalid_configured_request_bound_is_rejected(bound):
    config = MPCoordinatorConfig(extra_config={"request_index_max_requests": bound})
    with pytest.raises(ValueError):
        RequestIndex.from_config(config, views=None)  # type: ignore[arg-type]
