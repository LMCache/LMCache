# SPDX-License-Identifier: Apache-2.0
"""Tests for the outcome-attribution fields on ``MP_LOOKUP_PREFETCH_END``.

Covers the per-tier hit split (``l1_hit_tokens`` / ``l2_hit_tokens``) and
``early_exit_reason``.  ``fold_unfold_grouped`` is patched out: these tests are
about how its chunk count and the storage manager's per-tier hit counts become event
metadata, not about the fold itself (which needs the native kernel).
"""

# Standard
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchHandle
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
import lmcache.v1.multiprocess.modules.lookup as lookup_module

CHUNK_SIZE = 256


def _lookup_key(world_size: int) -> IPCCacheServerKey:
    """A lookup-side IPC key (worker_id None -> expand over all workers)."""
    return IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        worker_id=None,
        token_ids=(0,),
        start=0,
        end=0,
        request_id="req-1",
        cache_salt="salt",
        num_kv_readers=1,
    )


def _end_metadata(
    chunk_hashes: list[bytes],
    l1_hit_cells: int = 0,
    found_count: int = 0,
    num_groups: int = 1,
    world_size: int = 1,
    layout_found: bool = True,
    group_layouts_found: bool = True,
) -> dict:
    """Drive a full lookup/poll cycle and return the END event's metadata.

    Args:
        chunk_hashes: Chunk hashes the token hasher reports; empty triggers the
            ``empty_chunk_hashes`` early exit.
        l1_hit_cells: Hit cells (one per row and chunk) that L1 served.
        found_count: Prefix, in chunks, served after L2 completed.
        num_groups: Object groups per chunk.
        world_size: kv_rank shards per chunk.
        layout_found: False triggers the ``no_gpu_context`` early exit.
        group_layouts_found: False triggers the ``no_group_layout_descs``
            early exit.

    Returns:
        The metadata dict of the published ``MP_LOOKUP_PREFETCH_END`` event.
    """
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock() if layout_found else None
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1] * num_groups, world_size=world_size
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = (
        {group_id: MagicMock() for group_id in range(num_groups)}
        if group_layouts_found
        else {}
    )
    ctx.token_hasher.compute_chunk_hashes.return_value = chunk_hashes
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req-1",
        total_requested_keys=len(chunk_hashes) * world_size * num_groups,
        submit_time=time.monotonic(),
    )
    ctx.storage_manager.query_prefetch_hit_counts.return_value = (l1_hit_cells, 0)

    module = LookupModule(ctx)
    with patch.object(
        lookup_module, "fold_unfold_grouped", return_value=(found_count, MagicMock())
    ):
        module.lookup(_lookup_key(world_size), tp_size=1)
        module.query_prefetch_status("req-1")

    for call in ctx.event_bus.publish.call_args_list:
        event = call.args[0]
        if event.event_type is EventType.MP_LOOKUP_PREFETCH_END:
            return event.metadata
    raise AssertionError("no MP_LOOKUP_PREFETCH_END event was published")


@pytest.mark.parametrize(
    (
        "num_chunks",
        "num_groups",
        "l1_hit_cells",
        "found_count",
        "l1_chunks",
        "l2_chunks",
    ),
    [
        (4, 1, 4, 4, 4, 0),
        (4, 1, 0, 4, 0, 4),
        (4, 1, 2, 4, 2, 2),
        (4, 1, 0, 0, 0, 0),
        # Two rows per chunk: six L1 cells are three chunks, all L1-served.
        (3, 2, 6, 3, 3, 0),
    ],
    ids=[
        "all_l1",
        "all_l2",
        "l2_extends_l1",
        "cold_miss",
        "two_rows_per_chunk",
    ],
)
def test_end_event_splits_hit_tokens_by_tier(
    num_chunks,
    num_groups,
    l1_hit_cells,
    found_count,
    l1_chunks,
    l2_chunks,
):
    """L2 is credited with however far it extended the L1-servable prefix."""
    meta = _end_metadata(
        chunk_hashes=[f"c{i}".encode() for i in range(num_chunks)],
        l1_hit_cells=l1_hit_cells,
        found_count=found_count,
        num_groups=num_groups,
    )

    assert meta["l1_hit_tokens"] == l1_chunks * CHUNK_SIZE
    assert meta["l2_hit_tokens"] == l2_chunks * CHUNK_SIZE
    assert meta["l1_hit_tokens"] + meta["l2_hit_tokens"] == meta["hit_tokens"]
    assert meta["early_exit_reason"] == ""


def test_l1_hit_cells_above_found_count_is_clamped():
    """L1 can never be credited with more than the total hit."""
    meta = _end_metadata(
        chunk_hashes=[b"c0", b"c1", b"c2", b"c3"],
        l1_hit_cells=4,
        found_count=2,
    )

    assert meta["l1_hit_tokens"] == 2 * CHUNK_SIZE
    assert meta["l2_hit_tokens"] == 0
    assert meta["l1_hit_tokens"] + meta["l2_hit_tokens"] == meta["hit_tokens"]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"chunk_hashes": [b"c0"], "layout_found": False}, "no_gpu_context"),
        ({"chunk_hashes": []}, "empty_chunk_hashes"),
        (
            {"chunk_hashes": [b"c0"], "group_layouts_found": False},
            "no_group_layout_descs",
        ),
    ],
    ids=["no_gpu_context", "empty_chunk_hashes", "no_group_layout_descs"],
)
def test_early_exit_reason(kwargs, reason):
    """Each early-exit branch names itself on the END event."""
    meta = _end_metadata(**kwargs)

    assert meta["early_exit_reason"] == reason
    assert meta["requested_tokens"] == 0
    assert meta["hit_tokens"] == 0
