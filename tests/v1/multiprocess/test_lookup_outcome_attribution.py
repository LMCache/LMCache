# SPDX-License-Identifier: Apache-2.0
"""Tests for the outcome-attribution fields on ``MP_LOOKUP_PREFETCH_END``.

Covers the per-tier hit split (``l1_hit_tokens`` / ``l2_hit_tokens`` and
``l1_hit_keys`` / ``l2_hit_keys``) and ``early_exit_reason``.  The storage
manager is mocked to return a finished ``PrefetchResult`` grid; the fold runs
for real.
"""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchHandle, PrefetchResult
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule

CHUNK_SIZE = 256


def _lookup_key(
    world_size: int,
    start_chunks: int = 0,
    prompt_tokens: int = 1,
    engine_hit_tokens: int | None = None,
) -> IPCCacheServerKey:
    """A lookup-side IPC key (worker_id None -> expand over all workers).

    ``engine_hit_tokens`` becomes the connector's
    ``lmcache.hint.engine_computed_tokens`` hint; None sends no hint.
    """
    request_configs = (
        None
        if engine_hit_tokens is None
        else {"lmcache.hint.engine_computed_tokens": engine_hit_tokens}
    )
    return IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        worker_id=None,
        token_ids=tuple(range(prompt_tokens)),
        start=start_chunks * CHUNK_SIZE,
        end=0,
        request_id="req-1",
        cache_salt="salt",
        request_configs=request_configs,
        num_kv_readers=1,
    )


def _bitmap(pattern: str) -> Bitmap:
    """A bitmap with bit ``i`` set iff ``pattern[i] == "1"``."""
    bitmap = Bitmap(len(pattern))
    for i, bit in enumerate(pattern):
        if bit == "1":
            bitmap.set(i)
    return bitmap


def _result(l1_rows: list[str], l2_rows: list[str]) -> PrefetchResult:
    """A finished result from per-row L1 and L2 hit patterns (disjoint)."""
    hit_rows = [
        "".join("1" if "1" in (a, b) else "0" for a, b in zip(l1, l2, strict=True))
        for l1, l2 in zip(l1_rows, l2_rows, strict=True)
    ]
    return PrefetchResult(
        hit_cells=[_bitmap(row) for row in hit_rows],
        l1_hit_cells=[_bitmap(row) for row in l1_rows],
        l2_hit_cells=[_bitmap(row) for row in l2_rows],
    )


def _end_metadata(
    chunk_hashes: list[bytes],
    windows: list[int] | None = None,
    world_size: int = 1,
    l1_rows: list[str] | None = None,
    l2_rows: list[str] | None = None,
    layout_found: bool = True,
    group_layouts_found: bool = True,
    start_chunks: int = 0,
    prompt_tokens: int = 1,
    engine_hit_tokens: int | None = None,
) -> dict:
    """Drive a full lookup/poll cycle and return the END event's metadata.

    Args:
        chunk_hashes: Chunk hashes the token hasher reports; empty triggers the
            ``empty_chunk_hashes`` early exit.
        windows: Per-object-group window sizes (``-1`` is full attention);
            defaults to one full-attention group.
        world_size: kv_rank shards per chunk.
        l1_rows: Per-row L1 hit patterns, one row per (object group, kv rank)
            in group-major order; defaults to all misses.
        l2_rows: Per-row L2 hit patterns, same layout as ``l1_rows``;
            defaults to all misses.
        layout_found: False triggers the ``no_gpu_context`` early exit.
        group_layouts_found: False triggers the ``no_group_layout_descs``
            early exit.
        start_chunks: Engine-covered chunks the key skips; the hit patterns
            then cover only the chunks after them.
        prompt_tokens: Length of the key's prompt (the request-level
            denominator); the chunk hashes are mocked, so it is independent
            of ``chunk_hashes``.
        engine_hit_tokens: The engine's prefix-cache hit sent as a hint, or
            None for no hint.

    Returns:
        The metadata dict of the published ``MP_LOOKUP_PREFETCH_END`` event.
    """
    windows = windows if windows is not None else [-1]
    num_rows = len(windows) * world_size
    miss_rows = ["0" * (len(chunk_hashes) - start_chunks)] * num_rows
    l1_rows = l1_rows if l1_rows is not None else miss_rows
    l2_rows = l2_rows if l2_rows is not None else miss_rows

    ctx = MagicMock()
    # No previous lookup to settle: ``lookup`` claims any unreleased
    # result before ``begin_lookup`` resets it.
    _sess = ctx.session_manager.get_or_create.return_value
    _sess.take_prefetch_release.return_value = None
    ctx.chunk_size = CHUNK_SIZE
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock() if layout_found else None
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=windows, world_size=world_size
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = (
        {group_id: MagicMock() for group_id in range(len(windows))}
        if group_layouts_found
        else {}
    )
    ctx.token_hasher.compute_chunk_hashes.return_value = chunk_hashes
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req-1",
        total_requested_keys=len(chunk_hashes) * num_rows,
        submit_time=time.monotonic(),
    )
    ctx.storage_manager.query_prefetch_status.return_value = _result(l1_rows, l2_rows)

    module = LookupModule(ctx)
    module.lookup(
        _lookup_key(world_size, start_chunks, prompt_tokens, engine_hit_tokens),
        tp_size=1,
    )
    module.query_prefetch_status("req-1")

    for call in ctx.event_bus.publish.call_args_list:
        event = call.args[0]
        if event.event_type is EventType.MP_LOOKUP_PREFETCH_END:
            return event.metadata
    raise AssertionError("no MP_LOOKUP_PREFETCH_END event was published")


@pytest.mark.parametrize(
    (
        "windows",
        "world_size",
        "l1_rows",
        "l2_rows",
        "l1_chunks",
        "l2_chunks",
        "l1_keys",
        "l2_keys",
    ),
    [
        ([-1], 1, ["1111"], ["0000"], 4, 0, 4, 0),
        ([-1], 1, ["0000"], ["1111"], 0, 4, 0, 4),
        ([-1], 1, ["1100"], ["0011"], 2, 2, 2, 2),
        ([-1], 1, ["0000"], ["0000"], 0, 0, 0, 0),
        # L2 fills chunk 1: without it L1 serves only chunk 0.
        ([-1], 1, ["1011"], ["0100"], 1, 3, 3, 1),
        # Two kv ranks: one chunk spans two keys.
        ([-1], 2, ["111", "111"], ["000", "000"], 3, 0, 6, 0),
        # Sliding-window rows keep only their trailing chunk, all in L1.
        (
            [-1, 1],
            2,
            ["1111", "1111", "0001", "0001"],
            ["0000", "0000", "0000", "0000"],
            4,
            0,
            10,
            0,
        ),
        # L1 holds the full-attention row, L2 the window row: no chunk is
        # servable from L1 alone, yet most keys came from L1.
        ([-1, 1], 1, ["1111", "0000"], ["0000", "0001"], 0, 4, 4, 1),
    ],
    ids=[
        "all_l1",
        "all_l2",
        "l2_extends_l1",
        "cold_miss",
        "l2_fills_hole",
        "two_kv_ranks",
        "sliding_window_all_l1",
        "l2_serves_sliding_window",
    ],
)
def test_end_event_splits_hits_by_tier(
    windows,
    world_size,
    l1_rows,
    l2_rows,
    l1_chunks,
    l2_chunks,
    l1_keys,
    l2_keys,
):
    """L1 is credited with the prefix it serves alone, L2 with the rest;
    keys are counted per tier."""
    meta = _end_metadata(
        chunk_hashes=[f"c{i}".encode() for i in range(len(l1_rows[0]))],
        windows=windows,
        world_size=world_size,
        l1_rows=l1_rows,
        l2_rows=l2_rows,
    )

    assert meta["l1_hit_tokens"] == l1_chunks * CHUNK_SIZE
    assert meta["l2_hit_tokens"] == l2_chunks * CHUNK_SIZE
    assert meta["l1_hit_tokens"] + meta["l2_hit_tokens"] == meta["hit_tokens"]
    assert meta["l1_hit_keys"] == l1_keys
    assert meta["l2_hit_keys"] == l2_keys
    assert meta["early_exit_reason"] == ""


@pytest.mark.parametrize(
    ("l1_row", "l2_row", "l1_chunks", "l2_chunks"),
    [
        ("11", "00", 2, 0),
        ("00", "11", 0, 2),
        ("10", "01", 1, 1),
        ("00", "00", 0, 0),
    ],
    ids=["all_l1", "all_l2", "l2_extends_l1", "cold_miss"],
)
def test_engine_covered_prefix_is_not_an_l1_hit(l1_row, l2_row, l1_chunks, l2_chunks):
    """A prefix skipped via ``key.start`` is never looked up, so it is kept
    out of the hit-rate fields and reported as ``apc_covered_tokens``; only
    ``found_count`` stays absolute for the engine."""
    meta = _end_metadata(
        chunk_hashes=[f"c{i}".encode() for i in range(5)],
        l1_rows=[l1_row],
        l2_rows=[l2_row],
        start_chunks=3,
    )

    assert meta["apc_covered_tokens"] == 3 * CHUNK_SIZE
    assert meta["requested_tokens"] == 2 * CHUNK_SIZE
    assert meta["l1_hit_tokens"] == l1_chunks * CHUNK_SIZE
    assert meta["l2_hit_tokens"] == l2_chunks * CHUNK_SIZE
    assert meta["l1_hit_tokens"] + meta["l2_hit_tokens"] == meta["hit_tokens"]
    assert meta["found_count"] == 3 + l1_chunks + l2_chunks


def test_fully_engine_covered_lookup_reports_no_hit_tokens():
    """When the engine covers every chunk nothing is looked up, so nothing
    is requested or hit; the engine still sees the covered prefix."""
    meta = _end_metadata(
        chunk_hashes=[f"c{i}".encode() for i in range(4)], start_chunks=4
    )

    assert meta["apc_covered_tokens"] == 4 * CHUNK_SIZE
    assert meta["requested_tokens"] == 0
    assert meta["hit_tokens"] == 0
    assert meta["l1_hit_tokens"] == 0
    assert meta["found_count"] == 4


_PROMPT_TOKENS = 5 * CHUNK_SIZE + 100  # sub-chunk tail: counted, never hit


@pytest.mark.parametrize(
    ("l1_row", "engine_hit_tokens", "request_hit_tokens"),
    [
        # LMCache extends the engine's prefix: covered (3 chunks) + hit (1).
        ("10", 3 * CHUNK_SIZE + 50, 4 * CHUNK_SIZE),
        # Cold miss past the aligned start: the raw engine hit is longer.
        ("00", 3 * CHUNK_SIZE + 50, 3 * CHUNK_SIZE + 50),
    ],
    ids=["lmcache_extends_engine", "engine_hit_only"],
)
def test_request_level_fields_cover_the_whole_prompt(
    l1_row, engine_hit_tokens, request_hit_tokens
):
    """``prompt_tokens`` is the whole prompt, ``engine_hit_tokens`` the
    connector's raw hint, and ``request_hit_tokens`` the longer of the engine
    hit and the covered prefix plus LMCache's hit, never their sum."""
    meta = _end_metadata(
        chunk_hashes=[f"c{i}".encode() for i in range(5)],
        l1_rows=[l1_row],
        start_chunks=3,
        prompt_tokens=_PROMPT_TOKENS,
        engine_hit_tokens=engine_hit_tokens,
    )

    assert meta["prompt_tokens"] == _PROMPT_TOKENS
    assert meta["engine_hit_tokens"] == engine_hit_tokens
    assert meta["request_hit_tokens"] == request_hit_tokens
    # The lookup-level fields are untouched by the hint.
    assert meta["apc_covered_tokens"] == 3 * CHUNK_SIZE
    assert meta["requested_tokens"] == 2 * CHUNK_SIZE


def test_engine_hit_is_zero_without_hint():
    """With no connector hint the engine hit is unknown and reads 0; the
    request-level hit still covers the skipped prefix plus LMCache's hit."""
    meta = _end_metadata(
        chunk_hashes=[f"c{i}".encode() for i in range(5)],
        l1_rows=["10"],
        start_chunks=3,
        prompt_tokens=_PROMPT_TOKENS,
    )

    assert meta["engine_hit_tokens"] == 0
    assert meta["request_hit_tokens"] == 4 * CHUNK_SIZE


def test_early_exit_still_reports_request_level_fields():
    """An early exit looks nothing up, but the request still has a prompt and
    an engine hit, so the request-level fields describe it anyway."""
    meta = _end_metadata(
        chunk_hashes=[b"c0"],
        layout_found=False,
        prompt_tokens=_PROMPT_TOKENS,
        engine_hit_tokens=500,
    )

    assert meta["early_exit_reason"] == "no_gpu_context"
    assert meta["prompt_tokens"] == _PROMPT_TOKENS
    assert meta["engine_hit_tokens"] == 500
    assert meta["request_hit_tokens"] == 500
    assert meta["hit_tokens"] == 0


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
    assert meta["apc_covered_tokens"] == 0
    assert meta["hit_tokens"] == 0
    # No hint and nothing covered: the engine served nothing we know of.
    assert meta["prompt_tokens"] == 1
    assert meta["engine_hit_tokens"] == 0
    assert meta["request_hit_tokens"] == 0
    assert meta["l1_hit_keys"] == 0
    assert meta["l2_hit_keys"] == 0
