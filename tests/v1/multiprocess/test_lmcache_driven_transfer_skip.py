# SPDX-License-Identifier: Apache-2.0
"""Tests for the mamba store-skip / retrieve-window logic in
``lmcache_driven_transfer``.

- ``all_null_chunk_masks`` (store side): mark chunks whose block ids are all the
  null block so ``store`` never commits them.
- ``retrieve`` (read side): read/transfer only each object group's in-window
  suffix, None-padding the skipped prefix so the transfer path is unchanged.
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
    all_null_chunk_masks,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
)
from lmcache.v1.multiprocess.session import PrefetchTierSplit, Session

# ------------------------------------------------------------------ #
#  all_null_chunk_masks (store-side skip)                              #
# ------------------------------------------------------------------ #


def _og(kernel_group_indices):
    return ObjectGroupInfo(kernel_group_indices=list(kernel_group_indices))


def test_full_attention_group_never_null():
    # One real block per chunk -> nothing skipped.
    masks = all_null_chunk_masks(
        block_ids=[[1, 2, 3]],
        object_groups=[_og([0])],
        blocks_per_chunk=[1],
        num_chunks=3,
    )
    assert masks == [[False, False, False]]


def test_mamba_group_one_block_per_chunk_marks_null_prefix():
    # Align-mamba: only the last block is real; earlier chunks are the null
    # block (id 0) and must be marked skippable.
    masks = all_null_chunk_masks(
        block_ids=[[0, 0, 0, 7]],
        object_groups=[_og([0])],
        blocks_per_chunk=[1],
        num_chunks=4,
    )
    assert masks == [[True, True, True, False]]


def test_multi_block_per_chunk_null_only_when_all_blocks_zero():
    # chunk size = 2 blocks. Chunk 0 = [0, 0] (null), chunk 1 = [0, 9] (has a
    # real block in its second slot) -> not null.
    masks = all_null_chunk_masks(
        block_ids=[[0, 0, 0, 9]],
        object_groups=[_og([0])],
        blocks_per_chunk=[2],
        num_chunks=2,
    )
    assert masks == [[True, False]]


def test_two_object_groups_independent():
    # Group 0 = full attention (kernel group 0, all real); group 1 = mamba
    # (kernel group 1, null prefix). Masks are per object group.
    masks = all_null_chunk_masks(
        block_ids=[[1, 2, 3], [0, 0, 5]],
        object_groups=[_og([0]), _og([1])],
        blocks_per_chunk=[1, 1],
        num_chunks=3,
    )
    assert masks == [[False, False, False], [True, True, False]]


def test_object_group_null_only_when_all_its_kernel_groups_null():
    # An object group spanning two kernel groups: a chunk is null only if every
    # kernel group's blocks for that chunk are null.
    masks = all_null_chunk_masks(
        block_ids=[[0, 0], [0, 4]],
        object_groups=[_og([0, 1])],
        blocks_per_chunk=[1, 1],
        num_chunks=2,
    )
    # chunk 0: kg0=0 and kg1=0 -> null; chunk 1: kg0=0 but kg1=4 -> not null.
    assert masks == [[True, False]]


def test_zero_is_real_with_negative_null_block() -> None:
    masks = all_null_chunk_masks([[0, 0]], [_og([0])], [1], 2, -1)
    assert masks == [[False, False]]


def test_negative_null_marker_preserves_checkpoint_zero() -> None:
    masks = all_null_chunk_masks([[-1, 0, -1, 1]], [_og([0])], [1], 4, -1)
    assert masks == [[True, False, True, False]]


def _staging_context(
    num_kernel_groups: int,
    object_groups: list[ObjectGroupInfo],
    *,
    window_tokens: int = 2,
) -> MagicMock:
    """Build a CPU-only context that captures staged block IDs."""
    context = MagicMock()
    context.lmcache_tokens_per_chunk = 2
    context.calculate_num_blocks.side_effect = lambda tokens, group: tokens
    context.kv_layer_groups_manager = SimpleNamespace(
        num_kernel_groups=num_kernel_groups,
        object_groups=object_groups,
        get_subchunk_sw_size_tokens=lambda group: window_tokens,
    )
    context.stage_block_ids.side_effect = lambda ids: ids
    return context


# ------------------------------------------------------------------ #
#  retrieve (read-side window)                                         #
# ------------------------------------------------------------------ #


def _make_module(
    monkeypatch,
    num_chunks,
    num_chunks_in_sw,
    group_kinds=(),
    *,
    obj_keys=None,
    session=None,
    missing_keys=(),
):
    """Build an LMCacheDrivenTransferModule with its collaborators mocked, and
    return (module, read_calls, transfer_calls) capturing what retrieve reads
    and transfers per object group.

    ``obj_keys`` overrides the per-group keys ``resolve_obj_keys`` returns,
    ``session`` is what ``session_manager.get`` returns (``None``: no session)
    and ``missing_keys`` are keys ``unsafe_read`` reports absent.
    """
    num_object_groups = len(num_chunks_in_sw)

    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)

    kvlgm = SimpleNamespace(
        num_object_groups=num_object_groups,
        num_kernel_groups=num_object_groups,
        get_attn_desc=lambda: SimpleNamespace(
            num_chunks_in_sw=num_chunks_in_sw, group_kinds=tuple(group_kinds)
        ),
    )
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager = kvlgm
    cache_context.calculate_num_blocks.return_value = 1  # 1 block per chunk
    cache_context.max_batch_size = 8

    event_backend = MagicMock()
    entry = SimpleNamespace(
        cache_context=cache_context, model_name="m", event_backend=event_backend
    )
    module.get_and_touch_context_entry = MagicMock(return_value=entry)

    # Object keys: one distinct key per (group, chunk).
    if obj_keys is None:
        obj_keys = [
            [f"g{g}c{c}" for c in range(num_chunks)] for g in range(num_object_groups)
        ]
    ctx = MagicMock()
    ctx.get_read_owners.return_value = None
    ctx.session_manager.get.return_value = session
    ctx.chunk_size = 256
    ctx.null_block_id = 0
    ctx.resolve_obj_keys.return_value = obj_keys

    read_calls: list[list] = []

    def fake_read(keys, l1_owners=None):
        read_calls.append(list(keys))
        found = [key for key in keys if key not in missing_keys]
        return found, [MagicMock(get_size=MagicMock(return_value=10)) for _ in found]

    ctx.storage_manager.unsafe_read = MagicMock(side_effect=fake_read)
    module._ctx = ctx

    transfer_calls: list[tuple[int, list]] = []

    def fake_transfer(
        cache_context,
        block_ids,
        memory_objs,
        object_group_id,
        batch_size,
        skip_first_n_tokens,
        direction,
        *,
        transfer_key,
        block_ids_host=(),
    ):
        transfer_calls.append((object_group_id, list(memory_objs)))

    monkeypatch.setattr(mod, "transfer_kv_per_object_group", fake_transfer)
    monkeypatch.setattr(mod, "downsample_and_stage_block_ids", lambda cc, b: b)
    monkeypatch.setattr(mod, "submit_callback_to_stream", lambda *a, **k: None)
    monkeypatch.setattr(mod, "torch_dev", MagicMock())
    monkeypatch.setattr(mod, "Event", MagicMock())

    return module, read_calls, transfer_calls


def _make_checkpoint_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LMCacheDrivenTransferModule, MagicMock, list, list]:
    module, reads, transfers = _make_module(monkeypatch, 2, [-1, 1])
    context = _staging_context(3, [_og([0]), _og([1, 2])])
    context.kv_layer_groups_manager.num_object_groups = 2
    context.kv_layer_groups_manager.get_attn_desc = lambda: SimpleNamespace(
        num_chunks_in_sw=[-1, 1], group_kinds=("attention", "recurrent")
    )
    module.get_and_touch_context_entry(1).cache_context = context
    module.context.chunk_size = 2
    module.context.null_block_id = -1
    module.context.session_manager.get.return_value = None
    module.context.storage_manager.reserve_write.side_effect = lambda keys, layout: {
        key: MagicMock(get_size=MagicMock(return_value=10)) for key in keys
    }
    monkeypatch.setattr(
        mod, "downsample_and_stage_block_ids", downsample_and_stage_block_ids
    )
    monkeypatch.setattr(mod, "get_layout_desc", lambda *a, **kw: object())
    return module, context, reads, transfers


def test_store_reserves_real_page_zero_and_only_present_state_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, context, _reads, transfers = _make_checkpoint_module(monkeypatch)
    _handle, ok = module.store(
        SimpleNamespace(request_id="req", worker_id=1),
        1,
        [[0, 1, 2, 3], [-1, -1, 0, 1], [-1, -1, 2, 3]],
        b"producer",
    )
    assert ok
    assert [
        call.args[0]
        for call in cast(
            MagicMock, module.context.storage_manager.reserve_write
        ).call_args_list
    ] == [["g0c0", "g0c1"], ["g1c1"]]
    assert [obj is None for obj in transfers[1][1]] == [True, False]
    assert context.stage_block_ids.call_args.args[0] == [
        [0, 1, 2, 3],
        [-1, -1, 0, 1],
        [-1, -1, 2, 3],
    ]


def test_retrieve_reads_and_transfers_only_in_window(monkeypatch):
    # Group 0 = full attention (-1): whole prefix; group 1 = mamba window 1:
    # only the last chunk.
    num_chunks = 5
    module, read_calls, transfer_calls = _make_module(
        monkeypatch, num_chunks, num_chunks_in_sw=[-1, 1]
    )
    # 1 block per chunk -> block-id lists of length num_chunks (avoid underflow).
    gpu_block_ids = [[1, 2, 3, 4, 5], [0, 0, 0, 0, 9]]

    _handle, ok = module.retrieve(
        key=SimpleNamespace(request_id="req", cache_salt="salt"),
        instance_id=1,
        gpu_block_ids=gpu_block_ids,
        event_ipc_handle=b"x",
    )
    assert ok is True

    # Full-attention group reads all 5 keys; mamba group reads only the last.
    assert read_calls[0] == [f"g0c{c}" for c in range(5)]
    assert read_calls[1] == ["g1c4"]

    # memory_objs handed to the transfer stay full-length; the mamba group's
    # skipped prefix is None-padded (skip = num_chunks - window = 4).
    grp0, mem0 = transfer_calls[0]
    grp1, mem1 = transfer_calls[1]
    assert grp0 == 0 and len(mem0) == 5 and all(o is not None for o in mem0)
    assert grp1 == 1 and len(mem1) == 5
    assert [o is None for o in mem1] == [True, True, True, True, False]


def test_retrieve_full_attention_only_reads_everything(monkeypatch):
    # No sliding-window group: behavior is unchanged (read all, no None-pad).
    num_chunks = 3
    module, read_calls, transfer_calls = _make_module(
        monkeypatch, num_chunks, num_chunks_in_sw=[-1]
    )
    _handle, ok = module.retrieve(
        key=SimpleNamespace(request_id="req", cache_salt="salt"),
        instance_id=1,
        gpu_block_ids=[[1, 2, 3]],
        event_ipc_handle=b"x",
    )
    assert ok is True
    assert read_calls == [["g0c0", "g0c1", "g0c2"]]
    _grp, mem = transfer_calls[0]
    assert len(mem) == 3 and all(o is not None for o in mem)


def test_retrieve_never_reads_aux_groups(monkeypatch):
    """The std retrieve skips connector-private aux object groups.

    Their consumer is the CB retrieve, the op's block-id entry for them is a
    discard placeholder, and the lookup does not lock their keys -- reading
    them here would be an unlocked read of a plane nobody consumes.
    """
    num_chunks = 3
    module, read_calls, transfer_calls = _make_module(
        monkeypatch,
        num_chunks,
        num_chunks_in_sw=[1, -1, -1],
        group_kinds=("recurrent", "attention", "aux"),
    )
    gpu_block_ids = [[0, 0, 7], [1, 2, 3], [9, 9, 9]]

    _handle, ok = module.retrieve(
        key=SimpleNamespace(request_id="req", cache_salt="salt"),
        instance_id=1,
        gpu_block_ids=gpu_block_ids,
        event_ipc_handle=b"x",
    )
    assert ok is True

    # Recurrent group reads its one-block window; attention reads everything;
    # the aux group is read by NOBODY and transferred by nobody.
    assert read_calls == [["g0c2"], [f"g1c{c}" for c in range(3)]]
    assert [g for g, _ in transfer_calls] == [0, 1]


def test_failed_copy_releases_all_retained_owners_on_stream(monkeypatch):
    module, reads, _ = _make_module(monkeypatch, 2, [-1, 1])
    cache_context = module.get_and_touch_context_entry(1).cache_context
    cache_context.hold_imported_event.return_value = 7
    owners = {"g0c0": 10, "g0c1": 10, "g1c1": 20}
    completion = [(10, ["g0c0", "g0c1"]), (20, ["g1c1"])]
    module.context.get_read_owners.return_value = owners
    module.context.storage_manager.prepare_read_completion.return_value = completion
    callback = MagicMock()
    monkeypatch.setattr(mod, "submit_callback_to_stream", callback)
    monkeypatch.setattr(
        mod,
        "transfer_kv_per_object_group",
        MagicMock(side_effect=RuntimeError("partly enqueued transfer")),
    )
    _, ok = module.retrieve(
        SimpleNamespace(request_id="req", cache_salt="salt"),
        1,
        [[1, 2], [0, 3]],
        b"producer",
    )
    assert not ok
    assert reads == [["g0c0", "g0c1"]]
    module.context.storage_manager.prepare_read_completion.assert_called_once_with(
        ["g0c0", "g0c1", "g1c1"], owners
    )
    assert callback.call_args_list == [
        call(cache_context.cupy_stream, "release_imported_event", (1, 7)),
        call(cache_context.cupy_stream, "finish_read_by_owner", completion),
    ]
    module.context.storage_manager.finish_read_prefetched.assert_not_called()


# ------------------------------------------------------------------ #
#  downsample_and_stage_block_ids (DSv4 sub-chunk SWA)
# ------------------------------------------------------------------ #


def _dsv4_swa_cache_context(chunk_tokens: int, sw_tokens: int, tpb: int):
    """Fake context: slots_per_block == tpb so calculate_num_blocks = tokens/tpb."""

    def calculate_num_blocks(num_tokens: int, kernel_group_idx: int) -> int:
        del kernel_group_idx
        return num_tokens // tpb

    kgm = SimpleNamespace(
        num_kernel_groups=1,
        get_subchunk_sw_size_tokens=lambda kg: sw_tokens,
    )
    ctx = SimpleNamespace(
        kv_layer_groups_manager=kgm,
        lmcache_tokens_per_chunk=chunk_tokens,
        calculate_num_blocks=calculate_num_blocks,
        stage_block_ids=lambda ids: ids,
    )
    return ctx


@pytest.mark.no_shared_allocator
def test_downsample_keeps_last_window_of_each_chunk_dsv4_swa():
    """DSv4 SWA: chunk 4096, window 128, tpb 32 → keep last 4 block ids / chunk."""
    chunk, sw, tpb, n_chunks = 4096, 128, 32, 19
    ctx = _dsv4_swa_cache_context(chunk, sw, tpb)
    bpc = chunk // tpb  # 128
    keep = sw // tpb  # 4
    original = list(range(n_chunks * bpc))
    out = mod.downsample_and_stage_block_ids(ctx, [list(original)])
    assert len(out[0]) == n_chunks * keep
    for c in range(n_chunks):
        src = original[c * bpc : (c + 1) * bpc]
        got = out[0][c * keep : (c + 1) * keep]
        assert got == src[-keep:]
    # Retrieve of the last object uses start_object_idx = n_chunks-1.
    start = (n_chunks - 1) * keep
    assert out[0][start:] == original[-keep:]


@pytest.mark.no_shared_allocator
def test_downsample_full_attention_keeps_every_block():
    chunk, tpb, n_chunks = 4096, 128, 19
    ctx = _dsv4_swa_cache_context(chunk, sw_tokens=chunk, tpb=tpb)
    bpc = chunk // tpb
    raw = [list(range(n_chunks * bpc))]
    out = mod.downsample_and_stage_block_ids(ctx, [list(raw[0])])
    assert out[0] == raw[0]


# ------------------------------------------------------------------ #
#  retrieve (serve-side L1/L2 attribution on MP_RETRIEVE_END)        #
# ------------------------------------------------------------------ #


def _bits(pattern: str) -> Bitmap:
    """A bitmap with bit ``i`` set iff ``pattern[i] == "1"``."""
    bitmap = Bitmap(len(pattern))
    for i, bit in enumerate(pattern):
        if bit == "1":
            bitmap.set(i)
    return bitmap


def _real_obj_keys(num_chunks: int, num_groups: int, kv_rank: int):
    """Per-group ObjectKeys for the retrieve range, one per chunk."""
    return [
        [
            ObjectKey(
                chunk_hash=f"c{c}".encode(),
                model_name="m",
                kv_rank=kv_rank,
                object_group_id=g,
                cache_salt="salt",
            )
            for c in range(num_chunks)
        ]
        for g in range(num_groups)
    ]


def _session_with_tier_cells(rows: dict, start_chunk: int = 0) -> Session:
    """A request session whose lookup recorded ``(l1, l2)`` row patterns keyed
    by ``(object_group_id, kv_rank)``; bit ``j`` is chunk ``start_chunk + j``."""
    session = Session(request_id="req", hasher=MagicMock())
    session.record_prefetch_tier_split(
        PrefetchTierSplit(
            start_chunk=start_chunk,
            rows={row_id: (_bits(l1), _bits(l2)) for row_id, (l1, l2) in rows.items()},
        )
    )
    return session


def _by_group(meta: dict, field: str) -> list[int]:
    """Decode a comma-separated per-object-group count field."""
    return [int(x) for x in meta[field].split(",")]


def _retrieve_end_metadata() -> dict:
    """Metadata of the MP_RETRIEVE_END event handed to the (mocked) Event."""
    # ``mod.Event`` is a MagicMock here (see ``_make_module``); tell mypy so.
    event_mock: Any = mod.Event
    for event_call in event_mock.call_args_list:
        if event_call.kwargs.get("event_type") is EventType.MP_RETRIEVE_END:
            return event_call.kwargs["metadata"]
    raise AssertionError("no MP_RETRIEVE_END event was published")


def _retrieve_key(start_chunk: int = 0):
    return SimpleNamespace(
        request_id="req", cache_salt="salt", worker_id=0, start=start_chunk * 256
    )


@pytest.mark.parametrize(
    ("l1", "l2", "first_chunk", "num_keys", "expected"),
    [
        ("1100", "0011", 0, 4, (2, 2, 0)),
        ("1100", "0011", 2, 2, (0, 2, 0)),
        # Chunk 3 is outside the lookup's hit: unattributed, not guessed.
        ("1100", "0010", 1, 3, (1, 1, 1)),
        # A range running past the recorded rows attributes nothing there.
        ("1100", "0011", 2, 5, (0, 2, 3)),
        # A range starting below the looked-up rows: that key has no record.
        ("1100", "0011", -1, 3, (2, 0, 1)),
        ("1100", "0011", 0, 0, (0, 0, 0)),
    ],
)
def test_attribute_delivered_keys(l1, l2, first_chunk, num_keys, expected):
    rows = (_bits(l1), _bits(l2))
    assert mod.attribute_delivered_keys(rows, first_chunk, num_keys) == expected


def test_attribute_delivered_keys_without_record():
    assert mod.attribute_delivered_keys(None, 3, 4) == (0, 0, 4)


def test_retrieve_end_splits_delivered_keys_by_tier_per_group(monkeypatch):
    """Delivered keys are attributed per object group from the lookup's record."""
    kv_rank = 7
    session = _session_with_tier_cells(
        {
            # Full-attention group: chunks 0-2 were L1-resident, 3-4 came
            # from L2.
            (0, kv_rank): ("11100", "00011"),
            # Window group (window 1): only its last chunk was hit, from L2.
            (1, kv_rank): ("00000", "00001"),
        }
    )
    module, _reads, _transfers = _make_module(
        monkeypatch,
        5,
        num_chunks_in_sw=[-1, 1],
        obj_keys=_real_obj_keys(5, 2, kv_rank),
        session=session,
    )
    _handle, ok = module.retrieve(
        key=_retrieve_key(),
        instance_id=1,
        gpu_block_ids=[[1, 2, 3, 4, 5], [0, 0, 0, 0, 9]],
        event_ipc_handle=b"x",
    )
    assert ok is True

    meta = _retrieve_end_metadata()
    assert meta["retrieved_count"] == 6
    assert _by_group(meta, "l1_delivered_keys_by_group") == [3, 0]
    assert _by_group(meta, "l2_delivered_keys_by_group") == [2, 1]
    assert _by_group(meta, "unattributed_delivered_keys_by_group") == [0, 0]
    assert (meta["l1_delivered_keys"], meta["l2_delivered_keys"]) == (3, 3)
    assert meta["unattributed_delivered_keys"] == 0


def test_retrieve_end_reports_unattributed_without_lookup_record(monkeypatch):
    """No session record: every delivered key is unattributed, never guessed."""
    module, _reads, _transfers = _make_module(
        monkeypatch, 5, num_chunks_in_sw=[-1, 1], obj_keys=_real_obj_keys(5, 2, 0)
    )
    _handle, ok = module.retrieve(
        key=_retrieve_key(),
        instance_id=1,
        gpu_block_ids=[[1, 2, 3, 4, 5], [0, 0, 0, 0, 9]],
        event_ipc_handle=b"x",
    )
    assert ok is True

    meta = _retrieve_end_metadata()
    assert meta["retrieved_count"] == 6
    assert meta["unattributed_delivered_keys"] == 6
    assert _by_group(meta, "unattributed_delivered_keys_by_group") == [5, 1]
    assert _by_group(meta, "l1_delivered_keys_by_group") == [0, 0]
    assert _by_group(meta, "l2_delivered_keys_by_group") == [0, 0]


@pytest.mark.parametrize(
    ("start_chunk", "expected_l1", "expected_l2", "expected_unattributed"),
    [
        # Range [3, 5): the window group's last chunk (4) is the lookup's L2 hit.
        (3, [2, 0], [0, 1], [0, 0]),
        # Range [1, 3): the window group reads chunk 2, which the lookup never
        # hit (its window ended at chunk 4) -> unattributed, not guessed.
        (1, [2, 0], [0, 0], [0, 1]),
    ],
    ids=["range_ends_at_hit", "window_outside_hit"],
)
def test_sub_range_retrieve_uses_absolute_chunk_indices(
    monkeypatch, start_chunk, expected_l1, expected_l2, expected_unattributed
):
    """A retrieve of ``[start, end)`` maps its keys onto the lookup's rows by
    absolute chunk index, so chunked-prefill sub-ranges attribute correctly."""
    session = _session_with_tier_cells(
        {(0, 0): ("11111", "00000"), (1, 0): ("00000", "00001")}
    )
    num_chunks = 2
    module, _reads, _transfers = _make_module(
        monkeypatch,
        num_chunks,
        num_chunks_in_sw=[-1, 1],
        obj_keys=_real_obj_keys(num_chunks, 2, 0),
        session=session,
    )
    _handle, ok = module.retrieve(
        key=_retrieve_key(start_chunk),
        instance_id=1,
        gpu_block_ids=[[1, 2], [0, 9]],
        event_ipc_handle=b"x",
    )
    assert ok is True

    meta = _retrieve_end_metadata()
    assert _by_group(meta, "l1_delivered_keys_by_group") == expected_l1
    assert _by_group(meta, "l2_delivered_keys_by_group") == expected_l2
    assert (
        _by_group(meta, "unattributed_delivered_keys_by_group") == expected_unattributed
    )
    assert (
        meta["l1_delivered_keys"]
        + meta["l2_delivered_keys"]
        + meta["unattributed_delivered_keys"]
        == meta["retrieved_count"]
        == 3
    )


@pytest.mark.parametrize(
    (
        "retrieve_start",
        "num_chunks",
        "gpu_block_ids",
        "expected_l1",
        "expected_l2",
        "expected_unattributed",
        "expected_count",
    ),
    [
        # Retrieve == looked-up range [2, 5): bit 0 of each row is chunk 2.
        (2, 3, [[1, 2, 3], [0, 0, 9]], [3, 0], [0, 1], [0, 0], 4),
        # The engine's hit grew to chunk 3 before allocation: retrieve [3, 5).
        (3, 2, [[1, 2], [0, 9]], [2, 0], [0, 1], [0, 0], 3),
        # Retrieve [1, 3) reaches below the looked-up range: chunk 1 has no
        # record, and the window group's chunk 2 was not a hit.
        (1, 2, [[1, 2], [0, 9]], [1, 0], [0, 0], [1, 1], 3),
    ],
    ids=["same_range", "engine_hit_grew", "below_looked_up_range"],
)
def test_retrieve_indexes_rows_from_the_lookup_start(
    monkeypatch,
    retrieve_start,
    num_chunks,
    gpu_block_ids,
    expected_l1,
    expected_l2,
    expected_unattributed,
    expected_count,
):
    """After an engine-covered prefix the lookup's rows start at
    ``start_chunk``; the retrieve maps its absolute chunk range onto them."""
    session = _session_with_tier_cells(
        {(0, 0): ("111", "000"), (1, 0): ("000", "001")}, start_chunk=2
    )
    module, _reads, _transfers = _make_module(
        monkeypatch,
        num_chunks,
        num_chunks_in_sw=[-1, 1],
        obj_keys=_real_obj_keys(num_chunks, 2, 0),
        session=session,
    )
    _handle, ok = module.retrieve(
        key=_retrieve_key(retrieve_start),
        instance_id=1,
        gpu_block_ids=gpu_block_ids,
        event_ipc_handle=b"x",
    )
    assert ok is True

    meta = _retrieve_end_metadata()
    assert _by_group(meta, "l1_delivered_keys_by_group") == expected_l1
    assert _by_group(meta, "l2_delivered_keys_by_group") == expected_l2
    assert (
        _by_group(meta, "unattributed_delivered_keys_by_group") == expected_unattributed
    )
    assert (
        meta["l1_delivered_keys"]
        + meta["l2_delivered_keys"]
        + meta["unattributed_delivered_keys"]
        == meta["retrieved_count"]
        == expected_count
    )


def test_partial_failure_attributes_only_transferred_groups(monkeypatch):
    """A retrieve that fails on a later group counts, on both sides of the
    invariant, only the keys the earlier groups transferred."""
    keys = _real_obj_keys(3, 2, 0)
    session = _session_with_tier_cells({(0, 0): ("111", "000"), (1, 0): ("000", "001")})
    module, _reads, _transfers = _make_module(
        monkeypatch,
        3,
        num_chunks_in_sw=[-1, 1],
        obj_keys=keys,
        session=session,
        missing_keys=(keys[1][2],),
    )
    _handle, ok = module.retrieve(
        key=_retrieve_key(),
        instance_id=1,
        gpu_block_ids=[[1, 2, 3], [0, 0, 9]],
        event_ipc_handle=b"x",
    )
    assert ok is False

    meta = _retrieve_end_metadata()
    assert meta["retrieved_count"] == 3
    assert _by_group(meta, "l1_delivered_keys_by_group") == [3, 0]
    assert _by_group(meta, "l2_delivered_keys_by_group") == [0, 0]
    assert _by_group(meta, "unattributed_delivered_keys_by_group") == [0, 0]
