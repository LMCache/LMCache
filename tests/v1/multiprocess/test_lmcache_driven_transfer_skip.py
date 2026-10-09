# SPDX-License-Identifier: Apache-2.0
"""Tests for the store-skip / retrieve-window logic in
``lmcache_driven_transfer``.

- ``incomplete_chunk_masks`` (store side): reject objects missing retained blocks.
- ``retrieve`` (read side): read/transfer only each object group's in-window
  suffix, None-padding the skipped prefix so the transfer path is unchanged.
"""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager, ObjectGroupInfo
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
    incomplete_chunk_masks,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
)
import lmcache.lmcache_native as lmcache_native

# ------------------------------------------------------------------ #
#  incomplete_chunk_masks (store-side skip)                           #
# ------------------------------------------------------------------ #


def _og(kernel_group_indices):
    return ObjectGroupInfo(kernel_group_indices=list(kernel_group_indices))


def test_full_attention_group_never_null():
    # One real block per chunk -> nothing skipped.
    masks = incomplete_chunk_masks(
        block_ids=[[1, 2, 3]],
        object_groups=[_og([0])],
        blocks_per_chunk=[1],
        num_chunks=3,
    )
    assert masks == [[False, False, False]]


def test_mamba_group_one_block_per_chunk_marks_null_prefix():
    # Align-mamba: only the last block is real; earlier chunks are the null
    # block (id 0) and must be marked skippable.
    masks = incomplete_chunk_masks(
        block_ids=[[0, 0, 0, 7]],
        object_groups=[_og([0])],
        blocks_per_chunk=[1],
        num_chunks=4,
    )
    assert masks == [[True, True, True, False]]


def test_multi_block_per_chunk_requires_every_retained_block():
    masks = incomplete_chunk_masks(
        block_ids=[[0, 0, 0, 9]],
        object_groups=[_og([0])],
        blocks_per_chunk=[2],
        num_chunks=2,
    )
    assert masks == [[True, True]]


def test_two_object_groups_independent():
    # Group 0 = full attention (kernel group 0, all real); group 1 = mamba
    # (kernel group 1, null prefix). Masks are per object group.
    masks = incomplete_chunk_masks(
        block_ids=[[1, 2, 3], [0, 0, 5]],
        object_groups=[_og([0]), _og([1])],
        blocks_per_chunk=[1, 1],
        num_chunks=3,
    )
    assert masks == [[False, False, False], [True, True, False]]


def test_object_group_requires_every_kernel_group():
    masks = incomplete_chunk_masks(
        block_ids=[[0, 0], [0, 4]],
        object_groups=[_og([0, 1])],
        blocks_per_chunk=[1, 1],
        num_chunks=2,
    )
    assert masks == [[True, True]]


def test_zero_is_real_with_negative_null_block() -> None:
    masks = incomplete_chunk_masks([[0, 0]], [_og([0])], [1], 2, -1)
    assert masks == [[False, False]]


def test_negative_null_marker_preserves_checkpoint_zero() -> None:
    masks = incomplete_chunk_masks([[-1, 0, -1, 1]], [_og([0])], [1], 4, -1)
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


def _make_module(monkeypatch, num_chunks, num_chunks_in_sw, group_kinds=()):
    """Build an LMCacheDrivenTransferModule with its collaborators mocked, and
    return (module, read_calls, transfer_calls) capturing what retrieve reads
    and transfers per object group."""
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
    obj_keys = [
        [f"g{g}c{c}" for c in range(num_chunks)] for g in range(num_object_groups)
    ]
    ctx = MagicMock()
    ctx.get_read_owners.return_value = None
    ctx.chunk_size = 256
    ctx.null_block_id = 0
    ctx.resolve_obj_keys.return_value = obj_keys

    read_calls: list[list[str]] = []

    def fake_read(keys, l1_owners=None):
        read_calls.append(list(keys))
        return list(keys), [
            MagicMock(get_size=MagicMock(return_value=10)) for _ in keys
        ]

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
    merged: bool,
) -> tuple[LMCacheDrivenTransferModule, MagicMock, list, list]:
    windows = [-1] if merged else [-1, 1]
    groups = [_og([0, 1, 2])] if merged else [_og([0]), _og([1, 2])]
    module, reads, transfers = _make_module(monkeypatch, 2, windows)
    context = _staging_context(3, groups)
    context.kv_layer_groups_manager.num_object_groups = len(groups)
    context.kv_layer_groups_manager.get_attn_desc = lambda: SimpleNamespace(
        num_chunks_in_sw=windows,
        group_kinds=("attention",) if merged else ("attention", "recurrent"),
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


@pytest.mark.parametrize("merged", [False, True])
def test_store_reserves_real_page_zero_and_only_present_state_objects(
    monkeypatch: pytest.MonkeyPatch,
    merged: bool,
) -> None:
    # Align-mode recurrent state exists only in the final chunk. In merged
    # mode, that also prevents storing its attention-only prefix.
    module, context, _reads, transfers = _make_checkpoint_module(monkeypatch, merged)
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
    ] == ([["g0c1"]] if merged else [["g0c0", "g0c1"], ["g1c1"]])
    state_group = 0 if merged else 1
    assert [obj is None for obj in transfers[state_group][1]] == [True, False]
    assert context.stage_block_ids.call_args.args[0] == [
        [0, 1, 2, 3],
        [-1, -1, 0, 1],
        [-1, -1, 2, 3],
    ]


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize(
    "separate_object_groups,full_sw_kv,null_block_id",
    [
        pytest.param(False, False, 0, id="merged"),
        pytest.param(True, False, 0, id="separate"),
        pytest.param(False, True, 0, id="full-window"),
        pytest.param(True, False, -1, id="negative-null-marker"),
    ],
)
def test_store_requires_every_retained_block_in_hybrid_object(
    monkeypatch: pytest.MonkeyPatch,
    separate_object_groups: bool,
    full_sw_kv: bool,
    null_block_id: int,
) -> None:
    """A warm prefix can retain full-attention pages after SW pages are gone."""
    # Logical block sizes differ from physical slots in the compressed group.
    tokens_per_block = [64, 64, 256, 256, 256, 4, 8]
    slots_per_block = [64, 64, 64, 64, 2, 4, 8]
    windows = [128, 128, -1, -1, -1, 8, 128]
    engine_groups = [0, 1, 2, 2, 2, 3, 4]
    tensors = [
        torch.empty(
            8, slots, index + 1, dtype=torch.uint8 if index < 5 else torch.float32
        )
        for index, slots in enumerate(slots_per_block)
    ]
    manager = KVLayerGroupsManager(
        tensors,
        [lmcache_native.EngineKVFormat.NL_X_NB_BS_HS] * len(tensors),
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=engine_groups[index],
                layer_indices=(index,),
                tokens_per_block=tpb,
                sw_size_tokens=windows[index],
            )
            for index, tpb in enumerate(tokens_per_block)
        ],
        lmcache_tokens_per_chunk=256,
        separate_object_groups=separate_object_groups,
    )
    if full_sw_kv:
        manager.enable_full_sw_kv()
    module, _, transfers = _make_module(
        monkeypatch, 4, manager.get_attn_desc().num_chunks_in_sw
    )
    context = module.get_and_touch_context_entry(1).cache_context
    context.kv_layer_groups_manager = manager
    context.lmcache_tokens_per_chunk = 256
    context.calculate_num_blocks.side_effect = lambda tokens, group: (
        manager.calculate_num_blocks(group, tokens)
    )
    context.stage_block_ids.side_effect = lambda ids: ids
    module.context.null_block_id = null_block_id
    module.context.storage_manager.reserve_write.side_effect = lambda keys, layout: {
        key: MagicMock(get_size=MagicMock(return_value=10)) for key in keys
    }
    monkeypatch.setattr(
        mod, "downsample_and_stage_block_ids", downsample_and_stage_block_ids
    )
    monkeypatch.setattr(mod, "get_layout_desc", lambda *args, **kwargs: object())

    real_block = 1 if null_block_id == 0 else 0
    block_ids: list[list[int]] = []
    for tpb, window in zip(tokens_per_block, windows, strict=True):
        bpc = 256 // tpb
        if window == -1:
            block_ids.append([real_block] * (4 * bpc))
            continue
        keep = window // tpb
        # Chunk 0: every SW group absent. Chunk 1: only the narrow SW absent.
        # Chunk 2: one retained narrow-SW block absent. Chunk 3: only the
        # discarded prefix is absent, so its retained window is valid.
        chunks = [[null_block_id] * bpc, [real_block] * bpc, [real_block] * bpc]
        if window == 8:
            chunks[1] = [null_block_id] * bpc
            chunks[2][-keep] = null_block_id
        chunks.append([null_block_id] * (bpc - keep) + [real_block] * keep)
        block_ids.append([block for chunk in chunks for block in chunk])

    _, ok, stored = module.store_with_chunk_mask(
        SimpleNamespace(request_id="req", worker_id=1), 1, block_ids, b"producer"
    )

    assert ok
    assert stored == [False, False, False, not full_sw_kv]
    for group_id, group in enumerate(manager.object_groups):
        has_sw = any(windows[index] != -1 for index in group.kernel_group_indices)
        expected = ([] if full_sw_kv else [3]) if has_sw else list(range(4))
        assert [
            i for i, obj in enumerate(transfers[group_id][1]) if obj is not None
        ] == expected


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
