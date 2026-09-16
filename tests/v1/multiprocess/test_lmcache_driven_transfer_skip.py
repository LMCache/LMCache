# SPDX-License-Identifier: Apache-2.0
"""Tests for the mamba store-skip / retrieve-window logic in
``lmcache_driven_transfer``.

- ``all_null_chunk_masks`` (store side): mark chunks whose block ids are all the
  null block so ``store`` never commits them.
- ``retrieve`` (read side): read/transfer only each object group's in-window
  suffix, None-padding the skipped prefix so the transfer path is unchanged.
"""

# Standard
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
    all_null_chunk_masks,
    downsample_and_stage_block_ids,
)

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


def test_zero_is_real_without_a_null_block() -> None:
    masks = all_null_chunk_masks([[0, 0]], [_og([0])], [1], 2, [None])
    assert masks == [[False, False]]


def test_negative_null_marker_preserves_checkpoint_zero() -> None:
    masks = all_null_chunk_masks([[-1, 0, -1, 1]], [_og([0])], [1], 4, [-1])
    assert masks == [[True, False, True, False]]


def test_null_markers_are_per_kernel_group() -> None:
    masks = all_null_chunk_masks(
        [[0, 0, 0], [-1, -1, 0], [-1, 2, 3], [0, 0, 4]],
        [_og([0]), _og([1, 2]), _og([3])],
        [1, 1, 1, 1],
        3,
        [None, -1, -1, 0],
    )
    assert masks == [
        [False, False, False],
        [True, False, False],
        [True, True, False],
    ]


def _staging_context(
    null_ids: list[int | None],
    object_groups: list[ObjectGroupInfo],
    *,
    window_tokens: int = 2,
) -> MagicMock:
    """Build a CPU-only context that captures staged block IDs."""
    context = MagicMock()
    context.lmcache_tokens_per_chunk = 2
    context.calculate_num_blocks.side_effect = lambda tokens, group: tokens
    context.kv_layer_groups_manager = SimpleNamespace(
        num_kernel_groups=len(null_ids),
        kernel_groups=[SimpleNamespace(null_block_id=null_id) for null_id in null_ids],
        object_groups=object_groups,
        get_subchunk_sw_size_tokens=lambda group: window_tokens,
    )
    context.stage_block_ids.side_effect = lambda ids: ids
    return context


def test_absent_checkpoint_objects_stage_safe_placeholders() -> None:
    context = _staging_context([None, -1, -1], [_og([0]), _og([1, 2])])
    block_ids = [[0, 1, 2, 3], [-1, -1, 0, 1], [-1, -1, 2, 3]]
    masks = all_null_chunk_masks(
        block_ids,
        context.kv_layer_groups_manager.object_groups,
        [2] * 3,
        2,
        [None, -1, -1],
    )
    staged = downsample_and_stage_block_ids(context, block_ids, skipped_chunks=masks)
    assert staged == [[0, 1, 2, 3], [0, 0, 0, 1], [0, 0, 2, 3]]


def test_legacy_sliding_window_zero_placeholders_remain_valid() -> None:
    context = _staging_context([0], [_og([0])])
    assert downsample_and_stage_block_ids(context, [[0, 3]]) == [[0, 3]]


def test_invalid_blocks_outside_copy_window_are_not_staged() -> None:
    context = _staging_context([None], [_og([0])], window_tokens=1)
    assert downsample_and_stage_block_ids(context, [[-1, 0, -1, 3]]) == [[0, 3]]


def test_retrieve_skipped_prefix_accepts_absent_checkpoints() -> None:
    context = _staging_context([-1], [_og([0])])
    assert downsample_and_stage_block_ids(
        context,
        [[-1, -1, -1, 0]],
        skipped_chunks=[[True, False]],
        skip_first_n_tokens=3,
    ) == [[0, 0, 0, 0]]


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
    ctx.chunk_size = 256
    ctx.resolve_obj_keys.return_value = obj_keys

    read_calls: list[list[str]] = []

    @contextmanager
    def fake_read(keys):
        read_calls.append(list(keys))
        yield [MagicMock(get_size=MagicMock(return_value=10)) for _ in keys]

    ctx.storage_manager.read_prefetched_results = MagicMock(side_effect=fake_read)
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
    ):
        transfer_calls.append((object_group_id, list(memory_objs)))

    monkeypatch.setattr(mod, "transfer_kv_per_object_group", fake_transfer)
    monkeypatch.setattr(mod, "downsample_and_stage_block_ids", lambda cc, b, **kw: b)
    monkeypatch.setattr(mod, "submit_callback_to_stream", lambda *a, **k: None)
    monkeypatch.setattr(mod, "torch_dev", MagicMock())
    monkeypatch.setattr(mod, "Event", MagicMock())

    return module, read_calls, transfer_calls


def _make_checkpoint_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LMCacheDrivenTransferModule, MagicMock, list, list]:
    module, reads, transfers = _make_module(monkeypatch, 2, [-1, 1])
    context = _staging_context([None, -1, -1], [_og([0]), _og([1, 2])])
    context.kv_layer_groups_manager.num_object_groups = 2
    context.kv_layer_groups_manager.get_attn_desc = lambda: SimpleNamespace(
        num_chunks_in_sw=[-1, 1], group_kinds=("attention", "recurrent")
    )
    module.get_and_touch_context_entry(1).cache_context = context
    module.context.chunk_size = 2
    module.context.session_manager.get.return_value = None
    module.context.storage_manager.reserve_write.side_effect = (
        lambda keys, layout, mode: {
            key: MagicMock(get_size=MagicMock(return_value=10)) for key in keys
        }
    )
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
        [0, 0, 0, 1],
        [0, 0, 2, 3],
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
