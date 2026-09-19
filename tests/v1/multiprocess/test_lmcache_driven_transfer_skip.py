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
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
    all_null_chunk_masks,
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


# ------------------------------------------------------------------ #
#  selected-group store atomicity                                     #
# ------------------------------------------------------------------ #


def _make_store_module(monkeypatch, *, separate_object_groups=True):
    """Build a two-group store harness and capture transfers/callbacks."""
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    kvlgm = SimpleNamespace(
        num_object_groups=2,
        num_kernel_groups=2,
        kernel_groups=[
            SimpleNamespace(engine_group_idx=0),
            SimpleNamespace(engine_group_idx=1),
        ],
        object_groups=[_og([0]), _og([1])],
    )
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager = kvlgm
    cache_context.calculate_num_blocks.return_value = 1
    entry = SimpleNamespace(
        cache_context=cache_context,
        model_name="model",
        event_backend=MagicMock(),
    )
    entry.event_backend.export_event.return_value = b"done"
    module.get_and_touch_context_entry = MagicMock(return_value=entry)

    ctx = MagicMock()
    ctx.chunk_size = 256
    ctx.separate_object_groups = separate_object_groups
    ctx.resolve_obj_keys.side_effect = lambda _key, group_ids: [
        [f"g{group_id}"] for group_id in group_ids
    ]
    ctx.event_bus.has_subscribers.return_value = False
    module._ctx = ctx

    transfers: list[int] = []
    callbacks: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        mod,
        "downsample_and_stage_block_ids",
        lambda _cache_context, block_ids: block_ids,
    )
    monkeypatch.setattr(mod, "get_layout_desc", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        mod,
        "transfer_kv_per_object_group",
        lambda *args, object_group_id, **kwargs: transfers.append(object_group_id),
    )
    monkeypatch.setattr(
        mod,
        "submit_callback_to_stream",
        lambda _stream, kind, keys: callbacks.append((kind, list(keys))),
    )
    monkeypatch.setattr(mod, "torch_dev", MagicMock())
    monkeypatch.setattr(mod, "Event", MagicMock())
    return module, ctx, transfers, callbacks


def test_selected_group_store_commits_only_that_complete_object_group(monkeypatch):
    module, ctx, transfers, callbacks = _make_store_module(monkeypatch)
    memory_obj = MagicMock()
    memory_obj.get_size.return_value = 10
    ctx.storage_manager.reserve_write_with_status.return_value = {
        "g1": (L1Error.SUCCESS, memory_obj)
    }

    handle, ok = module.store_groups(
        key=SimpleNamespace(request_id="req", worker_id=1),
        instance_id=1,
        gpu_block_ids=[[11], [22]],
        event_ipc_handle=b"producer",
        selected_engine_group_ids=[1],
    )

    assert (handle, ok) == (b"done", True)
    assert transfers == [1]
    assert callbacks == [("finish_write", ["g1"])]


def test_partial_object_group_selection_is_rejected(monkeypatch):
    module, ctx, transfers, callbacks = _make_store_module(monkeypatch)
    cache_context = module.get_and_touch_context_entry.return_value.cache_context
    cache_context.kv_layer_groups_manager.object_groups = [_og([0, 1])]
    cache_context.kv_layer_groups_manager.num_object_groups = 1

    handle, ok = module.store_groups(
        key=SimpleNamespace(request_id="req", worker_id=1),
        instance_id=1,
        gpu_block_ids=[[11], [22]],
        event_ipc_handle=b"producer",
        selected_engine_group_ids=[0],
    )

    assert (handle, ok) == (b"", False)
    assert transfers == []
    assert callbacks == []
    ctx.storage_manager.reserve_write_with_status.assert_not_called()


def test_partial_store_requires_separate_object_groups(monkeypatch):
    module, ctx, transfers, callbacks = _make_store_module(
        monkeypatch, separate_object_groups=False
    )

    handle, ok = module.store_groups(
        key=SimpleNamespace(request_id="req", worker_id=1),
        instance_id=1,
        gpu_block_ids=[[11], [22]],
        event_ipc_handle=b"producer",
        selected_engine_group_ids=[1],
    )

    assert (handle, ok) == (b"", False)
    assert transfers == []
    assert callbacks == []
    ctx.storage_manager.reserve_write_with_status.assert_not_called()


def test_later_group_failure_aborts_every_reservation(monkeypatch):
    module, ctx, transfers, callbacks = _make_store_module(monkeypatch)
    group0_obj = MagicMock()
    group1_obj = MagicMock()
    group0_obj.get_size.return_value = 10
    group1_obj.get_size.return_value = 10
    ctx.storage_manager.reserve_write_with_status.side_effect = [
        {"g0": (L1Error.SUCCESS, group0_obj)},
        {"g1": (L1Error.SUCCESS, group1_obj)},
    ]

    def fail_second_group(*args, object_group_id, **kwargs):
        transfers.append(object_group_id)
        if object_group_id == 1:
            raise RuntimeError("copy failed")

    monkeypatch.setattr(mod, "transfer_kv_per_object_group", fail_second_group)

    handle, ok = module.store_groups(
        key=SimpleNamespace(request_id="req", worker_id=1),
        instance_id=1,
        gpu_block_ids=[[11], [22]],
        event_ipc_handle=b"producer",
        selected_engine_group_ids=[0, 1],
    )

    assert (handle, ok) == (b"done", False)
    assert transfers == [0, 1]
    assert callbacks == [("abort_write", ["g0", "g1"])]


def test_partial_reservation_aborts_and_skips_copy(monkeypatch):
    module, ctx, transfers, callbacks = _make_store_module(monkeypatch)
    cache_context = module.get_and_touch_context_entry.return_value.cache_context
    cache_context.calculate_num_blocks.return_value = 1
    ctx.resolve_obj_keys.side_effect = lambda _key, group_ids: [
        [f"g{group_id}c0", f"g{group_id}c1"] for group_id in group_ids
    ]
    reserved = MagicMock()
    reserved.get_size.return_value = 10
    ctx.storage_manager.reserve_write_with_status.return_value = {
        "g1c0": (L1Error.SUCCESS, reserved),
        "g1c1": (L1Error.OUT_OF_MEMORY, None),
    }

    handle, ok = module.store_groups(
        key=SimpleNamespace(request_id="req", worker_id=1),
        instance_id=1,
        gpu_block_ids=[[11, 12], [21, 22]],
        event_ipc_handle=b"producer",
        selected_engine_group_ids=[1],
    )

    assert (handle, ok) == (b"done", False)
    assert transfers == []
    assert callbacks == [("abort_write", ["g1c0"])]


def test_complete_existing_objects_satisfy_selected_store(monkeypatch):
    module, ctx, transfers, callbacks = _make_store_module(monkeypatch)
    ctx.storage_manager.reserve_write_with_status.return_value = {
        "g1": (L1Error.KEY_ALREADY_EXISTS, None)
    }

    handle, ok = module.store_groups(
        key=SimpleNamespace(request_id="req", worker_id=1),
        instance_id=1,
        gpu_block_ids=[[11], [22]],
        event_ipc_handle=b"producer",
        selected_engine_group_ids=[1],
    )

    assert (handle, ok) == (b"done", True)
    assert transfers == [1]
    assert callbacks == []


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
    monkeypatch.setattr(mod, "downsample_and_stage_block_ids", lambda cc, b: b)
    monkeypatch.setattr(mod, "submit_callback_to_stream", lambda *a, **k: None)
    monkeypatch.setattr(mod, "torch_dev", MagicMock())
    monkeypatch.setattr(mod, "Event", MagicMock())

    return module, read_calls, transfer_calls


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
