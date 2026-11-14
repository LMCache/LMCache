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
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
    all_null_chunk_masks,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
)
from lmcache.v1.multiprocess.streaming import FSStreamingReader, StreamingLoadTimeout

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
        object_groups=[],
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
        block_ids,
        memory_objs,
        *,
        object_group_id,
        batch_size,
        skip_first_n_tokens,
        direction,
        transfer_key,
        block_ids_host=(),
    ):
        transfer_calls.append((object_group_id, list(memory_objs)))

    cache_context.transfer_object_group.side_effect = fake_transfer
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
    entry = module.get_and_touch_context_entry(1)
    # Keep recording the copies made through the replacement context.
    context.transfer_object_group = entry.cache_context.transfer_object_group
    entry.cache_context = context
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


@pytest.mark.parametrize("load_success", [True, False])
def test_streaming_retrieve_batches_windows_and_releases_on_failure(
    monkeypatch, load_success
):
    module, reads, transfers = _make_module(monkeypatch, 3, [-1])
    reader = MagicMock(spec=FSStreamingReader)
    reader.pool = MagicMock()
    reader.operation_timeout = 30.0
    windows = [MagicMock() for _ in range(3)]
    for window in windows:
        window.memory_obj.get_size.return_value = 10
    reader.pool.acquire.side_effect = windows
    reader.pool.release.return_value = True
    reader.max_window_count.return_value = 3
    reader.load_many.return_value = load_success
    module._ctx.streaming_reader = reader
    session = module._ctx.session_manager.get.return_value
    session.is_current_streaming_range.return_value = True
    monkeypatch.setattr(mod, "get_layout_desc", lambda *a: object())
    monkeypatch.setattr(
        mod,
        "slice_staged_block_ids_for_window",
        lambda cc, gpu, host, index, count: (
            [gpu[0][index : index + count]],
            [host[0][index : index + count]],
        ),
    )
    monkeypatch.setattr(mod, "record_completion_event", lambda *a: MagicMock())
    synchronize = MagicMock()
    monkeypatch.setattr(mod, "synchronize_stream", synchronize)

    _, ok = module.retrieve(
        SimpleNamespace(
            request_id="req",
            cache_salt="salt",
            start=0,
            end=768,
            has_unfull_chunk=lambda _: False,
        ),
        1,
        [[1, 2, 3]],
        b"producer",
    )

    assert ok is load_success
    assert reads == []
    assert len(transfers) == int(load_success)
    if load_success:
        assert transfers[0][1] == [window.memory_obj for window in windows]
    assert reader.pool.acquire.call_count == 3
    if load_success:
        reader.release_after_transfer.assert_called_once()
        reader.pool.release.assert_not_called()
        synchronize.assert_not_called()
    else:
        assert reader.pool.release.call_count == 3
        synchronize.assert_called_once()


def test_streaming_retrieve_leaves_timed_out_windows_with_reader(monkeypatch):
    module, _, _ = _make_module(monkeypatch, 1, [-1])
    reader = MagicMock(spec=FSStreamingReader)
    reader.operation_timeout = 1.0
    reader.pool = MagicMock()
    reader.pool.acquire.return_value = MagicMock()
    reader.max_window_count.return_value = 1
    reader.load_many.side_effect = StreamingLoadTimeout("I/O still pending")
    module._ctx.streaming_reader = reader
    session = module._ctx.session_manager.get.return_value
    session.is_current_streaming_range.return_value = True
    monkeypatch.setattr(mod, "get_layout_desc", lambda *a: object())
    monkeypatch.setattr(mod, "synchronize_stream", lambda *a: None)

    _, ok = module.retrieve(
        SimpleNamespace(
            request_id="req",
            cache_salt="salt",
            start=0,
            end=256,
            has_unfull_chunk=lambda _: False,
        ),
        1,
        [[1]],
        b"producer",
    )

    assert not ok
    reader.pool.release.assert_not_called()


def test_mixed_streaming_reads_l1_and_fs_in_order_and_releases_l1(monkeypatch):
    module, reads, transfers = _make_module(monkeypatch, 3, [-1])
    reader = MagicMock(spec=FSStreamingReader)
    reader.pool = MagicMock()
    reader.operation_timeout = 30.0
    window = MagicMock()
    window.memory_obj.get_size.return_value = 10
    reader.pool.acquire.return_value = window
    reader.pool.release.return_value = True
    reader.max_window_count.return_value = 3
    reader.load_many.return_value = True
    module.context.streaming_reader = reader
    session = module.context.session_manager.get.return_value
    session.is_current_streaming_range.return_value = True
    owners = {"g0c0": 7, "g0c2": 7}
    session.get_prefetch_owners.return_value = owners
    completion = [(7, ["g0c0", "g0c2"])]
    module.context.storage_manager.prepare_read_completion.return_value = completion
    callback = MagicMock()
    monkeypatch.setattr(mod, "submit_callback_to_stream", callback)
    monkeypatch.setattr(mod, "get_layout_desc", lambda *a: object())
    monkeypatch.setattr(
        mod,
        "slice_staged_block_ids_for_window",
        lambda cc, gpu, host, index, count: (
            [gpu[0][index : index + count]],
            [host[0][index : index + count]],
        ),
    )
    monkeypatch.setattr(mod, "record_completion_event", lambda *a: MagicMock())
    monkeypatch.setattr(mod, "synchronize_stream", lambda *a: None)

    _, ok = module.retrieve(
        SimpleNamespace(
            request_id="req",
            cache_salt="salt",
            start=0,
            end=768,
            has_unfull_chunk=lambda _: False,
        ),
        1,
        [[1, 2, 3]],
        b"producer",
    )

    assert ok
    assert reads == [["g0c0"], ["g0c2"]]
    assert len(transfers) == 3
    assert reader.load_many.call_count == 1
    assert reader.load_many.call_args.args[0] == ["g0c1"]
    assert transfers[1][1] == [window.memory_obj]
    module.context.storage_manager.prepare_read_completion.assert_called_once_with(
        ["g0c0", "g0c2"], owners
    )
    assert any(
        call.args[1:] == ("finish_read_by_owner", completion)
        for call in callback.call_args_list
    )


def test_mixed_streaming_retrieves_and_releases_unfull_l1_tail(monkeypatch):
    module, reads, _ = _make_module(monkeypatch, 2, [-1])
    reader = MagicMock(spec=FSStreamingReader)
    reader.pool = MagicMock()
    reader.operation_timeout = 30.0
    window = MagicMock()
    window.memory_obj.get_size.return_value = 10
    reader.pool.acquire.return_value = window
    reader.max_window_count.return_value = 1
    reader.load_many.return_value = True
    module.context.streaming_reader = reader
    module.context.resolve_unfull_chunk_obj_keys.return_value = ["tail"]
    owners = {"g0c0": 7, "tail": 7}
    session = module.context.session_manager.get.return_value
    session.is_current_streaming_range.return_value = True
    session.get_prefetch_owners.return_value = owners
    tail_transfer = MagicMock(return_value=10)
    monkeypatch.setattr(module, "_retrieve_unfull_chunk_objects", tail_transfer)
    monkeypatch.setattr(mod, "get_layout_desc", lambda *a: object())
    monkeypatch.setattr(
        mod,
        "slice_staged_block_ids_for_window",
        lambda cc, gpu, host, index, count: (
            [gpu[0][index : index + count]],
            [host[0][index : index + count]],
        ),
    )
    monkeypatch.setattr(mod, "record_completion_event", lambda *a: MagicMock())
    monkeypatch.setattr(mod, "synchronize_stream", lambda *a: None)

    _, ok = module.retrieve(
        SimpleNamespace(
            request_id="req",
            cache_salt="salt",
            start=0,
            end=517,
            has_unfull_chunk=lambda _: True,
        ),
        1,
        [[1, 2, 3]],
        b"producer",
    )

    assert ok
    assert reads == [["g0c0"]]
    assert reader.load_many.call_args.args[0] == ["g0c1"]
    tail_transfer.assert_called_once()
    module.context.storage_manager.prepare_read_completion.assert_called_once_with(
        ["g0c0", "tail"], owners
    )


def test_store_reserves_real_page_zero_and_only_present_state_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, context, _reads, transfers = _make_checkpoint_module(monkeypatch)
    _handle, ok = module.store(
        SimpleNamespace(request_id="req", worker_id=1, has_unfull_chunk=_no_unfull),
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


def test_store_skip_chunks_drops_caller_masked_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-masked chunk (blend ``--enable-dedup-content-kv``) is never
    reserved in any object group and reports False in the stored mask."""
    module, _context, _reads, _transfers = _make_checkpoint_module(monkeypatch)
    key: Any = SimpleNamespace(
        request_id="req", worker_id=1, has_unfull_chunk=_no_unfull
    )
    _handle, ok, mask = module.store_with_chunk_mask(
        key,
        1,
        [[0, 1, 2, 3], [-1, -1, 0, 1], [-1, -1, 2, 3]],
        b"producer",
        skip_chunks=[False, True],
    )
    assert ok
    reserved = [
        call.args[0]
        for call in cast(
            MagicMock, module.context.storage_manager.reserve_write
        ).call_args_list
    ]
    assert all("g0c1" not in keys and "g1c1" not in keys for keys in reserved)
    assert ["g0c0"] in reserved
    assert mask[1] is False


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
        key=SimpleNamespace(
            request_id="req", cache_salt="salt", has_unfull_chunk=_no_unfull
        ),
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
        key=SimpleNamespace(
            request_id="req", cache_salt="salt", has_unfull_chunk=_no_unfull
        ),
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
        key=SimpleNamespace(
            request_id="req", cache_salt="salt", has_unfull_chunk=_no_unfull
        ),
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
    cache_context.transfer_object_group.side_effect = RuntimeError(
        "partly enqueued transfer"
    )
    _, ok = module.retrieve(
        SimpleNamespace(
            request_id="req", cache_salt="salt", has_unfull_chunk=_no_unfull
        ),
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
#  unfull chunk (store / retrieve in the same transaction)             #
# ------------------------------------------------------------------ #


def _unfull_key(**kwargs):
    return SimpleNamespace(
        request_id="req",
        worker_id=1,
        cache_salt="salt",
        end=3,
        has_unfull_chunk=lambda chunk_size: True,
        **kwargs,
    )


def _make_unfull_store_module(monkeypatch, obj_keys):
    """``obj_keys`` lists each object group's keys, its unfull chunk's last."""
    module, context, _reads, _transfers = _make_checkpoint_module(monkeypatch)
    module.context.resolve_obj_keys.return_value = [keys[:-1] for keys in obj_keys]
    module.context.resolve_unfull_chunk_obj_keys.return_value = [
        keys[-1] for keys in obj_keys
    ]
    reserved: list[tuple[list, bool]] = []

    def reserve_write(keys, layout, is_temporary=False):
        reserved.append((list(keys), is_temporary))
        return {key: MagicMock(get_size=MagicMock(return_value=10)) for key in keys}

    module.context.storage_manager.reserve_write.side_effect = reserve_write
    unfull_transfers: list[tuple[int, list, list]] = []
    monkeypatch.setattr(
        mod,
        "transfer_unfull_chunk",
        lambda cc, block_ids, skips, leads, memory_obj, group, buffer, direction: (
            unfull_transfers.append((group, list(block_ids), list(skips)))
        ),
    )
    monkeypatch.setattr(mod, "get_unfull_chunk_layout_desc", lambda *a: object())
    monkeypatch.setattr(mod, "get_unfull_chunk_staging_buffer", lambda *a: object())
    return module, reserved, unfull_transfers


def test_store_of_only_an_unfull_chunk_reserves_temporary_objects(monkeypatch):
    module, reserved, unfull_transfers = _make_unfull_store_module(
        monkeypatch, [["g0u"], ["g1u"]]
    )

    _handle, ok, mask = module.store_with_chunk_mask(
        _unfull_key(request_configs={"lmcache.temporary_unfull_chunk": True}),
        1,
        [[5], [6], [7]],
        b"producer",
    )

    assert ok and mask == [True]
    assert reserved[-2:] == [(["g0u"], True), (["g1u"], True)]
    # Each list is right-aligned to the 2-block window; the padding is skipped.
    fitted = ([[5, 5], [6, 6], [7, 7]], [1, 1, 1])
    assert unfull_transfers == [(0, *fitted), (1, *fitted)]


def test_store_skips_an_unfull_chunk_that_does_not_fit_the_staging_buffer(
    monkeypatch,
):
    module, reserved, unfull_transfers = _make_unfull_store_module(
        monkeypatch, [["g0c0", "g0u"], ["g1c0", "g1u"]]
    )
    monkeypatch.setattr(
        mod,
        "get_unfull_chunk_staging_buffer",
        lambda cc, group: None if group == 1 else object(),
    )

    _handle, ok, _mask = module.store_with_chunk_mask(
        _unfull_key(request_configs=None),
        1,
        [[0, 1, 5], [-1, 1, 6], [-1, 1, 7]],
        b"producer",
    )

    # The full chunk is still stored; no group of the tail is reserved.
    assert ok
    assert (["g0c0"], False) in reserved
    assert all(keys not in (["g0u"], ["g1u"]) for keys, _ in reserved)
    assert unfull_transfers == []


def test_store_appends_the_unfull_chunk_after_full_chunks(monkeypatch):
    module, reserved, unfull_transfers = _make_unfull_store_module(
        monkeypatch, [["g0c0", "g0u"], ["g1c0", "g1u"]]
    )

    _handle, ok, mask = module.store_with_chunk_mask(
        _unfull_key(request_configs=None),
        1,
        # Two blocks per full chunk, then the unfull chunk's block.
        [[0, 1, 5], [-1, 1, 6], [-1, 1, 7]],
        b"producer",
    )

    assert ok and mask == [True, True]
    assert (["g0c0"], False) in reserved and (["g0u"], False) in reserved
    assert [group for group, _, _ in unfull_transfers] == [0, 1]
    assert unfull_transfers[0][1:] == ([[5, 5], [6, 6], [7, 7]], [1, 1, 1])


def test_retrieve_of_only_an_unfull_chunk_skips_aux_groups(monkeypatch):
    module, read_calls, transfer_calls = _make_module(
        monkeypatch,
        1,
        num_chunks_in_sw=[1, -1, -1],
        group_kinds=("recurrent", "attention", "aux"),
    )
    module.context.resolve_obj_keys.return_value = [[], [], []]
    module.context.resolve_unfull_chunk_obj_keys.return_value = ["g0u", "g1u", "g2u"]
    monkeypatch.setattr(
        mod,
        "trim_and_stage_unfull_chunk_block_ids",
        lambda cc, b, num_tokens, skip_tokens: (b, [0] * len(b), [0] * len(b)),
    )
    monkeypatch.setattr(mod, "get_unfull_chunk_staging_buffer", lambda *a: object())
    unfull_transfers: list[int] = []
    monkeypatch.setattr(
        mod,
        "transfer_unfull_chunk",
        lambda cc, block_ids, skips, leads, memory_obj, group, buffer, direction: (
            unfull_transfers.append(group)
        ),
    )

    _handle, ok = module.retrieve(_unfull_key(), 1, [[4], [5], [6]], b"x")

    assert ok is True
    assert read_calls == [["g0u", "g1u"]]
    assert transfer_calls == []
    assert unfull_transfers == [0, 1]


def _no_unfull(chunk_size: int) -> bool:
    """``has_unfull_chunk`` of a key double that addresses full chunks only."""
    return False
