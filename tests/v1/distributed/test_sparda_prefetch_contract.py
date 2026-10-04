# SPDX-License-Identifier: Apache-2.0

"""Contract tests for generation-scoped logical prefetch leases."""

# Standard
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import gc
import threading
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchLockMode,
    PrefetchResult,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
    _SparsePrefetchJob,
)


def _key(name: bytes = b"key") -> ObjectKey:
    return ObjectKey(chunk_hash=name, model_name="contract-test", kv_rank=0)


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([1])], [torch.float16])


def _spec(keys, readers=1, lock_mode=PrefetchLockMode.LOCK):
    return PrefetchTaskSpec(
        key_groups=[GroupedObjectKeys(keys, 0, _layout())],
        num_kv_readers=readers,
        fetching_policy="full",
        lock_mode=lock_mode,
    )


def _result(size, indices):
    hits = Bitmap(size)
    hits.batched_set(indices)
    return PrefetchResult([hits], [hits], [Bitmap(size)])


def _storage():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._prefetch_handle_metadata = {}
    storage._prefetch_controller = Mock()
    storage._prefetch_controller.wait_prefetch_result.return_value = True
    storage.finish_read_prefetched = Mock()
    next_id = 0

    def submit(spec, external_request_id=""):
        nonlocal next_id
        handle = PrefetchHandle(next_id, external_request_id, spec.group_size, 0.0)
        next_id += 1
        return handle

    storage.submit_prefetch_task = Mock(side_effect=submit)
    storage.query_prefetch_status = Mock(return_value=_result(1, [0]))
    return storage


def _read_storage(key):
    storage = StorageManager.__new__(StorageManager)
    memory_obj = object()
    storage._l1_manager = SimpleNamespace(
        unsafe_read=Mock(return_value={key: (L1Error.SUCCESS, memory_obj)}),
        finish_read=Mock(return_value={key: L1Error.SUCCESS}),
    )
    storage._l1_managers_by_id = {0: storage._l1_manager}
    storage._event_bus = Mock()
    return storage, memory_obj


def test_read_prefetched_results_can_defer_release_until_copy_completion():
    key = _key(b"deferred-read")
    storage, memory_obj = _read_storage(key)
    with storage.read_prefetched_results([key], release_on_error=False) as objs:
        assert objs == [memory_obj]
    storage._l1_manager.finish_read.assert_not_called()
    storage.finish_read_prefetched([key])
    storage._l1_manager.finish_read.assert_called_once_with([key], read_locks=1)


def test_deferred_read_does_not_release_on_copy_exception():
    key = _key(b"deferred-copy-error")
    storage, _ = _read_storage(key)
    with pytest.raises(RuntimeError, match="copy failed"):
        with storage.read_prefetched_results([key], release_on_error=False):
            raise RuntimeError("copy failed")
    storage._l1_manager.finish_read.assert_not_called()
    storage.finish_read_prefetched([key])
    storage._l1_manager.finish_read.assert_called_once_with([key], read_locks=1)


def test_negative_generation_is_rejected_before_context_access():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module.get_and_touch_context_entry = Mock()
    assert module.sparse_prefetch(0, "request", -1, 1, [_key()]) is False
    module.get_and_touch_context_entry.assert_not_called()


def test_l1_only_handles_have_independent_idempotent_cleanup():
    storage = _storage()
    first_key, second_key = _key(b"first"), _key(b"second")
    first = storage.submit_prefetch_lease(_spec([first_key]))
    second = storage.submit_prefetch_lease(_spec([second_key]))
    storage.release_prefetch_task(first)
    storage.release_prefetch_task(first)
    assert storage.query_prefetch_lease(second) is not None
    storage.release_prefetch_task(second)
    assert storage.finish_read_prefetched.call_args_list == [
        (([first_key], 1),),
        (([second_key], 1),),
    ]


def test_warm_prefetch_cannot_acquire_a_read_lease():
    storage = _storage()
    with pytest.raises(ValueError, match="read locks"):
        storage.submit_prefetch_lease(
            _spec([_key()], lock_mode=PrefetchLockMode.NO_LOCK)
        )
    storage.submit_prefetch_task.assert_not_called()
    storage.finish_read_prefetched.assert_not_called()


def test_release_checks_handle_identity_and_is_idempotent():
    storage = _storage()
    key = _key(b"release")
    handle = storage.submit_prefetch_lease(_spec([key], readers=2))
    copied_handle = PrefetchHandle(handle.prefetch_request_id, "request", 1, 0.0)
    storage.release_prefetch_task(copied_handle)
    storage.finish_read_prefetched.assert_not_called()
    storage.release_prefetch_task(handle, [key])
    storage.release_prefetch_task(handle, [key])
    storage.finish_read_prefetched.assert_called_once_with([key], 2)


def test_release_waits_for_controller_cleanup_before_releasing_l1_lock():
    storage = _storage()
    key = _key(b"wait-before-release")
    handle = storage.submit_prefetch_lease(_spec([key]))
    order = []
    storage._prefetch_controller.wait_prefetch_result.side_effect = (
        lambda *_args: order.append("io_complete") or True
    )
    storage.finish_read_prefetched.side_effect = lambda *_args: order.append("unlock")
    storage.cancel_prefetch_task(handle)
    assert order == ["io_complete", "unlock"]


def test_release_keeps_observed_result_and_releases_only_retained_keys():
    storage = _storage()
    keys = [_key(bytes([i])) for i in range(3)]
    storage.query_prefetch_status.return_value = _result(3, [0, 2])
    handle = storage.submit_prefetch_lease(_spec(keys))
    assert storage.query_prefetch_lease(handle).hit_cells[0].get_indices_list() == [
        0,
        2,
    ]
    storage.release_prefetch_task(handle, [keys[0], keys[2]])
    storage.query_prefetch_status.assert_called_once_with(handle)
    storage.finish_read_prefetched.assert_called_once_with([keys[0], keys[2]], 1)


def test_release_failure_keeps_lease_for_retry():
    storage = _storage()
    key = _key(b"retry")
    handle = storage.submit_prefetch_lease(_spec([key]))
    storage.finish_read_prefetched.side_effect = [RuntimeError("cleanup failed"), None]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        storage.release_prefetch_task(handle)
    assert storage.query_prefetch_lease(handle) is not None
    storage.release_prefetch_task(handle)
    assert storage.query_prefetch_lease(handle) is None
    assert storage.finish_read_prefetched.call_count == 2


def test_release_rejects_keys_outside_the_lease():
    storage = _storage()
    key = _key(b"owned")
    handle = storage.submit_prefetch_lease(_spec([key]))
    with pytest.raises(ValueError, match="belong"):
        storage.release_prefetch_task(handle, [_key(b"foreign")])
    storage.finish_read_prefetched.assert_not_called()
    storage.release_prefetch_task(handle)
    storage.finish_read_prefetched.assert_called_once_with([key], 1)


def test_released_handle_bookkeeping_does_not_keep_handles_alive():
    storage = _storage()
    handle = storage.submit_prefetch_lease(_spec([_key()]))
    handle_ref = weakref.ref(handle)
    storage.release_prefetch_task(handle)
    storage.query_prefetch_status.reset_mock()
    del handle
    gc.collect()
    assert handle_ref() is None


def test_concurrent_sparse_submit_keeps_one_job_and_releases_loser():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()
    module.get_and_touch_context_entry = Mock(
        return_value=SimpleNamespace(model_name="contract-test", world_size=1)
    )
    module._ctx = SimpleNamespace(
        layout_desc_registry=SimpleNamespace(
            find_group_layout_descs=Mock(return_value={0: _layout()}),
            find_attn_desc=Mock(return_value=SimpleNamespace(num_object_groups=1)),
        )
    )

    handles = []
    submit_barrier = threading.Barrier(2)

    def submit_prefetch_lease(*_args, **_kwargs):
        submit_barrier.wait(timeout=5)
        handle = PrefetchHandle(
            prefetch_request_id=len(handles),
            external_request_id="request:0:1",
            total_requested_keys=1,
            submit_time=0.0,
        )
        handles.append(handle)
        return handle

    module._ctx.storage_manager = SimpleNamespace(
        submit_prefetch_lease=Mock(side_effect=submit_prefetch_lease),
        cancel_prefetch_task=Mock(),
    )

    results = []

    def submit():
        results.append(module.sparse_prefetch(0, "request", 0, 1, [_key()]))

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(results) == [True, True]
    assert len(module._sparse_jobs) == 1
    module._ctx.storage_manager.cancel_prefetch_task.assert_called_once()


def test_duplicate_sparse_submit_retries_loser_cleanup_after_failure():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()
    module.get_and_touch_context_entry = Mock(
        return_value=SimpleNamespace(model_name="contract-test", world_size=1)
    )
    module._ctx = SimpleNamespace(
        layout_desc_registry=SimpleNamespace(
            find_group_layout_descs=Mock(return_value={0: _layout()}),
            find_attn_desc=Mock(return_value=SimpleNamespace(num_object_groups=1)),
        )
    )

    handles = []
    submit_barrier = threading.Barrier(2)

    def submit_prefetch_lease(*_args, **_kwargs):
        submit_barrier.wait(timeout=5)
        handle = PrefetchHandle(
            prefetch_request_id=len(handles),
            external_request_id="request:0:1",
            total_requested_keys=1,
            submit_time=0.0,
        )
        handles.append(handle)
        return handle

    cancel = Mock(side_effect=[RuntimeError("temporary cleanup failure"), None])
    module._ctx.storage_manager = SimpleNamespace(
        submit_prefetch_lease=Mock(side_effect=submit_prefetch_lease),
        cancel_prefetch_task=cancel,
        release_prefetch_task=Mock(),
    )

    results = []

    def submit():
        results.append(module.sparse_prefetch(0, "request", 0, 1, [_key()]))

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(results) == [True, True]
    assert len(module._sparse_jobs) == 1
    assert len(module._sparse_orphan_handles) == 1

    assert module._cleanup_sparse_instance(0) is True
    assert module._sparse_orphan_handles == {}
    assert cancel.call_count == 2


def test_sparse_retrieve_waits_for_submitted_copy_before_release(monkeypatch):
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()

    key = _key(b"retrieve")
    handle = PrefetchHandle(
        prefetch_request_id=11,
        external_request_id="request:0:1",
        total_requested_keys=1,
        submit_time=0.0,
    )
    job = _SparsePrefetchJob(
        handle=handle,
        keys=(key,),
        instance_id=0,
        request_id="request",
        generation=0,
        layer_id=1,
        found_indices=(0,),
    )
    module._sparse_jobs[(0, "request", 0, 1)] = job

    order = []

    class _Stream:
        def synchronize(self):
            order.append("stream_synchronize")

    cache_context = SimpleNamespace(
        device=torch.device("cpu"),
        stream=_Stream(),
        cupy_stream=object(),
        kv_layer_groups_manager=SimpleNamespace(
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            num_kernel_groups=1,
        ),
        calculate_num_blocks=lambda *_args: 1,
    )
    event_backend = Mock()
    event_backend.create_event.return_value = object()
    event_backend.import_event.return_value = object()
    entry = SimpleNamespace(
        cache_context=cache_context,
        model_name="contract-test",
        world_size=1,
        event_backend=event_backend,
    )
    module.get_and_touch_context_entry = Mock(return_value=entry)

    @contextmanager
    def read_prefetched_results(_keys, **_kwargs):
        yield [object()]

    def release_prefetch_task(_handle, **_kwargs):
        order.append("release")

    module._ctx = SimpleNamespace(
        chunk_size=1,
        storage_manager=SimpleNamespace(
            read_prefetched_results=read_prefetched_results,
            release_prefetch_task=release_prefetch_task,
        ),
    )

    def stage_after_copy(*_args, **_kwargs):
        order.append("copy_submitted")
        raise RuntimeError("next object failed after the first H2D submit")

    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod

    monkeypatch.setattr(mod, "_stage_sparse_layer", stage_after_copy)
    monkeypatch.setattr(mod, "submit_callback_to_stream", Mock())
    monkeypatch.setattr(mod.torch_dev, "device", lambda _device: nullcontext())
    monkeypatch.setattr(mod.torch_dev, "stream", lambda _stream: nullcontext())

    _event, result = module.sparse_retrieve(
        0,
        "request",
        0,
        1,
        [key],
        [[7]],
        b"producer-event",
    )

    assert result == (False, [0])
    assert order == [
        "copy_submitted",
        "stream_synchronize",
        "release",
    ]
    assert module._sparse_jobs == {}


def test_sparse_copy_sync_failure_retains_job_and_lease():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()

    class _FailingStream:
        def synchronize(self):
            raise RuntimeError("stream is unavailable")

    handle = PrefetchHandle(
        prefetch_request_id=12,
        external_request_id="request:0:1",
        total_requested_keys=1,
        submit_time=0.0,
    )
    job = _SparsePrefetchJob(
        handle=handle,
        keys=(_key(b"retain"),),
        instance_id=0,
        request_id="request",
        generation=0,
        layer_id=1,
        copy_submitted=True,
        copy_stream=_FailingStream(),
    )
    module._sparse_jobs[(0, "request", 0, 1)] = job
    module._ctx = SimpleNamespace(
        storage_manager=SimpleNamespace(release_prefetch_task=Mock())
    )

    assert module._cleanup_sparse_job(job) is False
    assert module._sparse_jobs[(0, "request", 0, 1)] is job
    module._ctx.storage_manager.release_prefetch_task.assert_not_called()
    assert isinstance(job.last_error, RuntimeError)


def test_sparse_cleanup_releases_all_retrieved_keys_after_copy_sync():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()
    keys = (_key(b"miss"), _key(b"hit"))
    handle = PrefetchHandle(
        prefetch_request_id=16,
        external_request_id="request:0:1",
        total_requested_keys=2,
        submit_time=0.0,
    )
    job = _SparsePrefetchJob(
        handle=handle,
        keys=keys,
        instance_id=0,
        request_id="request",
        generation=0,
        layer_id=1,
        found_indices=(1,),
        copy_submitted=True,
        copy_synchronized=True,
    )
    module._sparse_jobs[(0, "request", 0, 1)] = job
    module._ctx = SimpleNamespace(
        storage_manager=SimpleNamespace(release_prefetch_task=Mock())
    )

    assert module._cleanup_sparse_job(job) is True

    module._ctx.storage_manager.release_prefetch_task.assert_called_once_with(
        handle, keys=[keys[1]]
    )
    assert module._sparse_jobs == {}


def test_sparse_cancel_marks_job_before_releasing_lease():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()
    handle = PrefetchHandle(
        prefetch_request_id=13,
        external_request_id="request:0:1",
        total_requested_keys=1,
        submit_time=0.0,
    )
    job = _SparsePrefetchJob(
        handle=handle,
        keys=(_key(b"cancel"),),
        instance_id=0,
        request_id="request",
        generation=0,
        layer_id=1,
    )
    module._sparse_jobs[(0, "request", 0, 1)] = job
    release_started = threading.Event()
    allow_release = threading.Event()

    def release_prefetch_task(_handle, **_kwargs):
        release_started.set()
        assert allow_release.wait(timeout=5)

    module._ctx = SimpleNamespace(
        storage_manager=SimpleNamespace(release_prefetch_task=release_prefetch_task)
    )
    cancel_thread = threading.Thread(
        target=lambda: module.sparse_cancel_prefetch(0, "request", 0, 1)
    )
    cancel_thread.start()
    assert release_started.wait(timeout=5)
    with job.condition:
        assert job.cancel_requested is True
    allow_release.set()
    cancel_thread.join(timeout=5)
    assert not cancel_thread.is_alive()
    assert module._sparse_jobs == {}


def test_sparse_retrieve_rejects_job_marked_for_cancel(monkeypatch):
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod

    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()
    key = _key(b"cancel-before-retrieve")
    handle = PrefetchHandle(
        prefetch_request_id=14,
        external_request_id="request:0:1",
        total_requested_keys=1,
        submit_time=0.0,
    )
    job = _SparsePrefetchJob(
        handle=handle,
        keys=(key,),
        instance_id=0,
        request_id="request",
        generation=0,
        layer_id=1,
        found_indices=(0,),
        cancel_requested=True,
    )
    module._sparse_jobs[(0, "request", 0, 1)] = job

    cache_context = SimpleNamespace(
        device=torch.device("cpu"),
        stream=object(),
        cupy_stream=object(),
        kv_layer_groups_manager=SimpleNamespace(
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            num_kernel_groups=1,
        ),
        calculate_num_blocks=lambda *_args: 1,
    )
    event_backend = Mock()
    event_backend.create_event.return_value = object()
    event_backend.import_event.return_value = object()
    event_backend.export_event.return_value = b"completion"
    entry = SimpleNamespace(
        cache_context=cache_context,
        model_name="contract-test",
        world_size=1,
        event_backend=event_backend,
    )
    module.get_and_touch_context_entry = Mock(return_value=entry)
    module._resolve_sparse_status = Mock(return_value=[0])

    @contextmanager
    def read_prefetched_results(_keys, **_kwargs):
        yield [object()]

    module._ctx = SimpleNamespace(
        chunk_size=1,
        storage_manager=SimpleNamespace(
            read_prefetched_results=read_prefetched_results,
        ),
    )
    stage = Mock()
    monkeypatch.setattr(mod, "_stage_sparse_layer", stage)
    monkeypatch.setattr(mod, "submit_callback_to_stream", Mock())
    monkeypatch.setattr(mod.torch_dev, "device", lambda _device: nullcontext())
    monkeypatch.setattr(mod.torch_dev, "stream", lambda _stream: nullcontext())

    _event, result = module.sparse_retrieve(
        0,
        "request",
        0,
        1,
        [key],
        [[7]],
        b"producer-event",
    )

    assert result == (False, [0])
    stage.assert_not_called()


def test_sparse_cancel_missing_job_is_idempotent():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()

    assert module.sparse_cancel_prefetch(0, "missing", 0, 1) is True


def test_sparse_completion_release_failure_leaves_job_retryable():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()
    key = _key(b"completion-release")
    handle = PrefetchHandle(
        prefetch_request_id=15,
        external_request_id="request:0:1",
        total_requested_keys=1,
        submit_time=0.0,
    )
    job = _SparsePrefetchJob(
        handle=handle,
        keys=(key,),
        instance_id=0,
        request_id="request",
        generation=0,
        layer_id=1,
        found_indices=(0,),
        retrieving=True,
        completion_submitted=True,
    )
    module._sparse_jobs[(0, "request", 0, 1)] = job
    release = Mock(side_effect=RuntimeError("lease still busy"))
    module._ctx = SimpleNamespace(
        storage_manager=SimpleNamespace(release_prefetch_task=release)
    )

    module._complete_sparse_prefetch((0, "request", 0, 1))

    assert module._sparse_jobs[(0, "request", 0, 1)] is job
    assert job.retrieving is False
    assert job.completed is False


@pytest.mark.parametrize("separate_groups", [False, True])
@pytest.mark.parametrize("preserve_head_geometry", [False, True])
def test_unified_sparse_copy_changes_only_the_selected_serving_layer(
    separate_groups, preserve_head_geometry
):
    """A single-layer request preserves other K/V layers and unselected pages."""
    # First Party
    from lmcache import lmcache_native
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        _stage_sparse_layer,
    )

    group_layers = [[0, 1], [2, 3]] if separate_groups else [[0, 1, 2, 3]]
    groups = [
        SimpleNamespace(layer_indices=layers, shape_desc=SimpleNamespace(bs=1))
        for layers in group_layers
    ]
    targets = [torch.full((10, 1, 2), -1.0) for _ in range(4)]
    sources = []
    for layers in group_layers:
        source = torch.empty(1, len(layers), 4, 2)
        for local, physical in enumerate(layers):
            source[0, local] = torch.arange(8).reshape(4, 2) + physical * 100
        sources.append(source.unsqueeze(-2) if preserve_head_geometry else source)
    context = SimpleNamespace(
        kv_tensors=targets,
        lmcache_tokens_per_chunk=4,
        kv_layer_groups_manager=SimpleNamespace(
            kernel_groups=groups,
            object_groups=[
                SimpleNamespace(kernel_group_indices=list(range(len(groups))))
            ],
        ),
        calculate_num_blocks=lambda tokens, group_id: tokens,
        get_engine_kv_format=lambda group_id: (
            lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_HS
            if preserve_head_geometry
            else lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
        ),
    )
    obj = SimpleNamespace(get_tensor=lambda group_position: sources[group_position])
    selected = [2, 7, 9, 4]
    _stage_sparse_layer(context, [obj], [selected] * len(groups), 0, 1)
    for physical in (1, 3):
        expected = torch.full((10, 1, 2), -1.0)
        expected[selected, 0] = (
            torch.arange(8, dtype=expected.dtype).reshape(4, 2) + physical * 100
        )
        torch.testing.assert_close(targets[physical], expected)
    assert torch.all(targets[0] == -1) and torch.all(targets[2] == -1)


def test_sparse_partial_hit_preserves_logical_key_positions():
    """Independent logical chunks are columns, not required counterpart rows."""
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_orphan_handles = {}
    module._sparse_jobs_lock = threading.Lock()
    module.get_and_touch_context_entry = Mock(
        return_value=SimpleNamespace(model_name="contract-test", world_size=1)
    )
    storage = _storage()
    storage.query_prefetch_status.return_value = _result(3, [0, 2])
    module._ctx = SimpleNamespace(
        storage_manager=storage,
        layout_desc_registry=SimpleNamespace(
            find_group_layout_descs=Mock(return_value={0: _layout()}),
            find_attn_desc=Mock(return_value=SimpleNamespace(num_object_groups=1)),
        ),
    )
    keys = [_key(bytes([i])) for i in range(3)]
    assert module.sparse_prefetch(0, "request", 0, 0, keys) is True
    spec = storage.submit_prefetch_task.call_args.args[0]
    assert len(spec.key_groups) == 1 and spec.key_groups[0].keys == keys
    assert module.sparse_query_prefetch(0, "request", 0, 0) == [0, 2]
    assert module.sparse_release_prefetch(0, "request", 0, 0) is True
    storage.finish_read_prefetched.assert_called_once_with([keys[0], keys[2]], 1)
