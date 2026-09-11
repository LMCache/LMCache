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
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    TrimPolicy,
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


def test_read_prefetched_results_can_defer_release_until_copy_completion():
    storage = StorageManager.__new__(StorageManager)
    key = _key(b"deferred-read")
    memory_obj = object()
    finish_read = Mock(return_value={key: L1Error.SUCCESS})
    storage._l1_manager = SimpleNamespace(
        unsafe_read=Mock(return_value={key: (L1Error.SUCCESS, memory_obj)}),
        finish_read=finish_read,
    )
    storage._event_bus = Mock()

    with storage.read_prefetched_results([key], release_on_exit=False) as objs:
        assert objs == [memory_obj]
    finish_read.assert_not_called()

    storage.finish_read_prefetched([key])
    finish_read.assert_called_once_with([key], read_locks=1)


def test_deferred_read_does_not_release_on_copy_exception():
    storage = StorageManager.__new__(StorageManager)
    key = _key(b"deferred-copy-error")
    memory_obj = object()
    finish_read = Mock(return_value={key: L1Error.SUCCESS})
    storage._l1_manager = SimpleNamespace(
        unsafe_read=Mock(return_value={key: (L1Error.SUCCESS, memory_obj)}),
        finish_read=finish_read,
    )
    storage._event_bus = Mock()

    with pytest.raises(RuntimeError, match="copy failed"):
        with storage.read_prefetched_results([key], release_on_exit=False):
            raise RuntimeError("copy failed")

    finish_read.assert_not_called()
    storage.finish_read_prefetched([key])
    finish_read.assert_called_once_with([key], read_locks=1)


def test_generation_is_validated_and_carried_by_spec_and_handle():
    spec = PrefetchRequestSpec(
        keys=[_key()],
        group_layout_descs={0: _layout()},
        policy=TrimPolicy.SPARSE,
        generation=4,
    )
    handle = PrefetchHandle(
        prefetch_request_id=3,
        external_request_id="request",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=spec.generation,
    )

    assert spec.generation == 4
    assert handle.generation == 4
    with pytest.raises(ValueError, match="generation"):
        PrefetchRequestSpec(
            keys=[_key()],
            group_layout_descs={0: _layout()},
            generation=-1,
        )


def test_l1_only_handles_have_independent_idempotent_cleanup():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage.finish_read_prefetched = Mock()

    first = PrefetchHandle(
        prefetch_request_id=-1,
        external_request_id="first",
        l1_found_indices=(0,),
        l1_hit_chunks=1,
        total_requested_keys=1,
        submit_time=0.0,
    )
    second = PrefetchHandle(
        prefetch_request_id=-1,
        external_request_id="second",
        l1_found_indices=(0,),
        l1_hit_chunks=1,
        total_requested_keys=1,
        submit_time=0.0,
    )
    first_key = _key(b"first")
    second_key = _key(b"second")
    storage._remember_prefetch_handle(first, [first_key], 1)
    storage._remember_prefetch_handle(second, [second_key], 1)

    storage._release_prefetch_lease(first)
    storage._release_prefetch_lease(first)

    assert id(second) in storage._prefetch_handle_metadata
    storage._release_prefetch_lease(second)
    assert storage.finish_read_prefetched.call_count == 2
    assert id(first) in storage._released_prefetch_handles
    assert id(second) in storage._released_prefetch_handles


def test_warm_handle_release_does_not_drop_nonexistent_read_locks():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage.finish_read_prefetched = Mock()

    handle = PrefetchHandle(
        prefetch_request_id=-1,
        external_request_id="warm",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
    )
    storage._remember_prefetch_handle(handle, [_key(b"warm")], 0)

    storage.release_prefetch_task(handle)

    assert storage.finish_read_prefetched.call_count == 0


def test_release_uses_generation_guard_and_is_idempotent():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage._prefetch_controller = Mock()
    storage._prefetch_controller.cancel_prefetch_request.return_value = False
    storage.query_prefetch_status = Mock(return_value=None)
    storage.finish_read_prefetched = Mock()

    key = _key(b"release")
    handle = PrefetchHandle(
        prefetch_request_id=8,
        external_request_id="request",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=12,
    )
    storage._remember_prefetch_handle(handle, [key], 2)

    storage.release_prefetch_task(handle, [key])
    storage.release_prefetch_task(handle, [key])

    storage._prefetch_controller.cancel_prefetch_request.assert_called_once_with(
        8, generation=12
    )
    storage._prefetch_controller.forget_prefetch_result.assert_called_once_with(
        8, generation=12
    )
    storage.finish_read_prefetched.assert_called_once_with([key], read_locks=2)


def test_release_waits_for_controller_cleanup_before_releasing_l1_lock():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage._prefetch_controller = Mock()
    storage._prefetch_controller.cancel_prefetch_request.return_value = True
    storage.wait_prefetch_status = Mock(return_value=True)
    storage.query_prefetch_status = Mock(return_value=None)
    storage.finish_read_prefetched = Mock()

    key = _key(b"wait-before-release")
    handle = PrefetchHandle(
        prefetch_request_id=9,
        external_request_id="request",
        l1_found_indices=(0,),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=13,
    )
    storage._remember_prefetch_handle(handle, [key], 1)

    storage.release_prefetch_task(handle)

    storage.wait_prefetch_status.assert_called_once_with(handle, timeout=None)
    storage.finish_read_prefetched.assert_called_once_with([key], read_locks=1)


def test_release_failure_keeps_lease_metadata_for_retry():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage.finish_read_prefetched = Mock(
        side_effect=[RuntimeError("read lock release failed"), None]
    )

    key = _key(b"retry-release")
    handle = PrefetchHandle(
        prefetch_request_id=-1,
        external_request_id="retry",
        l1_found_indices=(0,),
        l1_hit_chunks=1,
        total_requested_keys=1,
        submit_time=0.0,
    )
    storage._remember_prefetch_handle(handle, [key], 1)

    with pytest.raises(RuntimeError, match="read lock release failed"):
        storage._release_prefetch_lease(handle)

    assert id(handle) in storage._prefetch_handle_metadata
    assert id(handle) not in storage._released_prefetch_handles

    storage._release_prefetch_lease(handle)

    assert id(handle) not in storage._prefetch_handle_metadata
    assert storage.finish_read_prefetched.call_count == 2


def test_released_handle_bookkeeping_does_not_keep_handles_alive():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage.finish_read_prefetched = Mock()

    handle = PrefetchHandle(
        prefetch_request_id=-1,
        external_request_id="weakref",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=0,
        submit_time=0.0,
    )
    handle_id = id(handle)
    storage._remember_prefetch_handle(handle, [], 0)
    storage._release_prefetch_lease(handle)
    handle_ref = weakref.ref(handle)

    del handle
    gc.collect()

    assert handle_ref() is None
    assert handle_id not in storage._released_prefetch_handles


def test_concurrent_sparse_submit_keeps_one_job_and_releases_loser():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
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

    def submit_prefetch_task(*_args, **_kwargs):
        submit_barrier.wait(timeout=5)
        handle = PrefetchHandle(
            prefetch_request_id=len(handles),
            external_request_id="request:0:1",
            l1_found_indices=(),
            l1_hit_chunks=0,
            total_requested_keys=1,
            submit_time=0.0,
            generation=0,
        )
        handles.append(handle)
        return handle

    module._ctx.storage_manager = SimpleNamespace(
        submit_prefetch_task=Mock(side_effect=submit_prefetch_task),
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

    def submit_prefetch_task(*_args, **_kwargs):
        submit_barrier.wait(timeout=5)
        handle = PrefetchHandle(
            prefetch_request_id=len(handles),
            external_request_id="request:0:1",
            l1_found_indices=(),
            l1_hit_chunks=0,
            total_requested_keys=1,
            submit_time=0.0,
            generation=0,
        )
        handles.append(handle)
        return handle

    cancel = Mock(side_effect=[RuntimeError("temporary cleanup failure"), None])
    module._ctx.storage_manager = SimpleNamespace(
        submit_prefetch_task=Mock(side_effect=submit_prefetch_task),
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
    module._sparse_jobs_lock = threading.Lock()

    key = _key(b"retrieve")
    handle = PrefetchHandle(
        prefetch_request_id=11,
        external_request_id="request:0:1",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=0,
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
    module._sparse_jobs_lock = threading.Lock()

    class _FailingStream:
        def synchronize(self):
            raise RuntimeError("stream is unavailable")

    handle = PrefetchHandle(
        prefetch_request_id=12,
        external_request_id="request:0:1",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=0,
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


def test_sparse_cancel_marks_job_before_releasing_lease():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_jobs_lock = threading.Lock()
    handle = PrefetchHandle(
        prefetch_request_id=13,
        external_request_id="request:0:1",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=0,
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
    module._sparse_jobs_lock = threading.Lock()
    key = _key(b"cancel-before-retrieve")
    handle = PrefetchHandle(
        prefetch_request_id=14,
        external_request_id="request:0:1",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=0,
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


def test_sparse_completion_release_failure_leaves_job_retryable():
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._sparse_jobs = {}
    module._sparse_jobs_lock = threading.Lock()
    key = _key(b"completion-release")
    handle = PrefetchHandle(
        prefetch_request_id=15,
        external_request_id="request:0:1",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=0,
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
