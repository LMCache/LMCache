# SPDX-License-Identifier: Apache-2.0
"""Tests for transfer lifecycle cleanup in the NIXL storage backends.

When ``batched_submit_put_task`` returns before the transfer has completed
(always in the static backend; async mode in the dynamic backend), the
caller (``StorageManager.batched_put``) drops its reference to each source
``MemoryObj``.  Unless the backend takes its own reference for the duration
of the transfer, the page can be recycled by the allocator and overwritten
by a concurrent write while NIXL is still reading it.

Covers both ``NixlStaticStorageBackend`` and ``NixlDynamicStorageBackend``
(async and sync modes), including partial acquisition and read/write failure
cleanup.

These tests use lightweight mocks so they run without the NIXL Python
package or CUDA hardware: the *real* unbound methods are exercised on a
``Mock(spec=NixlStaticStorageBackend)`` while external dependencies
(NIXL agent, descriptor pool) are replaced by mocks.  The NIXL transfer
itself (``agent.post_blocking``) is gated on a ``threading.Event`` so the
in-flight window is controllable from the test.
"""

# Standard
from typing import List, Tuple
from unittest.mock import MagicMock, Mock
import asyncio
import os
import sys
import threading
import time
import types

# Third Party
import pytest
import torch


def _install_nixl_mock_if_absent() -> list[str]:
    """Temporarily install the minimal NIXL API needed at import time."""
    try:
        # Third Party
        import nixl  # noqa: F401
        import nixl._api  # noqa: F401

        return []
    except ImportError:
        pass

    nixl_bind_mock = MagicMock()
    nixl_bind_mock.nixlRegDList = object
    nixl_bind_mock.nixlXferDList = object
    nixl_bind_mock.nixlBackendError = Exception

    sync_t_mock = MagicMock()
    sync_t_mock.NIXL_THREAD_SYNC_STRICT = "NIXL_THREAD_SYNC_STRICT"

    api_mock = types.ModuleType("nixl._api")
    api_mock.nixl_agent = MagicMock  # type: ignore[attr-defined]
    api_mock.nixl_agent_config = MagicMock  # type: ignore[attr-defined]
    api_mock.nixl_prepped_dlist_handle = MagicMock  # type: ignore[attr-defined]
    api_mock.nixl_xfer_handle = MagicMock  # type: ignore[attr-defined]
    api_mock.nixlBind = nixl_bind_mock  # type: ignore[attr-defined]
    api_mock.nixl_thread_sync_t = sync_t_mock  # type: ignore[attr-defined]

    nixl_mock = types.ModuleType("nixl")
    nixl_mock._api = api_mock  # type: ignore[attr-defined]

    inserted = []
    for name, module in (("nixl", nixl_mock), ("nixl._api", api_mock)):
        if name not in sys.modules:
            sys.modules[name] = module
            inserted.append(name)
    return inserted


_NIXL_MOCK_KEYS = _install_nixl_mock_if_absent()

# First Party
from lmcache.utils import CacheEngineKey  # noqa: E402
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.nixl_storage_backend import (  # noqa: E402
    NixlDesc,
    NixlDynamicStorageAgent,
    NixlDynamicStorageBackend,
    NixlKeyMetadata,
    NixlStaticStorageBackend,
)

# The backend module has captured the real or temporary NIXL symbols. Remove
# our temporary modules immediately so other test modules can still skip when
# NIXL is unavailable instead of accidentally collecting against this fake.
for _name in _NIXL_MOCK_KEYS:
    sys.modules.pop(_name, None)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key(chunk_hash: int) -> CacheEngineKey:
    """Create a single CacheEngineKey for testing."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
    )


def _make_obj(num_floats: int = 8) -> TensorMemoryObj:
    """Create a standalone TensorMemoryObj with ref_count=1 (caller's ref)."""
    raw_data = torch.zeros(num_floats, dtype=torch.float32)
    metadata = MemoryObjMetadata(
        shape=torch.Size([num_floats]),
        dtype=torch.float32,
        address=0,
        phy_size=num_floats * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


class _LoopThread:
    """A real asyncio event loop running in a background thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        # Fire-and-forget put tasks raise on deliberately failed transfers;
        # collect instead of letting the default handler print at GC time.
        self.task_errors: list[dict] = []
        self.loop.set_exception_handler(
            lambda loop, context: self.task_errors.append(context)
        )
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()


@pytest.fixture
def loop_thread():
    lt = _LoopThread()
    try:
        yield lt
    finally:
        lt.close()


def _mock_static_backend(
    loop: asyncio.AbstractEventLoop, transfer_gate: threading.Event
):
    """Return a ``Mock(spec=NixlStaticStorageBackend)`` wired for put tasks.

    The real ``batched_submit_put_task``, ``mem_to_storage``,
    ``add_key_to_dict``, and ``exists_in_put_tasks`` implementations are
    bound onto the mock; the NIXL agent and descriptor pool are mocked.
    ``agent.post_blocking`` blocks until *transfer_gate* is set, giving the
    test a controllable in-flight window.
    """
    backend = Mock(spec=NixlStaticStorageBackend)
    backend.loop = loop
    backend.key_lock = threading.Lock()
    backend.progress_lock = threading.Lock()
    backend.key_dict = {}
    backend.progress_set = set()
    backend.cache_policy = Mock()

    backend.pool = Mock()
    backend.pool.get_num_available_descs.return_value = 100
    backend.pool.pop.side_effect = iter(range(100))

    backend.agent = Mock()
    backend.agent.get_mem_to_storage_handle.return_value = Mock()
    backend.agent.post_blocking.side_effect = lambda handle: transfer_gate.wait(
        timeout=10
    )
    backend.agent.release_handle.return_value = None

    for method in ("mem_to_storage", "add_key_to_dict", "exists_in_put_tasks"):
        setattr(
            backend,
            method,
            types.MethodType(getattr(NixlStaticStorageBackend, method), backend),
        )
    return backend


def _submit(backend, keys: List[CacheEngineKey], objs: List[MemoryObj]) -> None:
    NixlStaticStorageBackend.batched_submit_put_task(backend, keys, objs)


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_put(n: int) -> Tuple[List[CacheEngineKey], List[MemoryObj]]:
    return [_make_key(i) for i in range(n)], [_make_obj() for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStaticPutSourceBufferProtection:
    """The backend must hold a reference to each source buffer for the
    whole in-flight window, so the page cannot be recycled while NIXL is
    still reading it."""

    def test_source_buffer_held_during_inflight_put(self, loop_thread) -> None:
        gate = threading.Event()
        backend = _mock_static_backend(loop_thread.loop, gate)
        keys, objs = _make_put(2)

        _submit(backend, keys, objs)

        # The caller drops its reference as soon as submit returns
        # (StorageManager.batched_put does exactly this).
        for obj in objs:
            obj.ref_count_down()

        # The transfer has not completed (gate not set): the backend must
        # still hold its own reference, keeping the buffer alive.
        for obj in objs:
            assert obj.get_ref_count() >= 1, (
                "source buffer released while the NIXL transfer is still "
                "in flight — the page can be recycled and overwritten"
            )

        # Let the transfer finish before the loop fixture tears down, so
        # no pending task is destroyed with the loop.
        gate.set()
        assert _wait_until(lambda: all(obj.get_ref_count() == 0 for obj in objs))

    def test_source_buffer_released_after_transfer_completes(self, loop_thread) -> None:
        gate = threading.Event()
        backend = _mock_static_backend(loop_thread.loop, gate)
        keys, objs = _make_put(2)

        _submit(backend, keys, objs)
        for obj in objs:
            obj.ref_count_down()

        gate.set()

        assert _wait_until(lambda: all(obj.get_ref_count() == 0 for obj in objs)), (
            "backend did not release its reference after the transfer completed"
        )
        # Completion also clears the in-flight marker.
        assert _wait_until(lambda: not backend.progress_set)

    def test_deduplicated_keys_do_not_leak_references(self, loop_thread) -> None:
        """Keys dropped by the dedup-on-submit filter (already in
        ``progress_set``) must not acquire a reference that nobody
        releases."""
        gate = threading.Event()
        gate.set()  # transfers complete immediately
        backend = _mock_static_backend(loop_thread.loop, gate)
        keys, objs = _make_put(1)

        # The key is already being written: submit must drop it.
        backend.progress_set.add(keys[0])

        _submit(backend, keys, objs)

        assert objs[0].get_ref_count() == 1, (
            "deduplicated key leaked a reference on its source buffer"
        )

    def test_closed_loop_rolls_back_submit_state(self) -> None:
        """A synchronous scheduling failure must release submit-side refs."""
        loop = asyncio.new_event_loop()
        loop.close()
        gate = threading.Event()
        backend = _mock_static_backend(loop, gate)
        keys, objs = _make_put(2)

        _submit(backend, keys, objs)

        assert not backend.progress_set
        assert all(obj.get_ref_count() == 1 for obj in objs)


# ---------------------------------------------------------------------------
# Dynamic backend
# ---------------------------------------------------------------------------


def _mock_dynamic_backend(
    loop: asyncio.AbstractEventLoop,
    transfer_gate: threading.Event,
    async_mode: bool,
):
    """Return a ``Mock(spec=NixlDynamicStorageBackend)`` wired for put tasks.

    The real put-path methods are bound onto the mock; the NIXL agent and
    handle acquisition are mocked.  In async mode the transfer completes
    only once *transfer_gate* is set (polled via ``check_xfer_state``); in
    sync mode ``post_blocking`` blocks on the gate.
    """
    backend = Mock(spec=NixlDynamicStorageBackend)
    backend.loop = loop
    backend.async_mode = async_mode
    backend.progress_lock = threading.Lock()
    backend.progress_set = set()
    backend.memory_allocator = Mock()
    backend.memory_allocator.align_bytes = 4096
    backend._cache_add = Mock()

    # descs, reg_descs, xfer_handler, handle
    backend._acquire_storage_handle = Mock(return_value=([], Mock(), Mock(), Mock()))

    backend.agent = Mock()
    backend.agent.mem_type = "OBJ"
    backend.agent.post_async.return_value = "PROC"
    backend.agent.nixl_agent.check_xfer_state.side_effect = lambda handle: (
        "DONE" if transfer_gate.is_set() else "PROC"
    )
    backend.agent.post_blocking.side_effect = lambda handle: transfer_gate.wait(
        timeout=10
    )
    backend.agent.release_handle.return_value = None
    backend.agent.release_storage_handler.return_value = None

    for method in (
        "batched_submit_put_task",
        "mem_to_storage",
        "_submit_async_mem_to_storage",
        "_run_sync_mem_to_storage",
        "_wait_for_transfer",
        "exists_in_put_tasks",
    ):
        setattr(
            backend,
            method,
            types.MethodType(getattr(NixlDynamicStorageBackend, method), backend),
        )
    return backend


class TestDynamicPutSourceBufferProtection:
    """Regression guard: the dynamic backend already protects source
    buffers (``ref_count_up`` before scheduling in async mode, released in
    ``_wait_for_transfer``; sync mode blocks until completion).  These
    tests lock that invariant in."""

    def test_async_source_buffer_held_during_inflight_put(self, loop_thread) -> None:
        gate = threading.Event()
        backend = _mock_dynamic_backend(loop_thread.loop, gate, async_mode=True)
        keys, objs = _make_put(2)

        backend.batched_submit_put_task(keys, objs)

        for obj in objs:
            obj.ref_count_down()

        for obj in objs:
            assert obj.get_ref_count() >= 1, (
                "source buffer released while the NIXL transfer is still "
                "in flight — the page can be recycled and overwritten"
            )

        # Let the transfer finish before the loop fixture tears down, so
        # no pending task is destroyed with the loop.
        gate.set()
        assert _wait_until(lambda: all(obj.get_ref_count() == 0 for obj in objs))

    def test_async_source_buffer_released_after_transfer_completes(
        self, loop_thread
    ) -> None:
        gate = threading.Event()
        backend = _mock_dynamic_backend(loop_thread.loop, gate, async_mode=True)
        keys, objs = _make_put(2)

        backend.batched_submit_put_task(keys, objs)
        for obj in objs:
            obj.ref_count_down()

        gate.set()

        assert _wait_until(lambda: all(obj.get_ref_count() == 0 for obj in objs)), (
            "backend did not release its reference after the transfer completed"
        )
        assert _wait_until(lambda: not backend.progress_set)

    def test_sync_mode_does_not_leak_references(self, loop_thread) -> None:
        """Sync mode blocks until the transfer completes, so no backend
        reference is taken — and none must be released."""
        gate = threading.Event()
        gate.set()  # let post_blocking return immediately
        backend = _mock_dynamic_backend(loop_thread.loop, gate, async_mode=False)
        keys, objs = _make_put(2)

        backend.batched_submit_put_task(keys, objs)

        for obj in objs:
            assert obj.get_ref_count() == 1, (
                "sync-mode put changed the source buffer ref count"
            )
        assert not backend.progress_set

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_closed_loop_rolls_back_submit_state(self, async_mode: bool) -> None:
        loop = asyncio.new_event_loop()
        loop.close()
        gate = threading.Event()
        backend = _mock_dynamic_backend(loop, gate, async_mode=async_mode)
        keys, objs = _make_put(2)

        backend.batched_submit_put_task(keys, objs)

        assert not backend.progress_set
        assert all(obj.get_ref_count() == 1 for obj in objs)


# ---------------------------------------------------------------------------
# Write-failure cleanup
# ---------------------------------------------------------------------------


class TestWriteFailureCleanup:
    """A failed write must leave no trace: the key must not stay marked
    in-flight forever, buffer references must be released, nothing may be
    published as durable, and (static) key_dict must not point at a
    storage slot that was never written."""

    @staticmethod
    def _dynamic_backend(loop, gate, async_mode=True):
        backend = _mock_dynamic_backend(loop, gate, async_mode=async_mode)
        backend.presence_cache_only = False
        backend._cache_contains = Mock(return_value=False)
        backend.key_exists = Mock(return_value=False)
        for method in ("contains", "_exists_in_put_tasks_or_cache"):
            setattr(
                backend,
                method,
                types.MethodType(getattr(NixlDynamicStorageBackend, method), backend),
            )
        return backend

    def test_async_transfer_error_clears_inflight_entry(self, loop_thread) -> None:
        gate = threading.Event()
        backend = self._dynamic_backend(loop_thread.loop, gate)
        backend.agent.nixl_agent.check_xfer_state.side_effect = lambda handle: (
            "ERR" if gate.is_set() else "PROC"
        )
        keys, objs = _make_put(1)

        backend.batched_submit_put_task(keys, objs)
        gate.set()

        assert _wait_until(lambda: not backend.exists_in_put_tasks(keys[0])), (
            "failed transfer left the key marked in-flight forever"
        )
        assert backend.contains(keys[0]) is False
        backend._cache_add.assert_not_called()
        # Writer reference released despite the failure.
        assert _wait_until(lambda: all(o.get_ref_count() == 1 for o in objs))
        assert loop_thread.task_errors == []

    @pytest.mark.parametrize(
        "release_method", ["release_handle", "release_storage_handler"]
    )
    def test_async_release_failure_still_cleans_logical_state(
        self, loop_thread, release_method: str
    ) -> None:
        """A NIXL cleanup error must not pin keys or source buffers."""
        gate = threading.Event()
        backend = self._dynamic_backend(loop_thread.loop, gate)
        getattr(backend.agent, release_method).side_effect = RuntimeError(
            "release failed"
        )
        keys, objs = _make_put(1)

        backend.batched_submit_put_task(keys, objs)
        gate.set()

        assert _wait_until(lambda: not backend.progress_set)
        assert _wait_until(lambda: all(o.get_ref_count() == 1 for o in objs))
        backend.agent.release_storage_handler.assert_called_once()
        assert loop_thread.task_errors == []

    def test_async_post_failure_clears_inflight_entry(self, loop_thread) -> None:
        """Nobody queries the fire-and-forget future: a failure while
        posting the transfer must clean up after itself."""
        gate = threading.Event()
        gate.set()
        backend = self._dynamic_backend(loop_thread.loop, gate)
        backend.agent.post_async.side_effect = RuntimeError("post failed")
        keys, objs = _make_put(2)

        backend.batched_submit_put_task(keys, objs)

        assert _wait_until(lambda: not backend.progress_set), (
            "failed async post left keys marked in-flight forever"
        )
        backend._cache_add.assert_not_called()
        assert _wait_until(lambda: all(o.get_ref_count() == 1 for o in objs))
        assert loop_thread.task_errors == []

    def test_async_handle_acquisition_failure_is_swallowed(self, loop_thread) -> None:
        gate = threading.Event()
        gate.set()
        backend = self._dynamic_backend(loop_thread.loop, gate)
        backend._acquire_storage_handle.side_effect = RuntimeError(
            "registration failed"
        )
        keys, objs = _make_put(1)

        backend.batched_submit_put_task(keys, objs)

        assert _wait_until(lambda: not backend.progress_set)
        backend._cache_add.assert_not_called()
        assert _wait_until(lambda: all(o.get_ref_count() == 1 for o in objs))
        assert loop_thread.task_errors == []

    def test_file_handle_acquisition_failure_unlinks_created_file(
        self, tmp_path
    ) -> None:
        """Registration failure after O_CREAT must not publish an empty file."""
        backend = Mock(spec=NixlDynamicStorageBackend)
        backend.path = str(tmp_path)
        backend.direct_io_flag = 0
        backend._use_b128_object_keys = False
        backend.progress_lock = threading.Lock()
        backend.progress_set = set()
        backend.presence_cache_only = False
        backend._cache_contains = Mock(return_value=False)
        backend._cache_add = Mock()

        backend.agent = Mock()
        backend.agent.mem_type = "FILE"

        created_paths = []

        def fail_after_file_creation(descs, page_size):
            assert page_size == 4096
            assert len(descs) == 1
            assert descs[0].path is not None
            assert os.path.exists(descs[0].path)
            created_paths.append(descs[0].path)
            raise RuntimeError("file registration failed")

        backend.agent.create_batched_storage_handler.side_effect = (
            fail_after_file_creation
        )
        backend.agent.nixl_desc_exists.side_effect = (
            lambda meta_info, path: os.path.exists(os.path.join(path, meta_info))
        )

        for method in (
            "_format_object_key",
            "_format_object_key_url_safe",
            "_build_descs",
            "_acquire_storage_handle",
            "key_exists",
            "exists_in_put_tasks",
            "_exists_in_put_tasks_or_cache",
            "contains",
        ):
            setattr(
                backend,
                method,
                types.MethodType(getattr(NixlDynamicStorageBackend, method), backend),
            )

        key = _make_key(99)
        with pytest.raises(RuntimeError, match="file registration failed"):
            backend._acquire_storage_handle([key], [0], [0], page_size=4096, write=True)

        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])
        assert backend.contains(key) is False

    def test_sync_transfer_failure_is_swallowed_and_cleared(self, loop_thread) -> None:
        """A sync put failure must not propagate: StorageManager.batched_put
        would abort before its ref_count_down loop, leaking the source
        buffers of the whole batch across all backends (best-effort offload
        semantics, as proposed in #3956)."""
        gate = threading.Event()
        gate.set()
        backend = self._dynamic_backend(loop_thread.loop, gate, async_mode=False)
        backend.agent.post_blocking.side_effect = RuntimeError("transfer failed")
        on_complete = Mock()
        keys, objs = _make_put(1)

        backend.batched_submit_put_task(keys, objs, on_complete_callback=on_complete)

        assert not backend.progress_set, (
            "failed sync transfer left the key marked in-flight forever"
        )
        backend._cache_add.assert_not_called()
        # A failed put must not report completion.
        on_complete.assert_not_called()

    def test_static_failure_rolls_back_to_clean_miss(self, loop_thread) -> None:
        """A failed static transfer must not leave key_dict pointing at a
        pool slot that was never written — a lookup would serve garbage.
        The slot goes back to the pool."""
        gate = threading.Event()
        backend = _mock_static_backend(loop_thread.loop, gate)
        for method in ("contains", "remove"):
            setattr(
                backend,
                method,
                types.MethodType(getattr(NixlStaticStorageBackend, method), backend),
            )
        backend.agent.post_blocking.side_effect = RuntimeError("transfer failed")
        keys, objs = _make_put(1)

        _submit(backend, keys, objs)

        assert _wait_until(lambda: not backend.exists_in_put_tasks(keys[0])), (
            "failed transfer left the key marked in-flight forever"
        )
        assert backend.contains(keys[0]) is False
        assert not backend.key_dict
        backend.pool.push.assert_called_once_with(0)
        backend.agent.release_handle.assert_called_once_with(
            backend.agent.get_mem_to_storage_handle.return_value
        )
        assert _wait_until(lambda: all(o.get_ref_count() == 1 for o in objs))

    @pytest.mark.parametrize("failing_release", ["dlist", "registration"])
    def test_storage_handler_release_attempts_all_cleanup(
        self, tmp_path, failing_release: str
    ) -> None:
        """A NIXL release error must not skip later releases or FD closure."""
        agent = Mock(spec=NixlDynamicStorageAgent)
        agent.mem_type = "FILE"
        agent.nixl_agent = Mock()
        if failing_release == "dlist":
            agent.nixl_agent.release_dlist_handle.side_effect = RuntimeError(
                "dlist release failed"
            )
        else:
            agent.nixl_agent.deregister_memory.side_effect = RuntimeError(
                "deregistration failed"
            )

        path = tmp_path / "release-test"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        descs = [NixlDesc(device_id=fd, meta_info=path.name, path=str(path))]

        NixlDynamicStorageAgent.release_storage_handler(agent, Mock(), Mock(), descs)

        agent.nixl_agent.release_dlist_handle.assert_called_once()
        agent.nixl_agent.deregister_memory.assert_called_once()
        with pytest.raises(OSError):
            os.fstat(fd)


class TestReadAndAcquisitionCleanup:
    """Pre-existing read/acquisition failures must release owned resources."""

    @pytest.mark.parametrize("failure_stage", ["descs", "handler"])
    def test_partial_storage_handler_acquisition_deregisters_memory(
        self, failure_stage: str
    ) -> None:
        agent = Mock(spec=NixlDynamicStorageAgent)
        agent.agent_name = "test-agent"
        agent.mem_type = "OBJ"
        agent.nixl_agent = Mock()
        reg_descs = Mock()
        agent.nixl_agent.register_memory.return_value = reg_descs

        if failure_stage == "descs":
            agent.nixl_agent.get_xfer_descs.side_effect = RuntimeError(
                "descriptor preparation failed"
            )
        else:
            agent.nixl_agent.get_xfer_descs.return_value = Mock()
            agent.nixl_agent.prep_xfer_dlist.side_effect = RuntimeError(
                "handler preparation failed"
            )

        with pytest.raises(RuntimeError, match="preparation failed"):
            NixlDynamicStorageAgent.create_batched_storage_handler(
                agent,
                [NixlDesc(device_id=0, meta_info="test-key")],
                page_size=4096,
            )

        agent.nixl_agent.deregister_memory.assert_called_once_with(reg_descs)

    def test_partial_storage_handler_preserves_original_error(self) -> None:
        agent = Mock(spec=NixlDynamicStorageAgent)
        agent.mem_type = "OBJ"
        agent.nixl_agent = Mock()
        agent.nixl_agent.register_memory.return_value = Mock()
        agent.nixl_agent.get_xfer_descs.side_effect = RuntimeError(
            "descriptor preparation failed"
        )
        agent.nixl_agent.deregister_memory.side_effect = RuntimeError("rollback failed")

        with pytest.raises(RuntimeError, match="descriptor preparation failed"):
            NixlDynamicStorageAgent.create_batched_storage_handler(
                agent,
                [NixlDesc(device_id=0, meta_info="test-key")],
                page_size=4096,
            )

    @pytest.mark.parametrize("failure_stage", ["handle", "transfer"])
    def test_static_read_failure_releases_owned_resources(
        self, failure_stage: str
    ) -> None:
        backend = Mock(spec=NixlStaticStorageBackend)
        backend._local_cpu_backend = None
        backend.memory_allocator = Mock()
        obj = _make_obj()
        backend.memory_allocator.allocate.return_value = obj
        backend.agent = Mock()
        handle = Mock()
        backend.agent.get_storage_to_mem_handle.return_value = handle
        release_ref_counts = []
        backend.agent.release_handle.side_effect = (
            lambda released_handle: release_ref_counts.append(obj.get_ref_count())
        )

        if failure_stage == "handle":
            backend.agent.get_storage_to_mem_handle.side_effect = RuntimeError(
                "handle acquisition failed"
            )
        else:
            backend.agent.post_blocking.side_effect = RuntimeError(
                "read transfer failed"
            )

        metadata = NixlKeyMetadata(
            shape=obj.meta.shape,
            dtype=obj.meta.dtype,
            fmt=obj.meta.fmt,
            index=7,
        )
        with pytest.raises(RuntimeError, match="failed"):
            asyncio.run(
                NixlStaticStorageBackend._nixl_transfer_async(backend, [metadata])
            )

        assert obj.get_ref_count() == 0
        if failure_stage == "handle":
            backend.agent.release_handle.assert_not_called()
        else:
            backend.agent.release_handle.assert_called_once_with(handle)
            assert release_ref_counts == [1]
