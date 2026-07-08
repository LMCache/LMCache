# SPDX-License-Identifier: Apache-2.0
"""Tests for source-buffer ref counting in the NIXL storage backends.

When ``batched_submit_put_task`` returns before the transfer has completed
(always in the static backend; async mode in the dynamic backend), the
caller (``StorageManager.batched_put``) drops its reference to each source
``MemoryObj``.  Unless the backend takes its own reference for the duration
of the transfer, the page can be recycled by the allocator and overwritten
by a concurrent write while NIXL is still reading it.

Covers both ``NixlStaticStorageBackend`` and ``NixlDynamicStorageBackend``
(async and sync modes).

These tests use lightweight mocks so they run without NIXL or CUDA
hardware: the *real* unbound methods are exercised on a
``Mock(spec=NixlStaticStorageBackend)`` while external dependencies
(NIXL agent, descriptor pool) are replaced by mocks.  The NIXL transfer
itself (``agent.post_blocking``) is gated on a ``threading.Event`` so the
in-flight window is controllable from the test.
"""

# Standard
from typing import List, Tuple
from unittest.mock import Mock
import asyncio
import threading
import time
import types

# Third Party
import pytest
import torch

pytest.importorskip("nixl")

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.nixl_storage_backend import (
    NixlDynamicStorageBackend,
    NixlStaticStorageBackend,
)

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
