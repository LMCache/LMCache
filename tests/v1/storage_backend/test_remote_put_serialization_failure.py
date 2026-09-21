# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, start_loop_in_thread_with_exceptions
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

pytestmark = pytest.mark.no_shared_allocator


def _create_config(fs_path: Path) -> LMCacheEngineConfig:
    """Create the filesystem-backed configuration used by this regression.

    Args:
        fs_path: Empty filesystem directory that serves as the remote I/O target.

    Returns:
        A configuration that selects the real naive serializer and FS connector.
    """
    return LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        remote_url=f"fs://host:0/{fs_path}",
        remote_serde="naive",
        lmcache_instance_id="remote-serde-regression",
    )


def _create_metadata() -> LMCacheMetadata:
    """Create metadata matching the small CPU KV object used in the test.

    Returns:
        Metadata for one local worker using the standard KV layout.
    """
    return LMCacheMetadata(
        model_name="remote-serde-regression",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(1, 2, 8, 1, 16),
        chunk_size=8,
    )


def _create_key(chunk_hash: int) -> CacheEngineKey:
    """Create a key scoped to the fixture metadata.

    Args:
        chunk_hash: Deterministic chunk identity for the remote object.

    Returns:
        A key accepted by the configured remote backend.
    """
    return CacheEngineKey(
        model_name="remote-serde-regression",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.float32,
    )


def _allocate_memory_obj(backend: RemoteBackend, value: float) -> MemoryObj:
    """Allocate and initialize one small KV object through the real allocator.

    Args:
        backend: Backend whose local CPU allocator owns the returned object.
        value: Fill value used to make the persisted payload observable.

    Returns:
        An allocated KV object with a caller-held reference.
    """
    # ``LMCacheMetadata.get_shapes()`` folds the 5D KV metadata into this
    # 4D connector shape: [KV, layers, tokens, heads * head_size].
    memory_obj = backend.get_allocator_backend().allocate(
        torch.Size((2, 1, 8, 16)),
        torch.float32,
        MemoryFormat.KV_2LTD,
        eviction=False,
        busy_loop=False,
    )
    assert memory_obj is not None
    tensor = memory_obj.tensor
    assert tensor is not None
    tensor.fill_(value)
    return memory_obj


def _completion_event_callback(
    completion_event: threading.Event,
) -> Callable[[CacheEngineKey], None]:
    """Return a completion callback that records RemoteBackend callback delivery.

    Args:
        completion_event: Event set after the backend has processed the put future.

    Returns:
        A callback compatible with ``RemoteBackend.submit_put_task``.
    """

    def record_completion(key: CacheEngineKey) -> None:
        """Set the event after a put callback receives its completed key.

        Args:
            key: Completed remote-storage key supplied by RemoteBackend.
        """
        completion_event.set()

    return record_completion


def _assert_persisted_memory_obj(
    backend: RemoteBackend,
    key: CacheEngineKey,
    expected_shape: torch.Size,
    expected_bytes: bytes,
) -> None:
    """Read one object through RemoteBackend and verify its persisted layout.

    Args:
        backend: Backend used to retrieve the filesystem-backed object.
        key: Remote key that was successfully submitted.
        expected_shape: Logical shape expected from the connector metadata.
        expected_bytes: Exact byte payload expected from the successful put.
    """
    retrieved = backend.get_blocking(key)
    assert retrieved is not None
    try:
        assert retrieved.get_shape() == expected_shape
        assert bytes(retrieved.byte_array) == expected_bytes
    finally:
        retrieved.ref_count_down()


@pytest.fixture
def remote_backend(tmp_path: Path) -> Iterator[RemoteBackend]:
    """Create a real RemoteBackend with a 64 KiB CPU TensorMemoryAllocator.

    Args:
        tmp_path: Per-test directory passed to the real filesystem connector.

    Yields:
        A backend with a running event loop and an instrumented FS connector.
    """
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="remote-serde-regression-loop",
    )
    loop_thread.start()

    config = _create_config(tmp_path)
    metadata = _create_metadata()
    allocator = TensorMemoryAllocator(torch.empty(64 * 1024, dtype=torch.uint8))
    local_cpu_backend = LocalCPUBackend(
        config,
        metadata,
        memory_allocator=allocator,
    )
    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cpu",
    )

    try:
        assert backend.connection is not None
        assert isinstance(backend.connection, InstrumentedRemoteConnector)
        inner_connector = backend.connection.getWrappedConnector()
        assert inner_connector.meta_shapes == [torch.Size((2, 1, 8, 16))]
        assert inner_connector.full_chunk_size_bytes == 1024
        yield backend
    finally:
        backend.close()
        local_cpu_backend.close()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=10)
        loop.close()


@pytest.mark.parametrize("oracle", ("ref_count", "marker", "retry"))
def test_submit_put_task_rolls_back_state_after_serializer_failure(
    remote_backend: RemoteBackend,
    monkeypatch: pytest.MonkeyPatch,
    oracle: str,
) -> None:
    """Check each observable result of a serializer preparation failure.

    Args:
        remote_backend: Real backend with naive serialization and filesystem I/O.
        monkeypatch: Restores the serializer after the one-shot boundary failure.
        oracle: Independent cleanup result observed after the injected failure.
    """
    key = _create_key(1001)
    memory_obj = _allocate_memory_obj(remote_backend, 3.5)
    expected_bytes = bytes(memory_obj.byte_array)
    caller_ref_count = memory_obj.get_ref_count()
    original_serialize = remote_backend.serializer.serialize

    def raise_memory_error(memory_obj: MemoryObj) -> MemoryObj:
        """Inject a serializer-boundary failure before the real serializer runs.

        Args:
            memory_obj: Input object that the fake boundary deliberately rejects.

        Raises:
            MemoryError: Always, to model a recoverable preparation failure.
        """
        raise MemoryError("injected serializer boundary failure")

    try:
        monkeypatch.setattr(remote_backend.serializer, "serialize", raise_memory_error)
        with pytest.raises(MemoryError, match="injected serializer boundary failure"):
            remote_backend.submit_put_task(key, memory_obj)
    finally:
        monkeypatch.setattr(remote_backend.serializer, "serialize", original_serialize)

    try:
        if oracle == "ref_count":
            assert memory_obj.get_ref_count() == caller_ref_count
        elif oracle == "marker":
            assert not remote_backend.exists_in_put_tasks(key)
        else:
            retry_complete = threading.Event()
            retry_future = remote_backend.submit_put_task(
                key,
                memory_obj,
                on_complete_callback=_completion_event_callback(retry_complete),
            )
            retry_future.result(timeout=10)
            assert remote_backend.contains(key)
            assert retry_complete.wait(timeout=10)
            assert not remote_backend.exists_in_put_tasks(key)

            _assert_persisted_memory_obj(
                remote_backend,
                key,
                memory_obj.get_shape(),
                expected_bytes,
            )
            assert memory_obj.get_ref_count() == caller_ref_count
    finally:
        if memory_obj.get_ref_count() == caller_ref_count:
            memory_obj.ref_count_down()


def test_submit_put_task_normal_path_releases_only_backend_reference(
    remote_backend: RemoteBackend,
) -> None:
    """Keep the caller reference while the instrumented connector releases its own.

    Args:
        remote_backend: Real backend with naive serialization and filesystem I/O.
    """
    key = _create_key(1002)
    memory_obj = _allocate_memory_obj(remote_backend, 7.25)
    caller_ref_count = memory_obj.get_ref_count()
    put_complete = threading.Event()

    try:
        put_future = remote_backend.submit_put_task(
            key,
            memory_obj,
            on_complete_callback=_completion_event_callback(put_complete),
        )
        put_future.result(timeout=10)
        assert put_complete.wait(timeout=10)

        assert memory_obj.get_ref_count() == caller_ref_count
        assert not remote_backend.exists_in_put_tasks(key)
        assert remote_backend.contains(key)
        _assert_persisted_memory_obj(
            remote_backend,
            key,
            memory_obj.get_shape(),
            bytes(memory_obj.byte_array),
        )
    finally:
        memory_obj.ref_count_down()


def test_submit_put_task_reserves_key_before_ref_count_up(
    remote_backend: RemoteBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one same-key submit from entering serialization while another is gated.

    Args:
        remote_backend: Real backend with naive serialization and filesystem I/O.
        monkeypatch: Installs the deterministic ref-count and serializer gates.
    """
    key = _create_key(1003)
    first_memory_obj = _allocate_memory_obj(remote_backend, 11.0)
    duplicate_memory_obj = _allocate_memory_obj(remote_backend, 13.0)
    expected_bytes = bytes(first_memory_obj.byte_array)
    first_caller_ref_count = first_memory_obj.get_ref_count()
    duplicate_caller_ref_count = duplicate_memory_obj.get_ref_count()
    ref_count_up_entered = threading.Event()
    release_ref_count_up = threading.Event()
    serializer_calls: list[MemoryObj] = []
    first_errors: list[BaseException] = []
    original_ref_count_up = first_memory_obj.ref_count_up
    original_serialize = remote_backend.serializer.serialize

    def gate_ref_count_up() -> None:
        """Pause the first submit after it obtains its temporary reference.

        The marker must already be present before this gate opens, otherwise a
        second same-key submission could enter serialization.
        """
        original_ref_count_up()
        ref_count_up_entered.set()
        assert release_ref_count_up.wait(timeout=10)

    def raise_memory_error(memory_obj: MemoryObj) -> MemoryObj:
        """Record serializer entry and fail the gated submission.

        Args:
            memory_obj: Object that reached the serializer boundary.

        Raises:
            MemoryError: Always, after recording the serializer entry.
        """
        serializer_calls.append(memory_obj)
        raise MemoryError("injected concurrent serializer failure")

    def submit_first_memory_obj() -> None:
        """Run the gated submit and retain its propagated exception for assertions."""
        try:
            remote_backend.submit_put_task(key, first_memory_obj)
        except BaseException as error:
            first_errors.append(error)

    monkeypatch.setattr(first_memory_obj, "ref_count_up", gate_ref_count_up)
    monkeypatch.setattr(remote_backend.serializer, "serialize", raise_memory_error)
    first_thread = threading.Thread(
        target=submit_first_memory_obj,
        name="remote-serde-ref-count-gate",
    )
    second_error: BaseException | None = None

    try:
        first_thread.start()
        assert ref_count_up_entered.wait(timeout=10)

        try:
            second_future = remote_backend.submit_put_task(key, duplicate_memory_obj)
            second_future.result(timeout=10)
        except BaseException as error:
            second_error = error
        finally:
            release_ref_count_up.set()
            first_thread.join(timeout=10)

        assert not first_thread.is_alive()
        assert second_error is None
        assert len(serializer_calls) == 1
        assert serializer_calls[0] is first_memory_obj
        assert len(first_errors) == 1
        assert isinstance(first_errors[0], MemoryError)
        assert first_memory_obj.get_ref_count() == first_caller_ref_count
        assert duplicate_memory_obj.get_ref_count() == duplicate_caller_ref_count
        assert not remote_backend.exists_in_put_tasks(key)

        monkeypatch.setattr(remote_backend.serializer, "serialize", original_serialize)
        retry_complete = threading.Event()
        retry_future = remote_backend.submit_put_task(
            key,
            first_memory_obj,
            on_complete_callback=_completion_event_callback(retry_complete),
        )
        retry_future.result(timeout=10)
        assert remote_backend.contains(key)
        assert retry_complete.wait(timeout=10)
        assert not remote_backend.exists_in_put_tasks(key)
        _assert_persisted_memory_obj(
            remote_backend,
            key,
            first_memory_obj.get_shape(),
            expected_bytes,
        )
        assert first_memory_obj.get_ref_count() == first_caller_ref_count
    finally:
        release_ref_count_up.set()
        if first_thread.is_alive():
            first_thread.join(timeout=10)
        if first_memory_obj.get_ref_count() == first_caller_ref_count:
            first_memory_obj.ref_count_down()
        if duplicate_memory_obj.get_ref_count() == duplicate_caller_ref_count:
            duplicate_memory_obj.ref_count_down()
