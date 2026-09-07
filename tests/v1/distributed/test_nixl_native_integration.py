# SPDX-License-Identifier: Apache-2.0
"""Opt-in public-interface integration tests for native NIXL storage."""

# Standard
from pathlib import Path
import errno
import importlib.util
import os
import select
import threading
import time
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.nixl_native_l2_adapter import (
    NixlNativeL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_ALIGNMENT = 4096
_CHUNK_SIZE = 4096
_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

requires_nixl_integration = pytest.mark.skipif(
    os.environ.get("LMCACHE_NIXL_INTEGRATION") != "1",
    reason="set LMCACHE_NIXL_INTEGRATION=1 to run native NIXL tests",
)
requires_nixl_extension = pytest.mark.skipif(
    importlib.util.find_spec("lmcache.lmcache_nixl") is None,
    reason="optional lmcache_nixl extension is not built",
)
requires_nixl_object_integration = pytest.mark.skipif(
    os.environ.get("LMCACHE_NIXL_OBJECT_INTEGRATION") != "1",
    reason="set LMCACHE_NIXL_OBJECT_INTEGRATION=1 to run NIXL OBJ tests",
)


def _aligned_arena(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a page-aligned CPU byte arena and retain its owner."""
    owner = torch.empty(size + _ALIGNMENT, dtype=torch.uint8)
    offset = -owner.data_ptr() % _ALIGNMENT
    return owner, owner[offset : offset + size]


def _memory_obj(
    arena: torch.Tensor,
    offset: int,
    fill_value: int,
) -> TensorMemoryObj:
    """Create a public MemoryObj backed by one aligned arena chunk."""
    tensor = arena[offset : offset + _CHUNK_SIZE]
    tensor.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([_CHUNK_SIZE]),
        dtype=torch.uint8,
        address=offset,
        phy_size=_CHUNK_SIZE,
        fmt=MemoryFormat.BINARY_BUFFER,
        ref_count=1,
    )
    return TensorMemoryObj(tensor, metadata, parent_allocator=None)


def _wait_for_fd(event_fd: int, timeout: float = 10.0) -> None:
    """Wait for and consume one public adapter completion notification."""
    readable, _, _ = select.select([event_fd], [], [], timeout)
    assert readable, "timed out waiting for native NIXL completion"
    consume_fd(event_fd)


def _make_posix_adapter(
    base_path: Path,
    arena: torch.Tensor,
    *,
    use_direct_io: bool = False,
    num_workers: int = 1,
) -> L2AdapterInterface:
    """Create a POSIX native NIXL adapter for an existing L1 arena."""
    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "POSIX",
            "backend_params": {
                "file_path": str(base_path),
                "use_direct_io": str(use_direct_io).lower(),
            },
            "num_workers": num_workers,
        }
    )
    descriptor = L1MemoryDesc(
        ptr=arena.data_ptr(),
        size=arena.numel(),
        align_bytes=_ALIGNMENT,
    )
    return create_l2_adapter(config, descriptor)


def _supports_direct_io(base_path: Path) -> tuple[bool, str]:
    """Report whether the test filesystem accepts an O_DIRECT file."""
    direct_flag = getattr(os, "O_DIRECT", None)
    if direct_flag is None:
        return False, "Python does not expose O_DIRECT on this platform"

    probe_path = base_path / ".nixl-direct-io-probe"
    try:
        descriptor = os.open(
            probe_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR | direct_flag,
            0o600,
        )
    except OSError as error:
        unsupported_errors = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}
        if error.errno in unsupported_errors:
            return False, f"test filesystem does not support O_DIRECT: {error}"
        raise
    else:
        os.close(descriptor)
        return True, ""
    finally:
        probe_path.unlink(missing_ok=True)


def _wait_for_store_tasks(
    adapter: L2AdapterInterface,
    task_ids: set[int],
) -> None:
    """Wait until all requested public store tasks complete successfully."""
    pending = set(task_ids)
    deadline = time.monotonic() + 10.0
    while pending:
        _wait_for_fd(
            adapter.get_store_event_fd(),
            timeout=max(0.0, deadline - time.monotonic()),
        )
        completed = adapter.pop_completed_store_tasks()
        for task_id in pending & completed.keys():
            assert completed[task_id].is_successful()
        pending.difference_update(completed)


@requires_nixl_integration
@requires_nixl_extension
def test_posix_public_batch_round_trip_persistence_and_delete(
    tmp_path: Path,
) -> None:
    """Exercise FILE storage through only the public L2 adapter interface."""
    _, arena = _aligned_arena(6 * _CHUNK_SIZE)
    keys = [
        ObjectKey(
            chunk_hash=bytes.fromhex("00112233"),
            model_name="org/model",
            kv_rank=42,
            object_group_id=7,
            cache_salt="tenant",
        ),
        ObjectKey(
            chunk_hash=bytes.fromhex("44556677"),
            model_name="org/model",
            kv_rank=42,
            object_group_id=8,
        ),
        ObjectKey(
            chunk_hash=bytes.fromhex("8899aabb"),
            model_name="org/model",
            kv_rank=42,
            object_group_id=9,
        ),
    ]
    sources: list[MemoryObj] = [
        _memory_obj(arena, 0, 17),
        _memory_obj(arena, _CHUNK_SIZE, 29),
        _memory_obj(arena, 2 * _CHUNK_SIZE, 37),
    ]
    expected = [bytes(obj.byte_array) for obj in sources]

    adapter = _make_posix_adapter(tmp_path, arena)
    try:
        status = adapter.report_status()
        assert status["storage_type"] == "FILE"
        assert status["supports_delete"] is True
        assert status["atomic_publication"] is True
        task_id = adapter.submit_store_task(keys, sources)
        _wait_for_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[task_id].is_successful()
    finally:
        adapter.close()

    expected_names = {
        "org-SEP-model@0x0000002a@7@00112233@tenant.data",
        "org-SEP-model@0x0000002a@8@44556677.data",
        "org-SEP-model@0x0000002a@9@8899aabb.data",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names

    destinations: list[MemoryObj] = [
        _memory_obj(arena, 3 * _CHUNK_SIZE, 0),
        _memory_obj(arena, 4 * _CHUNK_SIZE, 0),
        _memory_obj(arena, 5 * _CHUNK_SIZE, 0),
    ]

    adapter = _make_posix_adapter(tmp_path, arena)
    try:
        lookup_id = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
        _wait_for_fd(adapter.get_lookup_and_lock_event_fd())
        lookup = adapter.query_lookup_and_lock_result(lookup_id)
        assert lookup is not None
        assert [lookup.test(index) for index in range(3)] == [True, True, True]

        load_id = adapter.submit_load_task(keys, destinations)
        _wait_for_fd(adapter.get_load_event_fd())
        loaded = adapter.query_load_result(load_id)
        assert loaded is not None
        assert [loaded.test(index) for index in range(3)] == [True, True, True]
        assert [bytes(obj.byte_array) for obj in destinations] == expected

        adapter.submit_unlock(keys)
        adapter.delete(keys)
        assert not list(tmp_path.iterdir())
    finally:
        adapter.close()


@requires_nixl_integration
@requires_nixl_extension
def test_posix_rejects_out_of_arena_and_accepts_unaligned_buffered_io(
    tmp_path: Path,
) -> None:
    """Reject foreign buffers while allowing unaligned buffered transfers."""
    _, arena = _aligned_arena(2 * _CHUNK_SIZE)
    _, foreign_arena = _aligned_arena(_CHUNK_SIZE)
    valid = _memory_obj(arena, _CHUNK_SIZE, 11)
    foreign = _memory_obj(foreign_arena, 0, 13)
    misaligned_tensor = arena[1 : 1 + _CHUNK_SIZE]
    misaligned = TensorMemoryObj(
        misaligned_tensor,
        MemoryObjMetadata(
            shape=torch.Size([_CHUNK_SIZE]),
            dtype=torch.uint8,
            address=1,
            phy_size=_CHUNK_SIZE,
            fmt=MemoryFormat.BINARY_BUFFER,
            ref_count=1,
        ),
        parent_allocator=None,
    )
    keys = [
        ObjectKey(ObjectKey.IntHash2Bytes(1), "model", 0),
        ObjectKey(ObjectKey.IntHash2Bytes(2), "model", 0),
        ObjectKey(ObjectKey.IntHash2Bytes(3), "model", 0),
    ]

    adapter = _make_posix_adapter(tmp_path, arena)
    try:
        valid_id = adapter.submit_store_task([keys[0]], [valid])
        _wait_for_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[valid_id].is_successful()

        foreign_id = adapter.submit_store_task([keys[1]], [foreign])
        _wait_for_fd(adapter.get_store_event_fd())
        assert not adapter.pop_completed_store_tasks()[foreign_id].is_successful()
        misaligned_id = adapter.submit_store_task([keys[2]], [misaligned])
        _wait_for_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[misaligned_id].is_successful()

        recovery_id = adapter.submit_store_task([keys[1]], [valid])
        _wait_for_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[recovery_id].is_successful()
        assert {path.name for path in tmp_path.iterdir()} == {
            "model@0x00000000@0@00000001.data",
            "model@0x00000000@0@00000002.data",
            "model@0x00000000@0@00000003.data",
        }
    finally:
        adapter.close()


@requires_nixl_integration
@requires_nixl_extension
def test_posix_repeated_round_trips_release_file_descriptors(
    tmp_path: Path,
) -> None:
    """Return to post-initialization fd baseline and release it on close."""
    _, arena = _aligned_arena(4 * _CHUNK_SIZE)
    keys = [
        ObjectKey(ObjectKey.IntHash2Bytes(index), "fd/model", 1, index)
        for index in range(1, 3)
    ]
    sources: list[MemoryObj] = [
        _memory_obj(arena, 0, 61),
        _memory_obj(arena, _CHUNK_SIZE, 67),
    ]
    destinations: list[MemoryObj] = [
        _memory_obj(arena, 2 * _CHUNK_SIZE, 0),
        _memory_obj(arena, 3 * _CHUNK_SIZE, 0),
    ]
    bootstrap = _make_posix_adapter(tmp_path, arena)
    bootstrap.close()
    post_initialization_baseline = len(os.listdir("/proc/self/fd"))

    adapter = _make_posix_adapter(tmp_path, arena)
    try:
        # Warm up plugin and worker paths before establishing the fd baseline.
        store_id = adapter.submit_store_task(keys, sources)
        _wait_for_store_tasks(adapter, {store_id})
        load_id = adapter.submit_load_task(keys, destinations)
        _wait_for_fd(adapter.get_load_event_fd())
        loaded = adapter.query_load_result(load_id)
        assert loaded is not None
        assert all(loaded.test(index) for index in range(len(keys)))
        initialized_baseline = len(os.listdir("/proc/self/fd"))

        for _ in range(5):
            store_id = adapter.submit_store_task(keys, sources)
            _wait_for_store_tasks(adapter, {store_id})
            load_id = adapter.submit_load_task(keys, destinations)
            _wait_for_fd(adapter.get_load_event_fd())
            loaded = adapter.query_load_result(load_id)
            assert loaded is not None
            assert all(loaded.test(index) for index in range(len(keys)))
            assert len(os.listdir("/proc/self/fd")) == initialized_baseline
    finally:
        adapter.close()

    assert len(os.listdir("/proc/self/fd")) == post_initialization_baseline
    assert not list(tmp_path.glob("*.tmp.*"))


@requires_nixl_integration
@requires_nixl_extension
def test_posix_concurrent_thread_atomic_publication(tmp_path: Path) -> None:
    """Publish one complete file from concurrent threads sharing an adapter."""
    _, arena = _aligned_arena(2 * _CHUNK_SIZE)
    key = ObjectKey(ObjectKey.IntHash2Bytes(77), "shared/model", 3, 5)
    fill_values = (71, 83)
    sources: list[MemoryObj] = [
        _memory_obj(arena, index * _CHUNK_SIZE, fill_value)
        for index, fill_value in enumerate(fill_values)
    ]
    adapter = _make_posix_adapter(tmp_path, arena, num_workers=2)
    final_path = tmp_path / "shared-SEP-model@0x00000003@5@0000004d.data"
    stop_observer = threading.Event()
    observed_contents: list[bytes] = []
    start = threading.Barrier(len(sources))
    task_ids: list[int] = []
    task_ids_lock = threading.Lock()

    def submit(source: MemoryObj) -> None:
        """Submit one store when both test threads reach the barrier."""
        start.wait(timeout=10.0)
        task_id = adapter.submit_store_task([key], [source])
        with task_ids_lock:
            task_ids.append(task_id)

    def observe_publication() -> None:
        """Record every visible final-file state during publication."""
        while not stop_observer.wait(0.0005):
            try:
                observed_contents.append(final_path.read_bytes())
            except FileNotFoundError:
                pass

    threads = [threading.Thread(target=submit, args=(source,)) for source in sources]
    observer = threading.Thread(target=observe_publication)
    try:
        observer.start()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive()
        assert len(task_ids) == len(sources)
        _wait_for_store_tasks(adapter, set(task_ids))
    finally:
        stop_observer.set()
        observer.join(timeout=10.0)
        assert not observer.is_alive()
        adapter.close()

    paths = list(tmp_path.iterdir())
    assert [path.name for path in paths] == [
        "shared-SEP-model@0x00000003@5@0000004d.data"
    ]
    contents = paths[0].read_bytes()
    assert len(contents) == _CHUNK_SIZE
    complete_contents = {bytes([value]) * _CHUNK_SIZE for value in fill_values}
    assert contents in complete_contents
    assert all(observed in complete_contents for observed in observed_contents)
    assert not list(tmp_path.glob("*.tmp.*"))


@requires_nixl_integration
@requires_nixl_extension
def test_posix_direct_io_batch_round_trip(tmp_path: Path) -> None:
    """Round-trip a path-mode batch with direct I/O when supported."""
    supported, reason = _supports_direct_io(tmp_path)
    if not supported:
        pytest.skip(reason)

    _, arena = _aligned_arena(4 * _CHUNK_SIZE)
    keys = [
        ObjectKey(ObjectKey.IntHash2Bytes(index), "direct/model", 2, index)
        for index in range(1, 3)
    ]
    sources: list[MemoryObj] = [
        _memory_obj(arena, 0, 101),
        _memory_obj(arena, _CHUNK_SIZE, 103),
    ]
    destinations: list[MemoryObj] = [
        _memory_obj(arena, 2 * _CHUNK_SIZE, 0),
        _memory_obj(arena, 3 * _CHUNK_SIZE, 0),
    ]
    expected = [bytes(source.byte_array) for source in sources]
    adapter = _make_posix_adapter(tmp_path, arena, use_direct_io=True)
    try:
        store_id = adapter.submit_store_task(keys, sources)
        _wait_for_store_tasks(adapter, {store_id})
        load_id = adapter.submit_load_task(keys, destinations)
        _wait_for_fd(adapter.get_load_event_fd())
        loaded = adapter.query_load_result(load_id)
        assert loaded is not None
        assert all(loaded.test(index) for index in range(len(keys)))
        assert [bytes(obj.byte_array) for obj in destinations] == expected
    finally:
        adapter.close()

    assert not list(tmp_path.glob("*.tmp.*"))


@requires_nixl_integration
@requires_nixl_extension
def test_failed_plugin_initialization_rolls_back_resources(tmp_path: Path) -> None:
    """Repeated plugin-discovery failures leak no descriptors or files."""
    _, arena = _aligned_arena(_CHUNK_SIZE)
    descriptor = L1MemoryDesc(
        ptr=arena.data_ptr(),
        size=arena.numel(),
        align_bytes=_ALIGNMENT,
    )
    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "NO_SUCH_NIXL_BACKEND",
            "backend_params": {
                "file_path": str(tmp_path),
                "use_direct_io": "false",
            },
            "num_workers": 2,
        }
    )
    descriptor_count = len(os.listdir("/proc/self/fd"))

    for _ in range(3):
        with pytest.raises(RuntimeError, match="plugin discovery"):
            create_l2_adapter(config, descriptor)

    assert len(os.listdir("/proc/self/fd")) == descriptor_count
    assert not list(tmp_path.iterdir())


@requires_nixl_object_integration
@requires_nixl_extension
def test_object_public_round_trip_and_capabilities() -> None:
    """Exercise whole-object store, lookup, and load through the public API."""
    endpoint = os.environ.get("LMCACHE_NIXL_OBJECT_ENDPOINT")
    bucket = os.environ.get("LMCACHE_NIXL_OBJECT_BUCKET")
    if not endpoint or not bucket:
        pytest.skip("OBJ integration requires endpoint and bucket environment values")

    _, arena = _aligned_arena(4 * _CHUNK_SIZE)
    keys = [
        ObjectKey(
            chunk_hash=bytes.fromhex("00112233"),
            model_name="lmcache/nixl-native-integration",
            kv_rank=42,
            object_group_id=7,
            cache_salt="tenant",
        ),
        ObjectKey(
            chunk_hash=bytes.fromhex("44556677"),
            model_name="lmcache/nixl-native-integration",
            kv_rank=42,
            object_group_id=8,
        ),
    ]
    sources: list[MemoryObj] = [
        _memory_obj(arena, 0, 41),
        _memory_obj(arena, _CHUNK_SIZE, 53),
    ]
    expected = [bytes(obj.byte_array) for obj in sources]
    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "OBJ",
            "backend_params": {
                "bucket": bucket,
                "endpoint_override": endpoint,
                "scheme": os.environ.get("LMCACHE_NIXL_OBJECT_SCHEME", "https"),
                "use_virtual_addressing": "false",
            },
            "num_workers": 1,
        }
    )
    descriptor = L1MemoryDesc(
        ptr=arena.data_ptr(),
        size=arena.numel(),
        align_bytes=_ALIGNMENT,
    )
    adapter = create_l2_adapter(config, descriptor)
    try:
        status = adapter.report_status()
        assert status["storage_type"] == "OBJECT"
        assert status["supports_delete"] is False
        assert status["atomic_publication"] is False
        assert "backend_params" not in status

        store_id = adapter.submit_store_task(keys, sources)
        _wait_for_fd(adapter.get_store_event_fd(), timeout=30.0)
        assert adapter.pop_completed_store_tasks()[store_id].is_successful()

        missing = ObjectKey(
            chunk_hash=uuid.uuid4().bytes,
            model_name="lmcache/nixl-native-missing",
            kv_rank=42,
            object_group_id=9,
        )
        lookup_id = adapter.submit_lookup_and_lock_task(
            [*keys, missing], {0: _EMPTY_LAYOUT}
        )
        _wait_for_fd(adapter.get_lookup_and_lock_event_fd(), timeout=30.0)
        lookup = adapter.query_lookup_and_lock_result(lookup_id)
        assert lookup is not None
        assert [lookup.test(index) for index in range(3)] == [True, True, False]

        destinations: list[MemoryObj] = [
            _memory_obj(arena, 2 * _CHUNK_SIZE, 0),
            _memory_obj(arena, 3 * _CHUNK_SIZE, 0),
        ]
        load_id = adapter.submit_load_task(keys, destinations)
        _wait_for_fd(adapter.get_load_event_fd(), timeout=30.0)
        loaded = adapter.query_load_result(load_id)
        assert loaded is not None
        assert [loaded.test(index) for index in range(2)] == [True, True]
        assert bytes(destinations[0].byte_array) == expected[0]
        assert bytes(destinations[1].byte_array) == expected[1]
        adapter.submit_unlock(keys)
    finally:
        adapter.close()
