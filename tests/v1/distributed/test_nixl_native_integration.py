# SPDX-License-Identifier: Apache-2.0
"""Opt-in public-interface integration tests for native NIXL storage."""

# Standard
from pathlib import Path
from typing import Any
import importlib.util
import multiprocessing
import os
import select
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
) -> L2AdapterInterface:
    """Create a POSIX native NIXL adapter for an existing L1 arena."""
    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "POSIX",
            "backend_params": {
                "file_path": str(base_path),
                "use_direct_io": "false",
            },
            "num_workers": 1,
        }
    )
    descriptor = L1MemoryDesc(
        ptr=arena.data_ptr(),
        size=arena.numel(),
        align_bytes=_ALIGNMENT,
    )
    return create_l2_adapter(config, descriptor)


def _store_shared_posix_key(
    base_path: str,
    fill_value: int,
    ready_queue: Any,
    start_event: Any,
    result_queue: Any,
) -> None:
    """Store one shared key from a spawned connector process."""
    adapter: L2AdapterInterface | None = None
    try:
        _, arena = _aligned_arena(_CHUNK_SIZE)
        source = _memory_obj(arena, 0, fill_value)
        key = ObjectKey(ObjectKey.IntHash2Bytes(77), "shared/model", 3, 5)
        adapter = _make_posix_adapter(Path(base_path), arena)
        ready_queue.put(True)
        if not start_event.wait(10.0):
            raise RuntimeError("timed out waiting for concurrent store start")
        task_id = adapter.submit_store_task([key], [source])
        _wait_for_fd(adapter.get_store_event_fd())
        result_queue.put(
            (fill_value, adapter.pop_completed_store_tasks()[task_id].is_successful())
        )
    except BaseException as error:
        result_queue.put((fill_value, repr(error)))
    finally:
        if adapter is not None:
            adapter.close()


@requires_nixl_integration
@requires_nixl_extension
def test_posix_public_round_trip_persistence_and_mixed_load(
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
        "org--model_0000002a_7_00112233@tenant.bin",
        "org--model_0000002a_8_44556677.bin",
        "org--model_0000002a_9_8899aabb.bin",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names

    truncated_path = tmp_path / "org--model_0000002a_9_8899aabb.bin"
    truncated_path.write_bytes(b"truncated")
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
        assert [loaded.test(index) for index in range(3)] == [True, True, False]
        assert bytes(destinations[0].byte_array) == expected[0]
        assert bytes(destinations[1].byte_array) == expected[1]
        assert bytes(destinations[2].byte_array) == bytes(_CHUNK_SIZE)

        adapter.submit_unlock(keys)
        adapter.delete(keys)
        assert not list(tmp_path.iterdir())
    finally:
        adapter.close()


@requires_nixl_integration
@requires_nixl_extension
def test_posix_rejects_out_of_arena_and_misaligned_buffers(
    tmp_path: Path,
) -> None:
    """Reject invalid buffers through normal asynchronous store results."""
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

        for key, invalid in zip(keys[1:], [foreign, misaligned], strict=True):
            task_id = adapter.submit_store_task([key], [invalid])
            _wait_for_fd(adapter.get_store_event_fd())
            assert not adapter.pop_completed_store_tasks()[task_id].is_successful()

        recovery_id = adapter.submit_store_task([keys[1]], [valid])
        _wait_for_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[recovery_id].is_successful()
        assert {path.name for path in tmp_path.iterdir()} == {
            "model_00000000_0_00000001.bin",
            "model_00000000_0_00000002.bin",
        }
    finally:
        adapter.close()
        adapter.close()


@requires_nixl_integration
@requires_nixl_extension
def test_posix_cross_process_atomic_publication(tmp_path: Path) -> None:
    """Publish one key concurrently from two independent connector processes."""
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    fill_values = (71, 83)
    processes = [
        context.Process(
            target=_store_shared_posix_key,
            args=(
                str(tmp_path),
                fill_value,
                ready_queue,
                start_event,
                result_queue,
            ),
        )
        for fill_value in fill_values
    ]

    for process in processes:
        process.start()
    for _ in processes:
        assert ready_queue.get(timeout=20.0) is True
    start_event.set()
    results = [result_queue.get(timeout=20.0) for _ in processes]
    for process in processes:
        process.join(timeout=20.0)
        assert process.exitcode == 0
    assert sorted(results) == [(71, True), (83, True)]

    paths = list(tmp_path.iterdir())
    assert [path.name for path in paths] == ["shared--model_00000003_5_0000004d.bin"]
    contents = paths[0].read_bytes()
    assert len(contents) == _CHUNK_SIZE
    assert contents in {bytes([value]) * _CHUNK_SIZE for value in fill_values}


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
