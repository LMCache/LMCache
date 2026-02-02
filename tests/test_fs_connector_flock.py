# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from pathlib import Path
from types import ModuleType
from typing import Any
import asyncio
import multiprocessing as mp
import sys
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector._file_lock import lock_path_for_file
from lmcache.v1.storage_backend.connector.fs_connector import FSConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from tests.v1.utils import create_test_memory_obj

try:
    # Standard
    import fcntl as _fcntl  # noqa: F401
except ImportError:  # pragma: no cover - non-Linux platforms
    fcntl: ModuleType | None = None
else:
    fcntl = _fcntl


def _create_test_key(key_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


def _create_test_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def _build_fs_connector(base_dir: str) -> tuple[FSConnector, asyncio.AbstractEventLoop]:
    config = LMCacheEngineConfig.from_defaults(chunk_size=256)
    metadata = _create_test_metadata()
    local_cpu_backend = LocalCPUBackend(
        config,
        metadata,
        memory_allocator=AdHocMemoryAllocator(device="cpu"),
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    connector = FSConnector(
        base_paths_str=base_dir,
        loop=loop,
        local_cpu_backend=local_cpu_backend,
        config=config,
    )
    return connector, loop


def _fill_payload(memory_obj, value: int) -> None:
    raw_tensor = memory_obj.raw_tensor
    if raw_tensor is not None:
        raw_tensor.fill_(value)
        return

    buffer = memory_obj.byte_array
    if isinstance(buffer, memoryview):
        try:
            buffer_cast = buffer.cast("B")
            buffer_cast[:] = bytes([value]) * len(buffer_cast)
        except (TypeError, NotImplementedError):
            buffer[:] = bytes([value]) * len(buffer)
    else:
        for i in range(len(buffer)):
            buffer[i] = value


def _put_worker(
    base_dir: str,
    key_id: int,
    payload_value: int,
    barrier: Any | None,
    started_event: Any | None,
    error_queue: mp.Queue,
) -> None:
    try:
        connector, loop = _build_fs_connector(base_dir)
        memory_obj = create_test_memory_obj()
        _fill_payload(memory_obj, payload_value)

        if started_event is not None:
            started_event.set()

        if barrier is not None:
            barrier.wait(timeout=20.0)

        loop.run_until_complete(connector.put(_create_test_key(key_id), memory_obj))
    except Exception as exc:
        error_queue.put(repr(exc))
        raise SystemExit(1) from exc
    finally:
        try:
            loop.close()
        except Exception:
            pass
        asyncio.set_event_loop(None)


@pytest.mark.skipif(
    sys.platform != "linux" or fcntl is None, reason="flock requires Linux fcntl"
)
def test_external_lock_blocks_put(tmp_path: Path) -> None:
    base_dir = str(tmp_path)
    key_id = 123
    connector, loop = _build_fs_connector(base_dir)
    file_path = connector._get_file_path(_create_test_key(key_id))
    lock_path = lock_path_for_file(file_path)
    loop.close()
    asyncio.set_event_loop(None)

    ctx = mp.get_context("spawn")
    started_event = ctx.Event()
    error_queue: mp.Queue = ctx.Queue()

    assert fcntl is not None
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        process = ctx.Process(
            target=_put_worker,
            args=(base_dir, key_id, 11, None, started_event, error_queue),
        )
        process.start()

        started = started_event.wait(timeout=15.0)
        if not started:
            process.terminate()
            process.join(timeout=5.0)
            errors = []
            while not error_queue.empty():
                errors.append(error_queue.get())
            pytest.fail(f"child did not start in time; errors={errors}")
        time.sleep(0.3)
        assert process.is_alive()

    process.join(timeout=5.0)
    assert process.exitcode == 0
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    assert not errors
    assert file_path.exists()


@pytest.mark.skipif(
    sys.platform != "linux" or fcntl is None, reason="flock requires Linux fcntl"
)
def test_concurrent_puts_same_key(tmp_path: Path) -> None:
    base_dir = str(tmp_path)
    key_id = 456

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    error_queue: mp.Queue = ctx.Queue()

    process_a = ctx.Process(
        target=_put_worker,
        args=(base_dir, key_id, 17, barrier, None, error_queue),
    )
    process_b = ctx.Process(
        target=_put_worker,
        args=(base_dir, key_id, 33, barrier, None, error_queue),
    )

    process_a.start()
    process_b.start()

    process_a.join(timeout=30.0)
    process_b.join(timeout=30.0)

    if process_a.is_alive():
        process_a.terminate()
        process_a.join(timeout=5.0)
    if process_b.is_alive():
        process_b.terminate()
        process_b.join(timeout=5.0)

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())

    assert process_a.exitcode == 0
    assert process_b.exitcode == 0
    assert not errors

    connector, loop = _build_fs_connector(base_dir)
    file_path = connector._get_file_path(_create_test_key(key_id))
    assert file_path.exists()

    memory_obj = loop.run_until_complete(connector.get(_create_test_key(key_id)))
    loop.close()
    asyncio.set_event_loop(None)
    assert memory_obj is not None
    raw_tensor = memory_obj.raw_tensor
    if raw_tensor is not None:
        first_byte = int(raw_tensor[0].item())
    else:
        buffer = memory_obj.byte_array
        if isinstance(buffer, memoryview):
            first_byte = buffer.cast("B")[0]
        else:
            first_byte = buffer[0]
    assert first_byte in (17, 33)
