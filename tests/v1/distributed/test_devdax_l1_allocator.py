# SPDX-License-Identifier: Apache-2.0
"""Tests for Device-DAX-backed L1 allocation.

The tests use a regular mmap-able file rather than requiring real
``/dev/dax`` hardware. That exercises the allocator contract and storage
manager wiring while keeping CI portable.
"""

# Standard
import gc
import os

import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    L1MemoryManagerConfig,
    parse_args,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.memory_management import DevDaxMemoryAllocator
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext


def _make_mmap_file(
    tmp_path, size: int = 4 * 1024 * 1024, name: str = "l1-devdax-test.bin"
) -> str:
    path = tmp_path / name
    with open(path, "wb") as f:
        f.truncate(size)
    return str(path)


def _key(seed: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=seed.to_bytes(4, "big") + b"\0" * 28,
        model_name="devdax-l1-test",
        kv_rank=0,
    )


def _layout(num_bytes: int = 4096) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([num_bytes])], dtypes=[torch.uint8])


def test_devdax_config_disables_lazy_and_shm(tmp_path):
    path = _make_mmap_file(tmp_path)

    cfg = L1MemoryManagerConfig(
        size_in_bytes=1024 * 1024,
        use_lazy=True,
        shm_name="lmcache_l1_pool_test",
        devdax_path=path,
    )

    assert cfg.devdax_path == path
    assert cfg.use_lazy is False
    assert cfg.shm_name == ""


def test_devdax_overflow_config_disables_lazy_and_shm(tmp_path):
    path = _make_mmap_file(tmp_path)

    cfg = L1MemoryManagerConfig(
        size_in_bytes=1024 * 1024,
        use_lazy=True,
        shm_name="lmcache_l1_pool_test",
        devdax_path=path,
        devdax_size_in_bytes=2 * 1024 * 1024,
    )

    assert cfg.devdax_path == path
    assert cfg.devdax_size_in_bytes == 2 * 1024 * 1024
    assert cfg.use_lazy is False
    assert cfg.shm_name == ""


def test_devdax_allocator_uses_mmap_backing_file(tmp_path):
    path = _make_mmap_file(tmp_path)
    allocator = DevDaxMemoryAllocator(
        size=1024 * 1024,
        device_path=path,
        align_bytes=4096,
    )

    objs = allocator.batched_allocate(torch.Size([4096]), torch.uint8, 2)
    assert objs is not None
    first = objs[0]
    assert first.data_ptr == allocator.buffer.data_ptr()
    assert first.shm_offset == 0

    first.raw_tensor.fill_(0x5A)
    allocator.batched_free(objs)
    del first
    del objs
    gc.collect()
    allocator.close()

    with open(path, "rb") as f:
        assert f.read(4096) == bytes([0x5A]) * 4096


def test_devdax_close_failure_preserves_allocator_state(tmp_path):
    path = _make_mmap_file(tmp_path)
    allocator = DevDaxMemoryAllocator(
        size=1024 * 1024,
        device_path=path,
        align_bytes=4096,
    )
    obj = allocator.allocate(torch.Size([4096]), torch.uint8)
    assert obj is not None

    with pytest.raises(BufferError):
        allocator.close()

    assert allocator.pin_allocator is not None
    assert allocator.buffer.numel() == 1024 * 1024

    allocator.free(obj)
    del obj
    gc.collect()
    allocator.close()


def test_l1_manager_round_trip_on_devdax_mapping(tmp_path):
    path = _make_mmap_file(tmp_path)
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1024 * 1024,
            use_lazy=True,
            devdax_path=path,
        )
    )
    manager = L1Manager(cfg)
    key = _key(1)

    write = manager.reserve_write([key], [False], _layout())
    assert write[key][0] == L1Error.SUCCESS
    obj = write[key][1]
    assert obj is not None
    obj.tensor.fill_(0x23)
    assert manager.finish_write([key])[key] == L1Error.SUCCESS

    read = manager.reserve_read([key])
    assert read[key][0] == L1Error.SUCCESS
    read_obj = read[key][1]
    assert read_obj is not None
    assert int(read_obj.tensor[0]) == 0x23
    assert manager.finish_read([key])[key] == L1Error.SUCCESS

    del write
    del read
    del obj
    del read_obj
    gc.collect()
    manager.close()

    with open(path, "rb") as f:
        assert f.read(1) == bytes([0x23])


def test_l1_memory_manager_spills_from_dram_to_devdax(tmp_path):
    path = _make_mmap_file(tmp_path, size=8192)
    manager = L1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=8192,
            use_lazy=False,
            align_bytes=4096,
            devdax_path=path,
            devdax_size_in_bytes=8192,
        )
    )

    error, objs = manager.allocate(_layout(4096), count=3)

    assert error == L1Error.SUCCESS
    assert len(objs) == 3
    assert objs[0].data_ptr == manager._allocator.buffer.data_ptr()
    assert objs[1].data_ptr == manager._allocator.buffer.data_ptr() + 4096
    assert objs[2].data_ptr == manager._allocator.devdax_allocator.buffer.data_ptr()
    used, total = manager.get_memory_usage()
    assert used == 3 * 4096
    assert total == 4 * 4096

    objs[2].raw_tensor.fill_(0x6D)
    manager.free(objs)
    used, total = manager.get_memory_usage()
    assert used == 0
    assert total == 4 * 4096
    manager.close()

    with open(path, "rb") as f:
        assert f.read(4096) == bytes([0x6D]) * 4096


def test_l1_memory_manager_reports_devdax_desc(tmp_path):
    path = _make_mmap_file(tmp_path)
    manager = L1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=1024 * 1024,
            use_lazy=False,
            devdax_path=path,
        )
    )

    desc = manager.get_l1_memory_desc()
    used, total = manager.get_memory_usage()

    assert desc.ptr != 0
    assert desc.size == 1024 * 1024
    assert desc.align_bytes == 4096
    assert used == 0
    assert total == 1024 * 1024
    manager.close()


def test_cli_parses_l1_devdax_path(tmp_path):
    path = _make_mmap_file(tmp_path)
    config = parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--l1-devdax-path",
            path,
        ]
    )

    mem_cfg = config.l1_manager_config.memory_config
    assert mem_cfg.devdax_path == path
    assert mem_cfg.use_lazy is False
    assert mem_cfg.shm_name == ""


def test_cli_parses_l1_devdax_overflow_size(tmp_path):
    path = _make_mmap_file(tmp_path)
    config = parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--l1-devdax-path",
            path,
            "--l1-devdax-overflow-size-gb",
            "2",
        ]
    )

    mem_cfg = config.l1_manager_config.memory_config
    assert mem_cfg.size_in_bytes == 1 << 30
    assert mem_cfg.devdax_path == path
    assert mem_cfg.devdax_size_in_bytes == 2 << 30
    assert mem_cfg.use_lazy is False
    assert mem_cfg.shm_name == ""


def test_devdax_l1_does_not_advertise_shm_pool(tmp_path):
    path = _make_mmap_file(tmp_path)
    config = parse_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--l1-devdax-path",
            path,
        ]
    )

    pool_info = MPCacheEngineContext._compute_shm_pool_info(config)

    assert pool_info == {"shm_name": "", "pool_size": 0}
    assert os.path.exists(path)
