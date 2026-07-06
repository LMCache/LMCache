# SPDX-License-Identifier: Apache-2.0
"""Tests for Device-DAX-backed L1 allocation.

The tests use a regular mmap-able file rather than requiring real
``/dev/dax`` hardware. That exercises the allocator contract and storage
manager wiring while keeping CI portable.
"""

# Standard
from typing import Any, cast
import argparse
import gc
import json
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
    get_type_name_for_config,
)
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.memory_management import DevDaxMemoryAllocator
from lmcache.v1.multiprocess.config import add_mp_server_args
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
import lmcache.v1.memory_management as memory_management


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


def _parse_mp_storage_args(args: list[str]) -> StorageManagerConfig:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    return parse_args_to_config(parser.parse_args(args))


class _FakeMooncakeL2Config:
    def __init__(self, setup_config: dict[str, str]) -> None:
        self.setup_config = setup_config


class _FakeExt:
    is_pin_supported = True

    def __init__(self, fake_runtime: "_FakeCudaRuntime") -> None:
        self._runtime = fake_runtime

    def pin_memory(self, ptr: int, size: int) -> bool:
        self._runtime.register_calls.append((ptr, size, 0))
        return self._runtime.register_error == 0

    def unpin_memory(self, ptr: int) -> bool:
        self._runtime.unregister_calls.append(ptr)
        return True


class _FakeCudaRuntime:
    def __init__(self, register_error: int = 0) -> None:
        self.register_error = register_error
        self.register_calls: list[tuple[int, int, int]] = []
        self.unregister_calls: list[int] = []
        self.synchronize_calls = 0
        self.ext = _FakeExt(self)

    def is_available(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def cudart(self) -> "_FakeCudaRuntime":
        return self

    def cudaHostRegister(self, ptr: int, size: int, flags: int) -> int:
        self.register_calls.append((ptr, size, flags))
        return self.register_error

    def cudaHostUnregister(self, ptr: int) -> int:
        self.unregister_calls.append(ptr)
        return 0


def _hybrid_storage_config(path: str, adapter_config: object) -> StorageManagerConfig:
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1024 * 1024,
                use_lazy=False,
                shm_name="",
                devdax_path=path,
                devdax_size_in_bytes=1024 * 1024,
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[cast(L2AdapterConfigBase, adapter_config)]
        ),
    )
    return config


def test_devdax_config_rejects_lazy_allocation(tmp_path):
    path = _make_mmap_file(tmp_path)

    with pytest.raises(ValueError, match="--no-l1-use-lazy"):
        L1MemoryManagerConfig(
            size_in_bytes=1024 * 1024,
            use_lazy=True,
            shm_name="",
            devdax_path=path,
        )


def test_devdax_config_rejects_shm(tmp_path):
    path = _make_mmap_file(tmp_path)

    with pytest.raises(ValueError, match="--shm-name"):
        L1MemoryManagerConfig(
            size_in_bytes=1024 * 1024,
            use_lazy=False,
            shm_name="lmcache_l1_pool_test",
            devdax_path=path,
            devdax_size_in_bytes=2 * 1024 * 1024,
        )


def test_devdax_config_accepts_explicit_lazy_and_shm_disable(tmp_path):
    path = _make_mmap_file(tmp_path)

    cfg = L1MemoryManagerConfig(
        size_in_bytes=1024 * 1024,
        use_lazy=False,
        shm_name="",
        devdax_path=path,
    )

    assert cfg.devdax_path == path
    assert cfg.use_lazy is False
    assert cfg.shm_name == ""


@pytest.mark.parametrize(
    ("adapter_name", "adapter_config"),
    [
        ("nixl_store", object()),
        ("nixl_store_dynamic", object()),
        ("mooncake_store", _FakeMooncakeL2Config({"protocol": "rdma"})),
    ],
)
def test_devdax_overflow_rejects_single_region_l2_adapters(
    tmp_path, monkeypatch, adapter_name, adapter_config
):
    path = _make_mmap_file(tmp_path)
    monkeypatch.setattr(
        "lmcache.v1.distributed.config.get_type_name_for_config",
        lambda _: adapter_name,
    )

    with pytest.raises(ValueError, match=adapter_name):
        _hybrid_storage_config(path, adapter_config)


def test_devdax_overflow_allows_mooncake_without_rdma(tmp_path, monkeypatch):
    path = _make_mmap_file(tmp_path)
    monkeypatch.setattr(
        "lmcache.v1.distributed.config.get_type_name_for_config",
        lambda _: "mooncake_store",
    )

    config = _hybrid_storage_config(path, _FakeMooncakeL2Config({"protocol": "tcp"}))

    assert config.l1_manager_config.memory_config.devdax_size_in_bytes == 1024 * 1024


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


def test_devdax_allocator_registers_cuda_host_mapping(tmp_path, monkeypatch):
    path = _make_mmap_file(tmp_path)
    cuda_runtime = _FakeCudaRuntime()
    monkeypatch.setattr(memory_management, "torch_device_type", "cuda")
    monkeypatch.setattr(memory_management, "torch_dev", cuda_runtime)

    allocator = DevDaxMemoryAllocator(
        size=1024 * 1024,
        device_path=path,
        align_bytes=4096,
    )
    ptr = allocator.buffer.data_ptr()

    assert cuda_runtime.register_calls == [(ptr, 1024 * 1024, 0)]
    allocator.close()
    assert cuda_runtime.unregister_calls == [ptr]


def test_devdax_allocator_falls_back_when_cuda_host_register_fails(
    tmp_path, monkeypatch
):
    path = _make_mmap_file(tmp_path)
    cuda_runtime = _FakeCudaRuntime(register_error=1)
    monkeypatch.setattr(memory_management, "torch_device_type", "cuda")
    monkeypatch.setattr(memory_management, "torch_dev", cuda_runtime)

    allocator = DevDaxMemoryAllocator(
        size=1024 * 1024,
        device_path=path,
        align_bytes=4096,
    )
    obj = allocator.allocate(torch.Size([4096]), torch.uint8)

    assert cuda_runtime.register_calls == [
        (allocator.buffer.data_ptr(), 1024 * 1024, 0)
    ]
    assert cuda_runtime.unregister_calls == []
    assert obj is not None
    allocator.free(obj)
    del obj
    gc.collect()
    allocator.close()
    assert cuda_runtime.unregister_calls == []


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

    assert allocator.devdax_allocator is not None
    assert allocator.devdax_buffer.numel() == 1024 * 1024

    allocator.free(obj)
    del obj
    gc.collect()
    allocator.close()


def test_l1_manager_round_trip_on_devdax_mapping(tmp_path):
    path = _make_mmap_file(tmp_path)
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1024 * 1024,
            use_lazy=False,
            shm_name="",
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


def test_devdax_l1_memory_manager_spills_from_dram_to_devdax(tmp_path):
    path = _make_mmap_file(tmp_path, size=8192)
    manager = DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=8192,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
            devdax_path=path,
            devdax_size_in_bytes=8192,
        )
    )

    error, objs = manager.allocate(_layout(4096), count=3)

    assert error == L1Error.SUCCESS
    assert len(objs) == 3
    assert isinstance(manager._allocator, DevDaxMemoryAllocator)
    assert manager._allocator.local_allocator is not None
    assert objs[0].parent() is manager._allocator.local_allocator
    assert objs[1].parent() is manager._allocator.local_allocator
    assert objs[2].parent() is manager._allocator
    assert objs[0].data_ptr == manager._allocator.local_allocator.buffer.data_ptr()
    assert (
        objs[1].data_ptr == manager._allocator.local_allocator.buffer.data_ptr() + 4096
    )
    assert objs[2].data_ptr == manager._allocator.devdax_buffer.data_ptr()
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


def test_devdax_l1_memory_manager_reports_devdax_desc(tmp_path):
    path = _make_mmap_file(tmp_path)
    manager = DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=1024 * 1024,
            use_lazy=False,
            shm_name="",
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
    config = _parse_mp_storage_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--no-l1-use-lazy",
            "--shm-name",
            "",
            "--l1-devdax-path",
            path,
        ]
    )

    mem_cfg = config.l1_manager_config.memory_config
    assert mem_cfg.devdax_path == path
    assert mem_cfg.use_lazy is False
    assert mem_cfg.shm_name == ""


def test_cli_rejects_devdax_l1_with_gds_l1(tmp_path):
    path = _make_mmap_file(tmp_path)

    with pytest.raises(ValueError, match="gds-l1-path"):
        _parse_mp_storage_args(
            [
                "--l1-size-gb",
                "1",
                "--eviction-policy",
                "LRU",
                "--no-l1-use-lazy",
                "--shm-name",
                "",
                "--l1-devdax-path",
                path,
                "--gds-l1-path",
                str(tmp_path),
            ]
        )


def test_cli_infers_l1_devdax_overflow_from_registered_dax_adapter(tmp_path):
    path = _make_mmap_file(tmp_path)
    config = _parse_mp_storage_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--no-l1-use-lazy",
            "--shm-name",
            "",
            "--l1-devdax-path",
            path,
            "--l2-adapter",
            ('{"type":"dax","device_path":"%s","max_dax_size_gb":2,"slot_bytes":4096}')
            % path,
        ]
    )

    mem_cfg = config.l1_manager_config.memory_config
    assert mem_cfg.size_in_bytes == 1 << 30
    assert mem_cfg.devdax_path == path
    assert mem_cfg.devdax_size_in_bytes == 2 << 30
    assert mem_cfg.use_lazy is False
    assert mem_cfg.shm_name == ""
    assert config.l2_adapter_config.adapters == []


@pytest.mark.parametrize(
    ("adapter_spec", "expected_adapter_type"),
    [
        (
            {
                "type": "raw_block",
                "device_path": "rawblock-l2.bin",
                "slot_bytes": 8192,
                "capacity_bytes": 16384,
                "meta_total_bytes": 4096,
                "use_odirect": False,
                "meta_enable_periodic": False,
                "load_checkpoint_on_init": False,
                "meta_verify_on_load": False,
            },
            "raw_block",
        ),
    ],
)
def test_cli_hybrid_l1_keeps_ordinary_l2_adapters(
    tmp_path, adapter_spec, expected_adapter_type
):
    path = _make_mmap_file(tmp_path)
    adapter_spec = {
        key: str(tmp_path / value) if key in ("base_path", "device_path") else value
        for key, value in adapter_spec.items()
    }

    config = _parse_mp_storage_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--no-l1-use-lazy",
            "--shm-name",
            "",
            "--l1-devdax-path",
            path,
            "--l2-adapter",
            json.dumps(
                {
                    "type": "dax",
                    "device_path": path,
                    "max_dax_size_gb": 2,
                    "slot_bytes": 4096,
                }
            ),
            "--l2-adapter",
            json.dumps(adapter_spec),
        ]
    )

    mem_cfg = config.l1_manager_config.memory_config
    assert mem_cfg.devdax_size_in_bytes == 2 << 30
    assert len(config.l2_adapter_config.adapters) == 1
    assert (
        get_type_name_for_config(config.l2_adapter_config.adapters[0])
        == expected_adapter_type
    )


def test_cli_hybrid_l1_splits_matching_dax_device_and_keeps_other_l2(tmp_path):
    l1_dax_path = _make_mmap_file(tmp_path, name="l1-devdax.bin")
    l2_dax_path = _make_mmap_file(tmp_path, name="l2-devdax.bin")

    config = _parse_mp_storage_args(
        [
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--no-l1-use-lazy",
            "--shm-name",
            "",
            "--l1-devdax-path",
            l1_dax_path,
            "--l2-adapter",
            json.dumps(
                {
                    "type": "dax",
                    "devices": [
                        {"device_path": l1_dax_path, "max_dax_size_gb": 2},
                        {"device_path": l2_dax_path, "max_dax_size_gb": 3},
                    ],
                    "slot_bytes": 4096,
                    "hotplug_enabled": True,
                    "num_store_workers": 2,
                    "num_lookup_workers": 3,
                    "num_load_workers": 4,
                }
            ),
        ]
    )

    mem_cfg = config.l1_manager_config.memory_config
    assert mem_cfg.devdax_size_in_bytes == 2 << 30
    assert len(config.l2_adapter_config.adapters) == 1

    dax_adapter = cast(Any, config.l2_adapter_config.adapters[0])
    assert get_type_name_for_config(dax_adapter) == "dax"
    assert [device.device_path for device in dax_adapter.devices] == [l2_dax_path]
    assert dax_adapter.max_dax_size_gb == 3
    assert dax_adapter.hotplug_enabled is True
    assert dax_adapter.num_store_workers == 2
    assert dax_adapter.num_lookup_workers == 3
    assert dax_adapter.num_load_workers == 4


def test_devdax_l1_does_not_advertise_shm_pool(tmp_path):
    path = _make_mmap_file(tmp_path)
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1024 * 1024,
                use_lazy=False,
                shm_name="",
                devdax_path=path,
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    context = MPCacheServerContext(config)

    try:
        assert context.shm_pool_info == {"shm_name": "", "pool_size": 0}
        assert os.path.exists(path)
    finally:
        context.storage_manager.close()
