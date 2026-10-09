# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm GTT L1 host memory backend (config and manager)."""

# Standard
from typing import Any
import ctypes
import json
import os
import time
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed import config as config_module
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    get_arg_parser,
    l1_exposes_single_memory_region,
    parse_args_to_config,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
    get_l2_adapter_config_class,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.memory_manager import l1_memory_manager
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.platform import current_device_spec

GIB = 1 << 30
MIB = 1 << 20
PAGE = os.sysconf("SC_PAGE_SIZE")

requires_rocm = pytest.mark.skipif(
    current_device_spec.backend_name != "rocm", reason="gtt backend is ROCm only"
)


@pytest.fixture
def gtt_env(monkeypatch):
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")


class _FakeGtt:
    """Fake GTT segment source over page-aligned ctypes buffers."""

    def __init__(self) -> None:
        self.allocated: list[int] = []
        self.freed: list[int] = []
        self._bufs: dict[int, ctypes.Array] = {}

    def alloc(self, size: int) -> int:
        buf = (ctypes.c_uint8 * (size + 4 * PAGE))()
        ptr = -(-ctypes.addressof(buf) // (4 * PAGE)) * (4 * PAGE)
        self._bufs[ptr] = buf
        self.allocated.append(ptr)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        del self._bufs[ptr]


@pytest.fixture
def fake_gtt(monkeypatch) -> _FakeGtt:
    fake = _FakeGtt()
    monkeypatch.setattr(l1_memory_manager, "alloc_gtt_segment", fake.alloc)
    monkeypatch.setattr(l1_memory_manager, "free_gtt_segment", fake.free)
    return fake


def _gtt_config(**overrides) -> L1MemoryManagerConfig:
    kwargs: dict[str, Any] = dict(
        size_in_bytes=4 * MIB,
        use_lazy=True,
        init_size_in_bytes=MIB,
        align_bytes=4 * PAGE,
        shm_name="",
        host_memory_backend="gtt",
    )
    kwargs.update(overrides)
    return L1MemoryManagerConfig(**kwargs)


def _storage_config(memory_config, adapters=()) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(memory_config=memory_config),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=list(adapters)),
    )


def _fs_config(adapter_type: str, path, use_odirect: bool):
    return get_l2_adapter_config_class(adapter_type).from_dict(
        {"base_path": str(path), "use_odirect": use_odirect}
    )


# Config ---------------------------------------------------------------------


def test_registered_is_the_default():
    cfg = L1MemoryManagerConfig(size_in_bytes=GIB, use_lazy=True)
    assert cfg.host_memory_backend == "registered"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="registered' or 'gtt"):
        L1MemoryManagerConfig(GIB, True, host_memory_backend="userptr")  # type: ignore[arg-type]


def test_gtt_rejected_off_rocm(monkeypatch, gtt_env):
    monkeypatch.setattr(
        config_module,
        "current_device_spec",
        types.SimpleNamespace(backend_name="cuda", is_pin_supported=True),
    )
    with pytest.raises(ValueError, match="only supported on ROCm"):
        _gtt_config()


@requires_rocm
def test_env_checked_before_any_allocation(monkeypatch, fake_gtt):
    for value in (None, "1"):
        if value is None:
            monkeypatch.delenv("HSA_USERPTR_FOR_PAGED_MEM", raising=False)
        else:
            monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", value)
        with pytest.raises(RuntimeError, match="HSA_USERPTR_FOR_PAGED_MEM=0"):
            L1MemoryManager(_gtt_config())
    assert fake_gtt.allocated == []
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")
    _gtt_config()


@requires_rocm
def test_gtt_requires_lazy(gtt_env):
    with pytest.raises(ValueError, match="requires --l1-use-lazy"):
        _gtt_config(use_lazy=False)


@requires_rocm
def test_legacy_cli(gtt_env):
    parser = get_arg_parser()
    args = parser.parse_args(
        [
            "--l1-size-gb",
            "1000",
            "--l1-init-size-gb",
            "600",
            "--eviction-policy",
            "LRU",
            "--l1-host-memory-backend",
            "gtt",
        ]
    )
    memory = parse_args_to_config(args).l1_manager_config.memory_config
    assert memory.host_memory_backend == "gtt"
    assert memory.init_size_in_bytes == 600 * GIB


def test_legacy_cli_default_unchanged():
    args = get_arg_parser().parse_args(
        ["--l1-size-gb", "8", "--eviction-policy", "LRU"]
    )
    memory = parse_args_to_config(args).l1_manager_config.memory_config
    assert memory.host_memory_backend == "registered"


@requires_rocm
def test_l1_manager_json(gtt_env):
    parser = get_arg_parser()
    raw = {"type": "DRAM", "tag": "dram", "size_gb": 1000, "init_size_gb": 1}
    args = parser.parse_args(
        [
            "--l1-manager",
            json.dumps({**raw, "host_memory_backend": "gtt"}),
            "--eviction-policy",
            "LRU",
        ]
    )
    memory = parse_args_to_config(args).l1_manager_config.memory_config
    assert memory.host_memory_backend == "gtt"

    with pytest.raises(ValueError, match="legacy L1 flags"):
        parse_args_to_config(
            parser.parse_args(
                [
                    "--l1-manager",
                    json.dumps(raw),
                    "--eviction-policy",
                    "LRU",
                    "--l1-host-memory-backend",
                    "gtt",
                ]
            )
        )


@requires_rocm
def test_single_region_consumers_rejected_at_config(monkeypatch, gtt_env):
    adapter = MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1)
    config = _storage_config(_gtt_config(), [adapter])
    assert l1_exposes_single_memory_region(config) is False

    monkeypatch.setattr(
        config_module, "requires_single_l1_memory_region", lambda _c: "nixl_store"
    )
    with pytest.raises(ValueError, match="registers L1 memory"):
        _storage_config(_gtt_config(), [adapter])


@requires_rocm
@pytest.mark.parametrize("adapter_type", ["fs", "fs_native"])
def test_direct_io_l2_rejected_with_gtt(gtt_env, tmp_path, adapter_type):
    with pytest.raises(ValueError, match=f"L2 adapter {adapter_type} registers"):
        _storage_config(_gtt_config(), [_fs_config(adapter_type, tmp_path, True)])
    _storage_config(_gtt_config(), [_fs_config(adapter_type, tmp_path, False)])


def test_direct_io_l2_still_allowed_with_registered_l1(tmp_path):
    _storage_config(
        L1MemoryManagerConfig(size_in_bytes=GIB, use_lazy=True),
        [_fs_config("fs_native", tmp_path, True)],
    )


# Manager --------------------------------------------------------------------


def _wait_capacity(manager: Any, total: int) -> None:
    deadline = time.time() + 10
    while manager.get_memory_usage()[1] < total and time.time() < deadline:
        time.sleep(0.01)
    assert manager.get_memory_usage()[1] == total


@requires_rocm
def test_manager_uses_segments_without_registration(monkeypatch, gtt_env, fake_gtt):
    pins: list[int] = []
    monkeypatch.setattr(
        current_device_spec, "pin_memory", lambda ptr, size, flags=0: pins.append(ptr)
    )
    # The initial 1 MiB and the background 3 MiB are separate segments.
    manager = L1MemoryManager(_gtt_config())
    _wait_capacity(manager, 4 * MIB)
    assert manager.get_l1_memory_desc() is None
    assert manager.memory_region_count() == 2

    layout = MemoryLayoutDesc(shapes=[torch.Size([MIB // 2])], dtypes=[torch.uint8])
    _, objs = manager.allocate(layout, 8)
    assert len(objs) == 8
    assert manager.get_memory_usage() == (4 * MIB, 4 * MIB)
    manager.free(objs)
    assert manager.get_memory_usage() == (0, 4 * MIB)
    manager.close()
    assert sorted(fake_gtt.freed) == sorted(fake_gtt.allocated)
    assert pins == []


@requires_rocm
def test_runtime_add_l2_adapter_rejected_with_gtt(gtt_env, fake_gtt, tmp_path):
    storage_manager = StorageManager(_storage_config(_gtt_config()))
    try:
        with pytest.raises(ValueError, match="L2 adapter fs registers"):
            storage_manager.add_l2_adapter(_fs_config("fs", tmp_path, True))
        assert storage_manager.add_l2_adapter(_fs_config("fs", tmp_path, False)) >= 0
    finally:
        storage_manager.close()
    assert sorted(fake_gtt.freed) == sorted(fake_gtt.allocated)


def test_registered_backend_keeps_lazy_allocator(fake_gtt):
    manager = L1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=128 * MIB, use_lazy=True, init_size_in_bytes=64 * MIB
        )
    )
    try:
        assert manager.memory_region_count() == 1
        desc = manager.get_l1_memory_desc()
        assert desc is not None and desc.size == 128 * MIB
    finally:
        manager.close()
    assert fake_gtt.allocated == []
