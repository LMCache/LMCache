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
from lmcache.v1.platform import current_device_spec
from lmcache.v1.platform.devices.rocm.gtt_host_memory import (
    GTT_MAX_ALLOCATION_BYTES,
    GttEnvironmentError,
)

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


def _gtt_config(**overrides) -> L1MemoryManagerConfig:
    kwargs: dict[str, Any] = dict(
        size_in_bytes=4 * MIB,
        use_lazy=True,
        init_size_in_bytes=MIB,
        align_bytes=4 * PAGE,
        shm_name="",
        host_memory_backend="gtt",
        gtt_segment_size_in_bytes=MIB,
    )
    kwargs.update(overrides)
    return L1MemoryManagerConfig(**kwargs)


def _storage_config(memory_config, adapters=()) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(memory_config=memory_config),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=list(adapters)),
    )


# Config ---------------------------------------------------------------------


def test_registered_is_the_default():
    cfg = L1MemoryManagerConfig(size_in_bytes=GIB, use_lazy=True)
    assert cfg.host_memory_backend == "registered"
    assert cfg.gtt_segment_size_in_bytes == 0


def test_unknown_backend_and_stray_segment_size_rejected():
    with pytest.raises(ValueError, match="registered' or 'gtt"):
        L1MemoryManagerConfig(GIB, True, host_memory_backend="userptr")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires host_memory_backend 'gtt'"):
        L1MemoryManagerConfig(GIB, True, gtt_segment_size_in_bytes=GIB)


def test_gtt_rejected_off_rocm(monkeypatch, gtt_env):
    monkeypatch.setattr(
        config_module,
        "current_device_spec",
        types.SimpleNamespace(backend_name="cuda", is_pin_supported=True),
    )
    with pytest.raises(ValueError, match="only supported on ROCm"):
        _gtt_config()


@requires_rocm
def test_gtt_env_preflight(monkeypatch):
    monkeypatch.delenv("HSA_USERPTR_FOR_PAGED_MEM", raising=False)
    with pytest.raises(GttEnvironmentError, match="HSA_USERPTR_FOR_PAGED_MEM"):
        _gtt_config()
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "1")
    with pytest.raises(GttEnvironmentError):
        _gtt_config()
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")
    _gtt_config()


@requires_rocm
def test_env_preflight_happens_before_any_allocation(monkeypatch):
    fake = _FakeGtt()
    monkeypatch.setattr(l1_memory_manager, "alloc_gtt_segment", fake.alloc)
    monkeypatch.delenv("HSA_USERPTR_FOR_PAGED_MEM", raising=False)
    with pytest.raises(GttEnvironmentError):
        L1MemoryManager(_gtt_config())
    assert fake.allocated == []


@requires_rocm
def test_gtt_segment_size_default_and_bounds(gtt_env):
    cfg = _gtt_config(
        size_in_bytes=1000 * GIB,
        init_size_in_bytes=GIB,
        align_bytes=16384,
        gtt_segment_size_in_bytes=0,
    )
    assert cfg.gtt_segment_size_in_bytes == GTT_MAX_ALLOCATION_BYTES - 16384
    with pytest.raises(ValueError, match="512 GiB"):
        _gtt_config(gtt_segment_size_in_bytes=GTT_MAX_ALLOCATION_BYTES)
    with pytest.raises(ValueError, match="multiple"):
        _gtt_config(gtt_segment_size_in_bytes=MIB + PAGE)


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
            "--l1-align-bytes",
            "16384",
            "--eviction-policy",
            "LRU",
            "--l1-host-memory-backend",
            "gtt",
        ]
    )
    memory = parse_args_to_config(args).l1_manager_config.memory_config
    assert memory.host_memory_backend == "gtt"
    assert memory.gtt_segment_size_in_bytes == GTT_MAX_ALLOCATION_BYTES - 16384
    assert memory.init_size_in_bytes == 600 * GIB

    args = parser.parse_args(
        [
            "--l1-size-gb",
            "100",
            "--eviction-policy",
            "LRU",
            "--l1-host-memory-backend",
            "gtt",
            "--l1-gtt-segment-size-gb",
            "64",
        ]
    )
    memory = parse_args_to_config(args).l1_manager_config.memory_config
    assert memory.gtt_segment_size_in_bytes == 64 * GIB

    args = parser.parse_args(
        [
            "--l1-size-gb",
            "100",
            "--eviction-policy",
            "LRU",
            "--l1-gtt-segment-size-gb",
            "512",
        ]
    )
    with pytest.raises(ValueError, match="requires host_memory_backend 'gtt'"):
        parse_args_to_config(args)


def test_legacy_cli_default_unchanged():
    args = get_arg_parser().parse_args(
        ["--l1-size-gb", "8", "--eviction-policy", "LRU"]
    )
    memory = parse_args_to_config(args).l1_manager_config.memory_config
    assert memory.host_memory_backend == "registered"
    assert memory.gtt_segment_size_in_bytes == 0


@requires_rocm
def test_l1_manager_json(gtt_env):
    parser = get_arg_parser()
    raw = {
        "type": "DRAM",
        "tag": "dram",
        "size_gb": 1000,
        "init_size_gb": 1,
        "align_bytes": 16384,
        "host_memory_backend": "gtt",
        "gtt_segment_size_gb": 256,
    }
    args = parser.parse_args(
        ["--l1-manager", json.dumps(raw), "--eviction-policy", "LRU"]
    )
    memory = parse_args_to_config(args).l1_manager_config.memory_config
    assert memory.host_memory_backend == "gtt"
    assert memory.gtt_segment_size_in_bytes == 256 * GIB

    raw["gtt_segment_size_gb"] = 512
    args = parser.parse_args(
        ["--l1-manager", json.dumps(raw), "--eviction-policy", "LRU"]
    )
    with pytest.raises(ValueError, match="512 GiB"):
        parse_args_to_config(args)

    with pytest.raises(ValueError, match="legacy L1 flags"):
        parse_args_to_config(
            parser.parse_args(
                [
                    "--l1-manager",
                    json.dumps({"type": "DRAM", "tag": "d", "size_gb": 1}),
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
    with pytest.raises(ValueError, match="nixl_store registers the L1 buffer"):
        _storage_config(_gtt_config(), [adapter])


@requires_rocm
@pytest.mark.parametrize("adapter_type", ["fs", "fs_native"])
def test_direct_io_l2_rejected_with_gtt(gtt_env, tmp_path, adapter_type):
    direct = get_l2_adapter_config_class(adapter_type).from_dict(
        {"base_path": str(tmp_path), "use_odirect": True}
    )
    with pytest.raises(ValueError, match=f"{adapter_type} does O_DIRECT"):
        _storage_config(_gtt_config(), [direct])
    buffered = get_l2_adapter_config_class(adapter_type).from_dict(
        {"base_path": str(tmp_path), "use_odirect": False}
    )
    _storage_config(_gtt_config(), [buffered])


def test_direct_io_l2_still_allowed_with_registered_l1(tmp_path):
    direct = get_l2_adapter_config_class("fs_native").from_dict(
        {"base_path": str(tmp_path), "use_odirect": True}
    )
    _storage_config(L1MemoryManagerConfig(size_in_bytes=GIB, use_lazy=True), [direct])


# Manager --------------------------------------------------------------------


@requires_rocm
def test_manager_uses_segments_without_registration(monkeypatch, gtt_env):
    fake = _FakeGtt()
    pins: list[int] = []
    monkeypatch.setattr(l1_memory_manager, "alloc_gtt_segment", fake.alloc)
    monkeypatch.setattr(l1_memory_manager, "free_gtt_segment", fake.free)
    monkeypatch.setattr(
        current_device_spec, "pin_memory", lambda ptr, size, flags=0: pins.append(ptr)
    )
    manager = L1MemoryManager(_gtt_config())
    deadline = time.time() + 10
    while manager.get_memory_usage()[1] < 4 * MIB and time.time() < deadline:
        time.sleep(0.01)
    assert manager.get_memory_usage() == (0, 4 * MIB)
    assert manager.get_l1_memory_desc() is None
    segments = manager.get_l1_memory_segments()
    assert [d.size for d in segments] == [MIB] * 4
    assert manager.memory_region_count() == 4

    layout = MemoryLayoutDesc(shapes=[torch.Size([MIB // 2])], dtypes=[torch.uint8])
    err, objs = manager.allocate(layout, 8)
    assert len(objs) == 8
    assert manager.get_memory_usage() == (4 * MIB, 4 * MIB)
    manager.free(objs)
    assert manager.get_memory_usage() == (0, 4 * MIB)
    manager.close()
    assert sorted(fake.freed) == sorted(fake.allocated) and len(fake.freed) == 4
    assert pins == []


def test_registered_backend_keeps_lazy_allocator(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(l1_memory_manager, "alloc_gtt_segment", calls.append)
    manager = L1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=128 * MIB, use_lazy=True, init_size_in_bytes=64 * MIB
        )
    )
    try:
        assert manager.memory_region_count() == 1
        desc = manager.get_l1_memory_desc()
        assert desc is not None and desc.size == 128 * MIB
        assert manager.get_l1_memory_segments() == [desc]
    finally:
        manager.close()
    assert calls == []
