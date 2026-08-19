# SPDX-License-Identifier: Apache-2.0
"""Tests for Device-DAX-backed L1 allocation.

The tests use a regular mmap-able file rather than requiring real
``/dev/dax`` hardware. That exercises the allocator contract and storage
manager wiring while keeping CI portable.
"""

# Standard
from pathlib import Path
from typing import Any, cast
import argparse
import gc
import json
import os
import stat

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
    requires_single_l1_memory_region,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.dax_l2_adapter import (
    DaxDeviceConfig,
    DaxL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.fault_inject_l2_adapter import (
    FaultInjectL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.reconfiguration import (
    L1ReconfigureError,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxMemoryAllocator,
    DevDaxRemoveMode,
)
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


def _open_fd_count(path: str) -> int:
    """Count this process's open descriptors that resolve to ``path``."""
    target = os.path.realpath(path)
    count = 0
    for name in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{name}") == target:
                count += 1
        except OSError:
            continue
    return count


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

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        self._runtime.register_calls.append((ptr, size, flags))
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


_FAKE_DAX_MAJOR = 511


def _fake_char_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    devices: dict[str, dict[str, Any]],
) -> list[str]:
    """Make descriptors opened on ``devices`` impersonate character devices.

    ``devices`` maps a mapping-file path to ``{"minor": int, "subsystem":
    str | None, "size": int | None, "align": int | None, "mode": int,
    "expose": str}``. ``mode`` defaults to ``stat.S_IFCHR``. ``expose``
    selects how the fake sysfs lists the device: ``"char"`` (default) creates
    a ``char/<major>:<minor>`` symlink like the kernel's ``/sys/dev/char``,
    ``"bus"`` creates only a ``bus/dax/<name>`` entry with a ``dev``
    attribute like ``/sys/bus/dax/devices``, ``"both"`` creates both, and
    ``"none"`` (or ``subsystem=None``) exposes nothing. Two paths given the
    same minor impersonate two nodes of one device.

    Returns:
        Canonical paths whose opened descriptors were reported as fake
        character devices.
    """
    sysfs_root = tmp_path / "sysfs"
    char_root = sysfs_root / "char"
    char_root.mkdir(parents=True, exist_ok=True)
    dax_bus_root = sysfs_root / "bus" / "dax"
    dax_bus_root.mkdir(parents=True, exist_ok=True)
    node_by_path: dict[str, tuple[int, int]] = {}
    for device_path, spec in devices.items():
        rdev = os.makedev(_FAKE_DAX_MAJOR, spec["minor"])
        canonical_path = os.path.realpath(device_path)
        node = (
            rdev,
            spec.get("mode", stat.S_IFCHR),
        )
        node_by_path[canonical_path] = node
        subsystem = spec.get("subsystem")
        expose = spec.get("expose", "char")
        char_link = char_root / f"{os.major(rdev)}:{os.minor(rdev)}"
        if subsystem is None or expose == "none" or char_link.exists():
            continue
        name = os.path.basename(device_path)
        device_dir = sysfs_root / "devices" / name
        device_dir.mkdir(parents=True)
        bus_dir = sysfs_root / "bus" / subsystem
        bus_dir.mkdir(parents=True, exist_ok=True)
        (device_dir / "subsystem").symlink_to(bus_dir)
        (device_dir / "dev").write_text(f"{os.major(rdev)}:{os.minor(rdev)}")
        for attribute in ("size", "align"):
            if spec.get(attribute) is not None:
                (device_dir / attribute).write_text(str(spec[attribute]))
        if expose in ("bus", "both"):
            (dax_bus_root / name).symlink_to(device_dir)
        if expose in ("char", "both"):
            char_link.symlink_to(device_dir)
    monkeypatch.setattr(
        "lmcache.v1.memory_allocators.devdax_memory_allocator._SYSFS_CHAR_ROOT",
        str(char_root),
    )
    monkeypatch.setattr(
        "lmcache.v1.memory_allocators.devdax_memory_allocator._DAX_BUS_ROOT",
        str(dax_bus_root),
    )
    real_stat = os.stat
    real_fstat = os.fstat
    fake_fstat_calls: list[str] = []

    def fake_device_status(
        path_status: os.stat_result,
        node: tuple[int, int],
    ) -> os.stat_result:
        rdev, mode = node
        fields = list(path_status)[:10]
        fields[stat.ST_MODE] = mode | 0o600
        fields[stat.ST_SIZE] = 0
        return os.stat_result(fields, {"st_rdev": rdev})

    def character_device_stat(
        path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        path_status = real_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if isinstance(path, int) or dir_fd is not None or not follow_symlinks:
            return path_status
        canonical_path = os.path.realpath(path)
        if isinstance(canonical_path, bytes):
            canonical_path = os.fsdecode(canonical_path)
        node = node_by_path.get(canonical_path)
        if node is None:
            return path_status
        return fake_device_status(path_status, node)

    def character_device_fstat(fd: int) -> os.stat_result:
        # Only descriptors opened on the listed mapping files impersonate a
        # DAX character device; every other caller keeps the real result.
        fd_status = real_fstat(fd)
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            return fd_status
        node = node_by_path.get(target)
        if node is None:
            return fd_status
        fake_fstat_calls.append(target)
        return fake_device_status(fd_status, node)

    monkeypatch.setattr(
        "lmcache.v1.memory_allocators.devdax_memory_allocator.os.stat",
        character_device_stat,
    )
    monkeypatch.setattr(
        "lmcache.v1.memory_allocators.devdax_memory_allocator.os.fstat",
        character_device_fstat,
    )
    return fake_fstat_calls


def _single_arena_allocator(primary: str) -> DevDaxMemoryAllocator:
    return DevDaxMemoryAllocator(size=4096, device_path=primary, align_bytes=4096)


def test_character_device_uses_dax_sysfs_capacity_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A character device is checked against the DAX sysfs entry found by its
    device number, not by its path name."""
    primary = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    capacity_limited = _make_mmap_file(tmp_path, size=8192, name="dax0.1")
    # Named unlike a DAX node on purpose: the sysfs lookup must not depend on
    # the basename.
    strictly_aligned = _make_mmap_file(tmp_path, size=8192, name="by-id-link")
    _fake_char_devices(
        tmp_path,
        monkeypatch,
        {
            primary: {"minor": 0, "subsystem": "dax", "size": 8192, "align": 4096},
            capacity_limited: {
                "minor": 1,
                "subsystem": "dax",
                "size": 4096,
                "align": 4096,
            },
            strictly_aligned: {
                "minor": 2,
                "subsystem": "dax",
                "size": 8192,
                "align": 8192,
            },
        },
    )

    allocator = _single_arena_allocator(primary)
    try:
        with pytest.raises(RuntimeError, match="exceeds.*capacity"):
            allocator.add_device(capacity_limited, 8192)
        with pytest.raises(ValueError, match="multiple of the device alignment"):
            allocator.add_device(strictly_aligned, 4096)
        # Both rejections happen before mmap: the pool is untouched and the
        # descriptor opened for validation is released.
        assert [s.device_path for s in allocator.arena_statuses()] == [primary]
        assert _open_fd_count(capacity_limited) == 0
        assert _open_fd_count(strictly_aligned) == 0
    finally:
        allocator.close()


def test_character_device_alias_uses_device_number_for_runtime_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second node with the same device number resolves to one arena."""
    primary = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    extra = _make_mmap_file(tmp_path, size=8192, name="dax0.1")
    alias = _make_mmap_file(tmp_path, size=8192, name="char-511-1")
    fake_fstat_calls = _fake_char_devices(
        tmp_path,
        monkeypatch,
        {
            primary: {"minor": 0, "subsystem": "dax", "size": 8192, "align": 4096},
            extra: {"minor": 1, "subsystem": "dax", "size": 8192, "align": 4096},
            alias: {"minor": 1, "subsystem": "dax", "size": 8192, "align": 4096},
        },
    )

    allocator = _single_arena_allocator(primary)
    try:
        allocator.add_device(extra, 4096)
        assert allocator.arena_status(alias).device_path == extra

        calls_before = len(fake_fstat_calls)
        with pytest.raises(ValueError, match="already mapped"):
            allocator.add_device(alias, 4096)
        assert len(fake_fstat_calls) == calls_before

        removed = allocator.remove_device(alias, DevDaxRemoveMode.DRAIN)
        assert removed.device_path == extra
        assert removed.state == DevDaxArenaState.REMOVED
        assert [status.device_path for status in allocator.arena_statuses()] == [
            primary
        ]
    finally:
        allocator.close()


def test_non_dax_character_device_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A character device outside the dax subsystem (e.g. /dev/zero) fails."""
    primary = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    zero_like = _make_mmap_file(tmp_path, size=8192, name="zero")
    _fake_char_devices(
        tmp_path,
        monkeypatch,
        {
            primary: {"minor": 0, "subsystem": "dax", "size": 8192, "align": 4096},
            zero_like: {"minor": 5, "subsystem": "mem"},
        },
    )

    allocator = _single_arena_allocator(primary)
    try:
        with pytest.raises(ValueError, match="not a Device-DAX device"):
            allocator.add_device(zero_like, 4096)
        assert [s.device_path for s in allocator.arena_statuses()] == [primary]
        assert _open_fd_count(zero_like) == 0
    finally:
        allocator.close()


def test_block_device_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only character devices and regular files can back an arena."""
    primary = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    disk = _make_mmap_file(tmp_path, size=8192, name="sdb")
    _fake_char_devices(
        tmp_path,
        monkeypatch,
        {
            primary: {"minor": 0, "subsystem": "dax", "size": 8192, "align": 4096},
            disk: {"minor": 16, "subsystem": None, "mode": stat.S_IFBLK},
        },
    )

    allocator = _single_arena_allocator(primary)
    try:
        with pytest.raises(ValueError, match="neither a character device"):
            allocator.add_device(disk, 4096)
        assert [s.device_path for s in allocator.arena_statuses()] == [primary]
        assert _open_fd_count(disk) == 0
    finally:
        allocator.close()


def test_character_device_without_sysfs_entry_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A character device that sysfs does not expose cannot be verified."""
    primary = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    unlisted = _make_mmap_file(tmp_path, size=8192, name="dax9.9")
    _fake_char_devices(
        tmp_path,
        monkeypatch,
        {
            primary: {"minor": 0, "subsystem": "dax", "size": 8192, "align": 4096},
            unlisted: {"minor": 9, "subsystem": "dax", "expose": "none"},
        },
    )

    allocator = _single_arena_allocator(primary)
    try:
        with pytest.raises(ValueError, match="cannot verify"):
            allocator.add_device(unlisted, 4096)
        assert [s.device_path for s in allocator.arena_statuses()] == [primary]
        assert _open_fd_count(unlisted) == 0
    finally:
        allocator.close()


def test_character_device_resolves_through_dax_bus_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With /sys/dev/char masked, the dax bus listing identifies the device."""
    primary = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    bus_only = _make_mmap_file(tmp_path, size=8192, name="dax0.3")
    _fake_char_devices(
        tmp_path,
        monkeypatch,
        {
            primary: {"minor": 0, "subsystem": "dax", "size": 8192, "align": 4096},
            bus_only: {
                "minor": 3,
                "subsystem": "dax",
                "size": 4096,
                "align": 4096,
                "expose": "bus",
            },
        },
    )

    allocator = _single_arena_allocator(primary)
    try:
        # The size attribute is read through the bus entry: 8192 > 4096.
        with pytest.raises(RuntimeError, match="exceeds.*capacity"):
            allocator.add_device(bus_only, 8192)
        added = allocator.add_device(bus_only, 4096)
        assert added.device_path == bus_only
        assert added.state == DevDaxArenaState.ACTIVE
    finally:
        allocator.close()


def test_character_device_with_unreadable_dax_attributes_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dax device whose align or size cannot be read is refused."""
    primary = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    opaque = _make_mmap_file(tmp_path, size=8192, name="dax0.4")
    _fake_char_devices(
        tmp_path,
        monkeypatch,
        {
            primary: {"minor": 0, "subsystem": "dax", "size": 8192, "align": 4096},
            opaque: {"minor": 4, "subsystem": "dax"},
        },
    )

    allocator = _single_arena_allocator(primary)
    try:
        with pytest.raises(ValueError, match="unreadable or zero"):
            allocator.add_device(opaque, 4096)
        assert [s.device_path for s in allocator.arena_statuses()] == [primary]
        assert _open_fd_count(opaque) == 0
    finally:
        allocator.close()


@pytest.mark.parametrize(
    ("zero_attribute", "expected_message"),
    [("align", "align=0, size=8192"), ("size", "align=4096, size=0")],
)
def test_character_device_with_zero_dax_attribute_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zero_attribute: str,
    expected_message: str,
) -> None:
    """Readable but zero DAX alignment or capacity is unverifiable."""
    primary = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    invalid = _make_mmap_file(tmp_path, size=8192, name="dax0.7")
    invalid_spec = {
        "minor": 7,
        "subsystem": "dax",
        "size": 8192,
        "align": 4096,
    }
    invalid_spec[zero_attribute] = 0
    _fake_char_devices(
        tmp_path,
        monkeypatch,
        {
            primary: {"minor": 0, "subsystem": "dax", "size": 8192, "align": 4096},
            invalid: invalid_spec,
        },
    )

    allocator = _single_arena_allocator(primary)
    try:
        with pytest.raises(ValueError, match=expected_message):
            allocator.add_device(invalid, 4096)
        assert [status.device_path for status in allocator.arena_statuses()] == [
            primary
        ]
        assert _open_fd_count(invalid) == 0
    finally:
        allocator.close()


def test_regular_file_never_consults_sysfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regular file named like a DAX node ignores a conflicting sysfs entry.

    Guards against reintroducing basename-keyed sysfs lookups: the base code
    already gated sysfs on the character-device type, so this passes there too.
    """
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    lookalike = _make_mmap_file(tmp_path, size=8192, name="dax0.0")
    sysfs_root = tmp_path / "sysfs"
    device_dir = sysfs_root / "devices" / "dax0.0"
    device_dir.mkdir(parents=True)
    # Either attribute would reject an 8192-byte mapping if it were consulted.
    (device_dir / "size").write_text("4096")
    (device_dir / "align").write_text("8192")
    (device_dir / "dev").write_text(f"{_FAKE_DAX_MAJOR}:0")
    dax_bus_root = sysfs_root / "bus" / "dax"
    dax_bus_root.mkdir(parents=True)
    (dax_bus_root / "dax0.0").symlink_to(device_dir)
    char_root = sysfs_root / "char"
    char_root.mkdir()
    (char_root / f"{_FAKE_DAX_MAJOR}:0").symlink_to(device_dir)
    monkeypatch.setattr(
        "lmcache.v1.memory_allocators.devdax_memory_allocator._SYSFS_CHAR_ROOT",
        str(char_root),
    )
    monkeypatch.setattr(
        "lmcache.v1.memory_allocators.devdax_memory_allocator._DAX_BUS_ROOT",
        str(dax_bus_root),
    )

    allocator = _single_arena_allocator(primary)
    try:
        added = allocator.add_device(lookalike, 8192)
        assert added.size_in_bytes == 8192
    finally:
        allocator.close()


def test_devdax_allocator_registers_cuda_host_mapping(tmp_path, monkeypatch):
    path = _make_mmap_file(tmp_path)
    cuda_runtime = _FakeCudaRuntime()
    monkeypatch.setattr(memory_management, "torch_device_type", "cuda")
    monkeypatch.setattr(memory_management, "torch_dev", cuda_runtime)
    monkeypatch.setattr(memory_management, "current_device_spec", cuda_runtime.ext)

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
    monkeypatch.setattr(memory_management, "current_device_spec", cuda_runtime.ext)

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


def test_storage_manager_routes_l1_devdax_reconfigure(tmp_path: Path) -> None:
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    storage_manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4096,
                    use_lazy=False,
                    shm_name="",
                    align_bytes=4096,
                    devdax_path=primary,
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )
    )
    try:
        (primary_status,) = storage_manager.get_l1_devdax_arena_statuses()
        assert primary_status.device_path == primary
        assert primary_status.is_primary is True

        # An uninterpretable user path is a 404 lookup miss through the full
        # manager chain rather than an unhandled 500.
        with pytest.raises(L1ReconfigureError) as nul_lookup_miss:
            storage_manager.remove_l1_devdax_device(
                "bad\x00path", DevDaxRemoveMode.DRAIN
            )
        assert nul_lookup_miss.value.status_code == 404

        added = storage_manager.add_l1_devdax_device(extra, 4096)
        assert added.device_path == extra
        assert added.state == DevDaxArenaState.ACTIVE

        removed = storage_manager.remove_l1_devdax_device(extra, DevDaxRemoveMode.DRAIN)
        assert removed.device_path == extra
        assert removed.state == DevDaxArenaState.REMOVED

        # Once unmapped the path is unknown: the allocator lookup miss is
        # translated into the HTTP-mappable 404, not a 409 state conflict.
        with pytest.raises(L1ReconfigureError) as lookup_miss:
            storage_manager.remove_l1_devdax_device(extra, DevDaxRemoveMode.DRAIN)
        assert lookup_miss.value.status_code == 404
    finally:
        storage_manager.close()


def test_storage_manager_l1_devdax_reconfigure_rejects_cpu_l1() -> None:
    storage_manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4096,
                    use_lazy=False,
                    shm_name="",
                    align_bytes=4096,
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )
    )
    try:
        with pytest.raises(L1ReconfigureError, match="Device-DAX"):
            storage_manager.get_l1_devdax_arena_statuses()
        with pytest.raises(L1ReconfigureError, match="not Device-DAX backed"):
            storage_manager.add_l1_devdax_device("unused", 4096)
    finally:
        storage_manager.close()


def _pure_devdax_storage_manager(
    primary: str, adapters: list[L2AdapterConfigBase] | None = None
) -> StorageManager:
    return StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4096,
                    use_lazy=False,
                    shm_name="",
                    align_bytes=4096,
                    devdax_path=primary,
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=adapters or []),
        )
    )


def _mock_l2_config() -> MockL2AdapterConfig:
    return MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)


_SINGLE_REGION_PREDICATE = (
    "lmcache.v1.distributed.storage_manager.requires_single_l1_memory_region"
)


def test_storage_manager_rejects_l1_devdax_add_with_single_region_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing the last single-region adapter permits runtime arena add."""
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    # The mock adapter stands in for NIXL, which needs a transfer engine.
    monkeypatch.setattr(_SINGLE_REGION_PREDICATE, lambda _config: "nixl_store")
    storage_manager = _pure_devdax_storage_manager(primary, [_mock_l2_config()])
    try:
        with pytest.raises(L1ReconfigureError, match="nixl_store") as rejected:
            storage_manager.add_l1_devdax_device(extra, 4096)
        assert rejected.value.status_code == 409
        statuses = storage_manager.get_l1_devdax_arena_statuses()
        assert [s.device_path for s in statuses] == [primary]
        assert _open_fd_count(extra) == 0

        storage_manager.delete_l2_adapter(0)
        added = storage_manager.add_l1_devdax_device(extra, 4096)
        assert added.device_path == extra
        assert added.state == DevDaxArenaState.ACTIVE
    finally:
        storage_manager.close()


def test_storage_manager_rejects_l1_add_of_l2_dax_device_alias(
    tmp_path: Path,
) -> None:
    """L1 cannot map a physical device already owned by DAX L2."""
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    l2_device = _make_mmap_file(tmp_path, size=4096, name="l2-device.bin")
    l1_alias = tmp_path / "l1-alias.bin"
    os.link(l2_device, l1_alias)
    dax_config = DaxL2AdapterConfig(
        devices=[
            DaxDeviceConfig(
                device_path=l2_device,
                max_dax_size_gb=4096 / (1024**3),
            )
        ],
        slot_bytes=4096,
    )
    storage_manager = _pure_devdax_storage_manager(primary, [dax_config])
    try:
        with pytest.raises(L1ReconfigureError) as conflict:
            storage_manager.add_l1_devdax_device(str(l1_alias), 4096)
        assert conflict.value.status_code == 409
        assert "already mapped by L2" in str(conflict.value)
        statuses = storage_manager.get_l1_devdax_arena_statuses()
        assert [status.device_path for status in statuses] == [primary]
    finally:
        storage_manager.close()


def test_storage_manager_rejects_single_region_adapter_with_runtime_arenas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror check: with two arenas mapped a NIXL-like adapter is refused
    until L1 is back to a single arena."""
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    storage_manager = _pure_devdax_storage_manager(primary)
    try:
        storage_manager.add_l1_devdax_device(extra, 4096)
        monkeypatch.setattr(_SINGLE_REGION_PREDICATE, lambda _config: "nixl_store")
        with pytest.raises(ValueError, match="nixl_store"):
            storage_manager.add_l2_adapter(_mock_l2_config())

        removed = storage_manager.remove_l1_devdax_device(extra, DevDaxRemoveMode.DRAIN)
        assert removed.state == DevDaxArenaState.REMOVED
        adapter_id = storage_manager.add_l2_adapter(_mock_l2_config())
        assert adapter_id >= 0
    finally:
        storage_manager.close()


def _wrapped_in_fault_inject(inner: L2AdapterConfigBase) -> FaultInjectL2AdapterConfig:
    return FaultInjectL2AdapterConfig(
        inner_config=inner, rate=0.0, seed=0, gap_indices=()
    )


def _type_name_nixl_unless_wrapper(config: object) -> str:
    return (
        "fault_inject"
        if isinstance(config, FaultInjectL2AdapterConfig)
        else "nixl_store"
    )


def test_single_region_predicate_sees_through_wrapper_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.distributed.config.get_type_name_for_config",
        _type_name_nixl_unless_wrapper,
    )
    wrapper = _wrapped_in_fault_inject(_mock_l2_config())
    assert requires_single_l1_memory_region(wrapper) == "nixl_store"


def test_storage_manager_rejects_single_region_adapter_while_arena_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draining arena holding a live object still counts as a region."""
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    storage_manager = _pure_devdax_storage_manager(primary)
    try:
        key_a, key_b = _key(1), _key(2)
        # A fills the primary arena.
        assert set(storage_manager.reserve_write([key_a], _layout(), "new")) == {key_a}
        storage_manager.finish_write([key_a])
        storage_manager.add_l1_devdax_device(extra, 4096)
        # B lands on the extra arena and stays write-reserved, so it can be
        # neither evicted nor freed while the arena drains.
        pending = storage_manager.reserve_write([key_b], _layout(), "new")
        assert set(pending) == {key_b}
        draining = storage_manager.remove_l1_devdax_device(
            extra, DevDaxRemoveMode.DRAIN
        )
        assert draining.state == DevDaxArenaState.DRAINING
        assert draining.active_allocations == 1

        monkeypatch.setattr(_SINGLE_REGION_PREDICATE, lambda _config: "nixl_store")
        with pytest.raises(ValueError, match="2 regions"):
            storage_manager.add_l2_adapter(_mock_l2_config())

        storage_manager.finish_write([key_b])
        del pending
        storage_manager.delete_l1_keys([key_b])
        gc.collect()
        statuses = storage_manager.get_l1_devdax_arena_statuses()
        assert [s.device_path for s in statuses] == [primary]
        assert storage_manager.add_l2_adapter(_mock_l2_config()) >= 0
    finally:
        storage_manager.close()


def test_storage_manager_rejects_single_region_adapter_on_hybrid_l1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hybrid DRAM + Device-DAX is already two regions with a single arena."""
    path = _make_mmap_file(tmp_path)
    storage_manager = StorageManager(
        StorageManagerConfig(
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
            l2_adapter_config=L2AdaptersConfig(adapters=[]),
        )
    )
    try:
        monkeypatch.setattr(_SINGLE_REGION_PREDICATE, lambda _config: "nixl_store")
        with pytest.raises(ValueError, match="2 regions"):
            storage_manager.add_l2_adapter(_mock_l2_config())
    finally:
        storage_manager.close()


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


def _pure_devdax_manager(path: str, size: int = 4096) -> DevDaxL1MemoryManager:
    """Build a pure Device-DAX L1 manager whose single arena has ``size`` bytes."""
    return DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=size,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
            devdax_path=path,
        )
    )


def test_add_device_serves_overflow_after_primary_full(tmp_path):
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    manager = _pure_devdax_manager(primary)

    error, first = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS
    assert len(first) == 1

    # The primary arena is full, so the next allocation fails until we grow.
    error, spilled = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.OUT_OF_MEMORY
    assert spilled == []

    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    status = manager.add_device(extra, 4096)
    assert status.device_path == extra
    assert status.state == DevDaxArenaState.ACTIVE
    assert status.is_primary is False
    assert status.size_in_bytes == 4096

    error, second = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS
    assert len(second) == 1

    statuses = manager.get_arena_statuses()
    assert [status.device_path for status in statuses] == [primary, extra]
    used, total = manager.get_memory_usage()
    assert total == 8192
    assert used == 8192

    manager.free(first)
    manager.free(second)
    del first
    del second
    gc.collect()
    manager.close()


def test_remove_device_reaps_empty_arena_immediately(tmp_path):
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    manager = _pure_devdax_manager(primary)

    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    manager.add_device(extra, 4096)
    assert len(manager.get_arena_statuses()) == 2

    status = manager.remove_device(extra)
    assert status.device_path == extra
    assert status.state == DevDaxArenaState.REMOVED

    assert [status.device_path for status in manager.get_arena_statuses()] == [primary]
    used, total = manager.get_memory_usage()
    assert total == 4096
    assert used == 0
    manager.close()


def test_remove_device_drains_until_allocations_freed(tmp_path):
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    manager = _pure_devdax_manager(primary)

    # Fill the primary arena so the spilled object must land on the extra arena.
    error, first = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS

    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    manager.add_device(extra, 4096)
    error, second = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS

    status = manager.remove_device(extra)
    assert status.state == DevDaxArenaState.DRAINING
    assert status.active_allocations == 1

    # A draining arena accepts no new allocations, so with the primary full the
    # allocation fails rather than reusing the arena being retired.
    error, blocked = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.OUT_OF_MEMORY
    assert blocked == []
    assert [status.state for status in manager.get_arena_statuses()] == [
        DevDaxArenaState.ACTIVE,
        DevDaxArenaState.DRAINING,
    ]

    # Freeing the arena's last allocation unmaps it automatically.
    manager.free(second)
    del second
    gc.collect()
    assert [status.device_path for status in manager.get_arena_statuses()] == [primary]

    manager.free(first)
    del first
    gc.collect()
    manager.close()


def test_draining_arena_capacity_excluded_from_total(tmp_path):
    # A draining arena's free space is not usable headroom, so its capacity must
    # drop out of the total the moment it starts draining; its live bytes still
    # count as used. Otherwise the eviction watermark (used / total) is diluted
    # by capacity that is going away and eviction never triggers.
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    manager = _pure_devdax_manager(primary)

    error, first = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS

    extra = _make_mmap_file(tmp_path, size=8192, name="extra.bin")
    manager.add_device(extra, 8192)
    error, second = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS

    # Both arenas active: total counts all capacity.
    used, total = manager.get_memory_usage()
    assert (used, total) == (8192, 12288)

    status = manager.remove_device(extra)
    assert status.state == DevDaxArenaState.DRAINING

    # Draining: the extra arena's 8192 bytes leave the total, but its live 4096
    # bytes still count as used, so used now exceeds total (ratio > 1) and the
    # watermark is satisfied.
    used, total = manager.get_memory_usage()
    assert (used, total) == (8192, 4096)

    # Once the draining arena is unmapped, both totals reflect the primary only.
    manager.free(second)
    del second
    gc.collect()
    used, total = manager.get_memory_usage()
    assert (used, total) == (4096, 4096)

    manager.free(first)
    del first
    gc.collect()
    manager.close()


def test_remove_device_defers_unmap_while_external_views_alive(tmp_path):
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    manager = _pure_devdax_manager(primary)

    # Fill the primary arena so the second object lands on the extra arena.
    error, first = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS

    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    manager.add_device(extra, 4096)
    error, second = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS

    # Keep a view into the arena beyond the free, as a reader still consuming
    # the tensor would.
    lingering_view = second[0].tensor

    manager.remove_device(extra)
    manager.free(second)
    del second
    gc.collect()

    # The arena is fully drained but cannot unmap while the view is alive, so
    # it stays in the pool as DRAINING instead of crashing the free.
    statuses = manager.get_arena_statuses()
    assert [status.state for status in statuses] == [
        DevDaxArenaState.ACTIVE,
        DevDaxArenaState.DRAINING,
    ]
    assert statuses[1].active_allocations == 0

    # Once the view is gone, the next free retries and reaps the arena.
    del lingering_view
    gc.collect()
    manager.free(first)
    del first
    gc.collect()
    assert [status.device_path for status in manager.get_arena_statuses()] == [primary]
    manager.close()


def test_reap_synchronizes_device_before_unmap(tmp_path, monkeypatch):
    """A drain reap must fence the device before it unmaps an arena.

    L1 hands out raw pinned host pointers, and some GPU connectors release an
    object's pin after only a device-side stream wait (no host sync). A transfer
    reading the mapping can therefore still be in flight when the last L1
    allocation is freed. The reap must ``torch_dev.synchronize()`` before it
    unregisters/unmaps the arena, otherwise the munmap/cudaHostUnregister races
    that in-flight transfer. This asserts the ordering, and that no fence is
    wasted while the arena still has live allocations.
    """
    events: list[str] = []

    class _OrderedExt:
        is_pin_supported = True

        def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
            return True

        def unpin_memory(self, ptr: int) -> bool:
            events.append("unpin")
            return True

    class _OrderedRuntime:
        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            events.append("sync")

    monkeypatch.setattr(memory_management, "torch_device_type", "cuda")
    monkeypatch.setattr(memory_management, "torch_dev", _OrderedRuntime())
    monkeypatch.setattr(memory_management, "current_device_spec", _OrderedExt())

    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    manager = _pure_devdax_manager(primary)

    # Fill the primary so the next object lands on the removable extra arena.
    error, first = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS
    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    manager.add_device(extra, 4096)
    error, second = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS

    # Draining with a live allocation has nothing to unmap yet, so it must not
    # fence the device.
    manager.remove_device(extra)
    assert events == []

    # Freeing the arena's last allocation reaps it: fence FIRST, then unmap.
    manager.free(second)
    del second
    gc.collect()
    assert [status.device_path for status in manager.get_arena_statuses()] == [primary]
    assert events == ["sync", "unpin"], (
        f"reap must synchronize the device before unmapping; got {events}"
    )

    manager.free(first)
    del first
    gc.collect()
    manager.close()


def test_add_device_releases_mapping_when_arena_setup_fails(tmp_path, monkeypatch):
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    allocator = DevDaxMemoryAllocator(
        size=4096,
        device_path=primary,
        align_bytes=4096,
    )

    captured = {}

    def _failing_pin(self, arena):
        captured["arena"] = arena
        raise RuntimeError("pin registration failed")

    monkeypatch.setattr(DevDaxMemoryAllocator, "_register_arena_pin", _failing_pin)
    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    with pytest.raises(RuntimeError, match="pin registration failed"):
        allocator.add_device(extra, 4096)

    # The failed arena never joins the pool and its mapping is unmapped.
    assert [status.device_path for status in allocator.arena_statuses()] == [primary]
    assert captured["arena"].mmap_obj.closed
    allocator.close()


def test_remove_primary_arena_rejected(tmp_path):
    primary = _make_mmap_file(tmp_path, size=4096)
    manager = _pure_devdax_manager(primary)

    with pytest.raises(L1ReconfigureError, match="primary"):
        manager.remove_device(primary)

    # The primary arena survives the rejected removal.
    assert [status.device_path for status in manager.get_arena_statuses()] == [primary]
    manager.close()


def test_add_duplicate_device_rejected(tmp_path):
    primary = _make_mmap_file(tmp_path, size=4096, name="primary.bin")
    manager = _pure_devdax_manager(primary)

    extra = _make_mmap_file(tmp_path, size=4096, name="extra.bin")
    manager.add_device(extra, 4096)
    with pytest.raises(L1ReconfigureError, match="already mapped"):
        manager.add_device(extra, 4096)

    assert len(manager.get_arena_statuses()) == 2
    manager.close()


def test_add_device_validates_arguments(tmp_path):
    primary = _make_mmap_file(tmp_path, size=4096)
    manager = _pure_devdax_manager(primary)

    with pytest.raises(L1ReconfigureError, match="device_path"):
        manager.add_device("", 4096)
    with pytest.raises(L1ReconfigureError, match="size_in_bytes"):
        manager.add_device(str(tmp_path / "unused.bin"), 0)

    manager.close()


def test_hybrid_initial_devdax_arena_is_removable(tmp_path):
    # In hybrid mode DRAM is the primary L1 region, so the initial Device-DAX
    # arena is removable overflow rather than primary.
    path = _make_mmap_file(tmp_path, size=4096)
    manager = DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
            devdax_path=path,
            devdax_size_in_bytes=4096,
        )
    )

    statuses = manager.get_arena_statuses()
    assert len(statuses) == 1
    assert statuses[0].is_primary is False

    status = manager.remove_device(path)
    assert status.state == DevDaxArenaState.REMOVED
    assert manager.get_arena_statuses() == []

    # DRAM still serves allocations after the overflow arena is gone.
    error, objs = manager.allocate(_layout(4096), count=1)
    assert error == L1Error.SUCCESS
    manager.free(objs)
    del objs
    gc.collect()
    manager.close()


def test_allocator_batched_allocation_spans_arenas(tmp_path):
    first_path = _make_mmap_file(tmp_path, size=8192, name="arena-1.bin")
    allocator = DevDaxMemoryAllocator(
        size=8192,
        device_path=first_path,
        align_bytes=4096,
    )
    second_path = _make_mmap_file(tmp_path, size=8192, name="arena-2.bin")
    allocator.add_device(second_path, 8192)

    # Four 4096-byte slots: two from each arena.
    objs = allocator.batched_allocate(torch.Size([4096]), torch.uint8, 4)
    assert objs is not None
    assert len(objs) == 4
    used, total = allocator.get_memory_usage()
    assert total == 16384
    assert used == 16384

    # One more object cannot be satisfied; the partial attempt rolls back and
    # leaves the pool intact.
    assert allocator.batched_allocate(torch.Size([4096]), torch.uint8, 1) is None
    used_after, total_after = allocator.get_memory_usage()
    assert used_after == 16384
    assert total_after == 16384

    allocator.batched_free(objs)
    del objs
    gc.collect()
    allocator.close()


def test_hybrid_allocator_reports_per_object_medium(tmp_path):
    """DRAM fills first; overflow objects land in (and report) the DAX
    arena, so per-key medium attribution is exact."""
    path = _make_mmap_file(tmp_path)
    allocator = DevDaxMemoryAllocator(
        size=1024 * 1024,
        device_path=path,
        local_size=2 * 4096,  # DRAM pool fits exactly two objects
        shm_name=None,
        align_bytes=4096,
    )
    try:
        objs = allocator.batched_allocate(torch.Size([4096]), torch.uint8, 4)
        assert objs is not None
        media = [allocator.is_devdax_obj(obj) for obj in objs]
        assert media == [False, False, True, True]
        allocator.batched_free(objs)
        del objs
        gc.collect()
    finally:
        allocator.close()


def test_hybrid_manager_get_backend_type_reports_per_object_medium(tmp_path):
    """DevDaxL1MemoryManager.get_backend_type maps the allocator's answer onto
    the L1BackendType enum for hybrid DRAM+DAX."""
    path = _make_mmap_file(tmp_path)
    config = L1MemoryManagerConfig(
        size_in_bytes=2 * 4096,  # DRAM pool fits exactly two objects
        use_lazy=False,
        shm_name="",
        devdax_path=path,
        devdax_size_in_bytes=1024 * 1024,
    )
    manager = DevDaxL1MemoryManager(config)
    try:
        err, objs = manager.allocate(_layout(4096), 4)
        assert err == L1Error.SUCCESS
        backends = [manager.get_backend_type(obj) for obj in objs]
        assert backends == [
            L1BackendType.DRAM,
            L1BackendType.DRAM,
            L1BackendType.DEVDAX,
            L1BackendType.DEVDAX,
        ]
        manager.free(objs)
        del objs
        gc.collect()
    finally:
        manager.close()
