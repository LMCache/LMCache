# SPDX-License-Identifier: Apache-2.0
"""Public L1 allocation behavior with a simulated HugeTLB native allocator."""

# Standard
from pathlib import Path
from types import SimpleNamespace
import ctypes

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_allocators.devdax_memory_allocator import DevDaxMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
import lmcache.v1.distributed.config as config_module
import lmcache.v1.memory_management as memory_management

HUGEPAGE_SIZE = 2 << 20
LOGICAL_SIZE = HUGEPAGE_SIZE + 4096


@pytest.fixture
def native_hugepages(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Simulate native mappings while recording their rounded owned lengths."""
    state = SimpleNamespace(mapped=[], freed=[], buffers=[])

    def allocate_buffer(mapped_size: int) -> int:
        buffer = ctypes.create_string_buffer(mapped_size + HUGEPAGE_SIZE)
        ptr = (ctypes.addressof(buffer) + HUGEPAGE_SIZE - 1) & ~(HUGEPAGE_SIZE - 1)
        state.buffers.append(buffer)
        state.mapped.append((ptr, mapped_size))
        return ptr

    def allocate(size: int, flags: int) -> int:
        mapped_size = (size + HUGEPAGE_SIZE - 1) // HUGEPAGE_SIZE * HUGEPAGE_SIZE
        return allocate_buffer(mapped_size)

    def allocate_plain(size: int, flags: int) -> int:
        return allocate_buffer(size)

    def free(ptr: int, size: int) -> None:
        mapped_size = (size + HUGEPAGE_SIZE - 1) // HUGEPAGE_SIZE * HUGEPAGE_SIZE
        state.freed.append((ptr, mapped_size))

    def free_plain(ptr: int) -> None:
        state.freed.append((ptr, LOGICAL_SIZE))

    monkeypatch.setattr(
        config_module,
        "current_device_spec",
        SimpleNamespace(device_type="cuda", is_pin_supported=True),
    )
    monkeypatch.setattr(config_module.sys, "platform", "linux")
    monkeypatch.setattr(
        config_module,
        "import_module",
        lambda name: SimpleNamespace(
            alloc_hugepage_pinned_ptr=allocate,
            free_hugepage_pinned_ptr=free,
            alloc_hugepage_pinned_numa_ptr=allocate,
            free_hugepage_pinned_numa_ptr=free,
        ),
    )
    monkeypatch.setattr(
        memory_management.device_ops, "alloc_hugepage_pinned_ptr", allocate
    )
    monkeypatch.setattr(memory_management.device_ops, "free_hugepage_pinned_ptr", free)
    monkeypatch.setattr(
        memory_management.device_ops, "alloc_pinned_ptr", allocate_plain
    )
    monkeypatch.setattr(memory_management.device_ops, "free_pinned_ptr", free_plain)
    monkeypatch.setattr(
        memory_management,
        "current_device_spec",
        SimpleNamespace(is_pin_supported=False),
    )
    monkeypatch.setattr(memory_management.torch_dev, "is_available", lambda: False)
    return state


def _config(
    use_hugepages: bool,
    devdax_path: str | None = None,
    devdax_size_in_bytes: int = 0,
) -> L1MemoryManagerConfig:
    """Create an eager anonymous L1 config for the public manager tests."""
    return L1MemoryManagerConfig(
        size_in_bytes=LOGICAL_SIZE,
        use_hugepages=use_hugepages,
        use_lazy=False,
        shm_name="",
        devdax_path=devdax_path,
        devdax_size_in_bytes=devdax_size_in_bytes,
    )


def _layout() -> MemoryLayoutDesc:
    """Describe one 4 KiB tensor object."""
    return MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])


@pytest.mark.parametrize("use_hugepages", [False, True])
def test_regular_l1_capacity_and_reuse(
    native_hugepages: SimpleNamespace, use_hugepages: bool
) -> None:
    """The manager exposes only logical bytes and reuses freed aligned objects."""
    manager = L1MemoryManager(_config(use_hugepages))
    desc = manager.get_l1_memory_desc()
    assert desc.size == LOGICAL_SIZE
    assert manager.get_memory_usage() == (0, LOGICAL_SIZE)
    mapped_size = 2 * HUGEPAGE_SIZE if use_hugepages else LOGICAL_SIZE
    assert native_hugepages.mapped == [(desc.ptr, mapped_size)]

    error, objects = manager.allocate(_layout(), count=LOGICAL_SIZE // 4096)
    assert error == L1Error.SUCCESS
    assert len(objects) == LOGICAL_SIZE // 4096
    assert all(obj.data_ptr % 4096 == 0 for obj in objects)
    assert all(obj.data_ptr < desc.ptr + LOGICAL_SIZE for obj in objects)
    tensor = objects[0].raw_tensor
    assert tensor is not None
    tensor.fill_(0x5A)
    assert tensor[0].item() == 0x5A
    assert manager.get_backend_type(objects[0]) == L1BackendType.DRAM

    error, extra = manager.allocate(_layout(), count=1)
    assert (error, extra) == (L1Error.OUT_OF_MEMORY, [])
    assert manager.free(objects) == L1Error.SUCCESS
    error, partial = manager.allocate(_layout(), count=LOGICAL_SIZE // 4096 + 1)
    assert (error, partial) == (L1Error.OUT_OF_MEMORY, [])
    assert manager.get_memory_usage() == (0, LOGICAL_SIZE)
    error, reused = manager.allocate(_layout(), count=1)
    assert error == L1Error.SUCCESS
    assert reused[0].data_ptr == desc.ptr
    manager.free(reused)
    manager.close()
    manager.close()
    assert native_hugepages.freed == [(desc.ptr, mapped_size)]


@pytest.mark.parametrize("use_hugepages", [False, True])
def test_hybrid_l1_keeps_dram_and_devdax_backends(
    native_hugepages: SimpleNamespace, tmp_path: Path, use_hugepages: bool
) -> None:
    """The hybrid DRAM pool uses hugepages while overflow stays on Device-DAX."""
    path = tmp_path / "devdax.bin"
    path.write_bytes(bytes(4096))
    manager = DevDaxL1MemoryManager(
        _config(use_hugepages, devdax_path=str(path), devdax_size_in_bytes=4096)
    )
    desc = manager.get_l1_memory_desc()
    assert desc.size == LOGICAL_SIZE
    error, objects = manager.allocate(_layout(), count=LOGICAL_SIZE // 4096 + 1)
    assert error == L1Error.SUCCESS
    assert manager.get_backend_type(objects[0]) == L1BackendType.DRAM
    assert manager.get_backend_type(objects[-1]) == L1BackendType.DEVDAX
    tensor = objects[-1].raw_tensor
    assert tensor is not None
    tensor.fill_(0x37)
    assert tensor[0].item() == 0x37
    del tensor
    assert manager.get_memory_usage() == (LOGICAL_SIZE + 4096, LOGICAL_SIZE + 4096)
    manager.free(objects)
    manager.close()
    manager.close()
    mapped_size = 2 * HUGEPAGE_SIZE if use_hugepages else LOGICAL_SIZE
    assert native_hugepages.freed == [(desc.ptr, mapped_size)]

def test_hybrid_mapping_failure_releases_dram(
    native_hugepages: SimpleNamespace, tmp_path: Path
) -> None:
    """A failed Device-DAX setup closes its already mapped DRAM pool."""
    with pytest.raises(OSError):
        DevDaxL1MemoryManager(
            _config(
                True,
                devdax_path=str(tmp_path / "missing-device"),
                devdax_size_in_bytes=4096,
            )
        )
    assert native_hugepages.freed == native_hugepages.mapped


def test_hybrid_arena_setup_failure_releases_dram(
    native_hugepages: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failure after Device-DAX mapping also releases the local DRAM pool."""
    path = tmp_path / "devdax.bin"
    path.write_bytes(bytes(4096))

    def fail_pin(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Device-DAX setup failed")

    monkeypatch.setattr(DevDaxMemoryAllocator, "_register_arena_pin", fail_pin)
    with pytest.raises(RuntimeError, match="Device-DAX setup failed"):
        DevDaxL1MemoryManager(
            _config(True, devdax_path=str(path), devdax_size_in_bytes=4096)
        )
    assert native_hugepages.freed == native_hugepages.mapped


@pytest.mark.parametrize("message", ["mmap failed", "cudaHostRegister failed"])
def test_native_allocation_failure_has_no_fallback(
    native_hugepages: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """Native mapping and registration errors abort manager construction."""

    def fail_native(size: int, flags: int) -> int:
        raise RuntimeError(message)

    monkeypatch.setattr(
        memory_management.device_ops, "alloc_hugepage_pinned_ptr", fail_native
    )
    with pytest.raises(RuntimeError, match=message):
        L1MemoryManager(_config(True))
    assert native_hugepages.mapped == []

@pytest.mark.parametrize(
    ("native_error", "action"),
    [
        ("mmap failed (errno=12): Cannot allocate memory", "Check the 2 MiB pool"),
        ("mmap failed (errno=1): Operation not permitted", "permission denied"),
        ("mbind failed (errno=1): Operation not permitted", "NUMA binding failed"),
        ("cudaHostRegister failed: invalid argument", "GPU registration failed"),
    ],
)
