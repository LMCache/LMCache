# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LazyMemoryAllocator's deferred ("lazy") pinning contract.

Host pinning (``cudaHostRegister``) creates a CUDA context on the calling
thread's current device, and ``LazyMemoryAllocator.__init__`` runs before any
worker's device is known. The allocator therefore must not pin (or touch the
device at all) during construction; pinning is bound to a device later via
``ensure_pinning``, triggered at the latest by the first allocation.

These tests verify that contract by observing device side effects (the
pin/unpin calls and the device context they run in) rather than by inspecting
private state, so they need no GPU.
"""

# Standard
from unittest.mock import call, patch
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator

PIN_CHUNK = LazyMemoryAllocator.PIN_CHUNK_SIZE

MODULE = "lmcache.v1.memory_allocators.lazy_memory_allocator"


@pytest.fixture
def mock_device_spec():
    """Patch the pin/unpin backend (``current_device_spec``)."""
    with patch(f"{MODULE}.current_device_spec") as spec:
        spec.is_pin_supported = True
        spec.pin_memory.return_value = True
        yield spec


@pytest.fixture
def mock_torch_dev():
    """Patch the torch device module; a MagicMock auto-supports ``with``."""
    with patch(f"{MODULE}.torch_dev") as td:
        td.is_available.return_value = True
        td.current_device.return_value = 0
        yield td


@pytest.fixture
def mock_tma():
    """Patch TensorMemoryAllocator to isolate the lazy logic from it."""
    with patch(f"{MODULE}.TensorMemoryAllocator") as tma:
        yield tma


def _make_allocator(
    init: int = PIN_CHUNK, final: int = PIN_CHUNK
) -> LazyMemoryAllocator:
    """Build an allocator. init == final keeps the expand thread a no-op."""
    return LazyMemoryAllocator(init, final)


def test_construction_does_not_touch_device(mock_device_spec, mock_torch_dev, mock_tma):
    """The core contract: __init__ neither pins nor queries the device."""
    _make_allocator()

    mock_device_spec.pin_memory.assert_not_called()
    mock_torch_dev.current_device.assert_not_called()
    mock_torch_dev.device.assert_not_called()


def test_ensure_pinning_pins_initial_chunk_on_device(
    mock_device_spec, mock_torch_dev, mock_tma
):
    """ensure_pinning(d) pins the initial chunk inside device d's context."""
    allocator = _make_allocator()

    allocator.ensure_pinning(3)

    mock_device_spec.pin_memory.assert_called_once()
    mock_torch_dev.device.assert_called_with(3)

    allocator.close()


def test_ensure_pinning_is_idempotent(mock_device_spec, mock_torch_dev, mock_tma):
    """A second ensure_pinning does not re-pin or rebind the device."""
    allocator = _make_allocator()

    allocator.ensure_pinning(3)
    pins_after_first = mock_device_spec.pin_memory.call_count

    allocator.ensure_pinning(5)

    assert mock_device_spec.pin_memory.call_count == pins_after_first
    assert call(5) not in mock_torch_dev.device.call_args_list

    allocator.close()


def test_concurrent_ensure_pinning_pins_once(
    mock_device_spec, mock_torch_dev, mock_tma
):
    """Under concurrent ensure_pinning calls, exactly one thread pins."""
    allocator = _make_allocator()
    start = threading.Barrier(8)

    def worker(device: int) -> None:
        start.wait()  # maximize contention on the init lock
        allocator.ensure_pinning(device)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    mock_device_spec.pin_memory.assert_called_once()

    allocator.close()


def test_allocate_triggers_pinning_on_current_device(
    mock_device_spec, mock_torch_dev, mock_tma
):
    """The first allocate lazily pins on the current (worker) device."""
    mock_torch_dev.current_device.return_value = 2
    allocator = _make_allocator()

    mock_device_spec.pin_memory.assert_not_called()

    allocator.allocate(torch.Size([16]), torch.uint8)

    mock_device_spec.pin_memory.assert_called_once()
    mock_torch_dev.device.assert_called_with(2)
    mock_tma.return_value.allocate.assert_called_once()

    allocator.close()


def test_batched_allocate_triggers_pinning(mock_device_spec, mock_torch_dev, mock_tma):
    """The first batched_allocate also triggers the lazy pinning."""
    allocator = _make_allocator()

    allocator.batched_allocate(torch.Size([16]), torch.uint8, batch_size=2)

    mock_device_spec.pin_memory.assert_called_once()
    mock_tma.return_value.batched_allocate.assert_called_once()

    allocator.close()


def test_second_allocate_does_not_repin(mock_device_spec, mock_torch_dev, mock_tma):
    """Once pinned, later allocations do not pin again."""
    allocator = _make_allocator()

    allocator.allocate(torch.Size([16]), torch.uint8)
    pins_after_first = mock_device_spec.pin_memory.call_count

    allocator.allocate(torch.Size([16]), torch.uint8)

    assert mock_device_spec.pin_memory.call_count == pins_after_first

    allocator.close()


def test_close_before_pinning_is_safe(mock_device_spec, mock_torch_dev, mock_tma):
    """close() before any pinning neither raises nor unpins."""
    allocator = _make_allocator()

    allocator.close()

    mock_device_spec.unpin_memory.assert_not_called()


def test_ensure_pinning_after_close_is_noop(mock_device_spec, mock_torch_dev, mock_tma):
    """ensure_pinning after close() must not pin (close/first-use race guard)."""
    allocator = _make_allocator()
    allocator.close()

    allocator.ensure_pinning(3)

    mock_device_spec.pin_memory.assert_not_called()


def test_close_after_pinning_unpins(mock_device_spec, mock_torch_dev, mock_tma):
    """close() after pinning unpins the chunk it recorded."""
    allocator = _make_allocator()
    allocator.ensure_pinning(3)

    allocator.close()

    mock_device_spec.unpin_memory.assert_called_once()
