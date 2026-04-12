# SPDX-License-Identifier: Apache-2.0
"""Cross-platform GPU connector abstraction.

Provides mock replacements for CUDA primitives (events,
streams, device/stream guards) and CPU-side memcpy helpers
so that the MP server can run on CPU-only platforms.

Public API::

    from lmcache.v1.platform.gpu_connector import (
        MockCudaEvent,
        MockCudaStream,
        noop_device_guard,
    )
"""

# Future
from __future__ import annotations

# Standard
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.capabilities import HAS_CUDA

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


# ------------------------------------------------------------------
# Factory helpers — callers use these instead of torch.cuda.* directly
# ------------------------------------------------------------------


def create_ipc_event() -> Any:
    """Create an interprocess CUDA event, or a mock."""
    if HAS_CUDA:
        return torch.cuda.Event(interprocess=True)
    return MockCudaEvent(interprocess=True)


def event_from_ipc_handle(device: Any, handle: bytes) -> Any:
    """Reconstruct a CUDA event from an IPC handle."""
    if HAS_CUDA:
        return torch.cuda.Event.from_ipc_handle(device, handle)
    return MockCudaEvent.from_ipc_handle(device, handle)


def device_guard(device: Any) -> Any:
    """Context manager: ``torch.cuda.device()`` or no-op."""
    if HAS_CUDA:
        return torch.cuda.device(device)
    return noop_device_guard(device)


def stream_guard(stream: Any) -> Any:
    """Context manager: ``torch.cuda.stream()`` or no-op."""
    if HAS_CUDA:
        return torch.cuda.stream(stream)
    return noop_stream_guard(stream)


# ------------------------------------------------------------------
# Mock CUDA Event
# ------------------------------------------------------------------


class MockCudaEvent:
    """Drop-in replacement for ``torch.cuda.Event``.

    ``record()`` and ``wait()`` are no-ops.
    ``ipc_handle()`` returns an empty bytes object.
    ``from_ipc_handle()`` returns a new ``MockCudaEvent``.
    """

    def __init__(
        self,
        *,
        interprocess: bool = False,
        enable_timing: bool = False,
    ) -> None:
        pass

    def record(self, stream: Any = None) -> None:  # noqa: ARG002
        pass

    def wait(self, stream: Any = None) -> None:  # noqa: ARG002
        pass

    def ipc_handle(self) -> bytes:
        return b""

    @classmethod
    def from_ipc_handle(
        cls,
        device: Any,  # noqa: ARG003
        handle: bytes,  # noqa: ARG003
    ) -> "MockCudaEvent":
        return cls()


# ------------------------------------------------------------------
# Mock cupy-compatible stream
# ------------------------------------------------------------------


class MockCudaStream:
    """Drop-in for ``cupy.cuda.ExternalStream``.

    ``launch_host_func(fn, *args)`` calls *fn* synchronously.
    """

    @property
    def ptr(self) -> int:
        return 0

    def launch_host_func(
        self,
        fn: Callable[..., Any],
        *args: Any,
    ) -> None:
        fn(*args)


# ------------------------------------------------------------------
# No-op device / stream context managers
# ------------------------------------------------------------------


@contextmanager
def noop_device_guard(
    device: Any,  # noqa: ARG001
) -> Generator[None, None, None]:
    """No-op replacement for ``torch.cuda.device(...)``."""
    yield


@contextmanager
def noop_stream_guard(
    stream: Any,  # noqa: ARG001
) -> Generator[None, None, None]:
    """No-op replacement for ``torch.cuda.stream(...)``."""
    yield


# ------------------------------------------------------------------
# CPU memcpy helpers (real data movement on CPU)
# ------------------------------------------------------------------


def mock_memcpy_async_d2h(
    gpu_buffer: torch.Tensor,
    memory_obj: "MemoryObj",
) -> None:
    """CPU replacement for ``lmcache_memcpy_async_d2h``.

    Copies data from the (CPU) transfer buffer into the
    memory object's raw tensor.
    """
    raw = memory_obj.raw_tensor
    if raw is not None:
        mem_size = memory_obj.get_size()
        raw.view(torch.uint8)[:mem_size].copy_(gpu_buffer.view(torch.uint8)[:mem_size])


def mock_memcpy_async_h2d(
    memory_obj: "MemoryObj",
    gpu_buffer: torch.Tensor,
) -> None:
    """CPU replacement for ``lmcache_memcpy_async_h2d``.

    Copies data from the memory object into the (CPU)
    transfer buffer.
    """
    raw = memory_obj.raw_tensor
    if raw is not None:
        mem_size = memory_obj.get_size()
        gpu_buffer.view(torch.uint8)[:mem_size].copy_(raw.view(torch.uint8)[:mem_size])


def mock_multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: torch.Tensor,
    lmcache_objects_ptrs: list[int],
    block_ids: torch.Tensor,
    device: Any,
    direction: Any,
    shape_desc: Any,
    lmcache_chunk_size: int,
    gpu_kv_format: Any,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """CPU replacement for ``lmc_ops.multi_layer_block_kv_transfer``.

    Delegates to the real CPU implementation in
    ``non_cuda_equivalents``.
    """
    # First Party
    from lmcache.non_cuda_equivalents import (
        multi_layer_block_kv_transfer,
    )

    multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor,
        lmcache_objects_ptrs,
        block_ids,
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        gpu_kv_format,
        skip_prefix_n_blocks,
    )
