# SPDX-License-Identifier: Apache-2.0
"""Regression tests for lazy MUSA copies at pin-registration boundaries."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.gpu_connector import gpu_ops, musa_connectors
from lmcache.v1.gpu_connector.kv_format.detectors import vllm as vllm_detector
from lmcache.v1.gpu_connector.musa_connectors import (
    VLLMPagedMemLayerwiseMUSAConnector,
    VLLMPagedMemMUSAConnectorV2,
)
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.platform.musa import device_ops as musa_device_ops
from lmcache.v1.platform.musa.device_ops import MusaDeviceOps
from lmcache.v1.storage_backend.naive_serde import cachegen_encoder


class _FakeMemoryObj:
    """Minimal lazy-memory object surface consumed by ``gpu_ops``."""

    def __init__(
        self,
        ptr: int,
        size: int,
        host_offset: int,
        tensor: torch.Tensor | None = None,
        fmt: MemoryFormat | None = None,
    ) -> None:
        self.data_ptr = ptr
        self.raw_tensor = tensor if tensor is not None else object()
        self.tensor = tensor
        self.meta = SimpleNamespace(address=host_offset, fmt=fmt)
        self.metadata = self.meta
        self._size = size
        self._parent = LazyMemoryAllocator.__new__(LazyMemoryAllocator)

    def get_size(self) -> int:
        """Return the represented host span in bytes."""
        return self._size

    def parent(self) -> LazyMemoryAllocator:
        """Return a lazy allocator marker for the production branch."""
        return self._parent


class _FakeMusaStream:
    """CPU-only stream double for public MUSA connector tests."""

    def wait_stream(self, stream: object) -> None:
        """Accept a stream dependency without device work."""

    def synchronize(self) -> None:
        """Accept a host synchronization without device work."""


class _FakeMusaModule:
    """CPU-only ``torch.musa`` surface used by connector generators."""

    def Stream(self) -> _FakeMusaStream:
        """Return a fake transfer stream."""
        return _FakeMusaStream()

    def current_stream(self) -> _FakeMusaStream:
        """Return a fake current stream."""
        return _FakeMusaStream()

    def stream(self, stream: object) -> nullcontext[None]:
        """Return a no-op stream context."""
        return nullcontext()

    def synchronize(self) -> None:
        """Accept a device-wide synchronization without device work."""


_FAKE_MUSA_DEVICE = SimpleNamespace(type="musa")


class _FakeMusaTensor(torch.Tensor):
    """CPU tensor subclass that reports a MUSA device for connector setup."""

    @property
    def device(self) -> SimpleNamespace:  # type: ignore[override]
        """Report the fake MUSA device without requiring MUSA hardware."""
        return _FAKE_MUSA_DEVICE


def _install_fake_musa_pointer_views(
    monkeypatch: pytest.MonkeyPatch,
    device_buffer: torch.Tensor,
) -> list[tuple[int, int]]:
    """Map fake MUSA pointers to slices of one CPU tensor for unit tests."""
    calls: list[tuple[int, int]] = []
    device_base = device_buffer.data_ptr()
    monkeypatch.setattr(
        musa_device_ops,
        "_current_musa_device",
        lambda: torch.device("cpu"),
    )

    def _construct(
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        assert dtype is torch.uint8
        size = shape[0]
        calls.append((ptr, size))
        start = ptr - device_base
        return device_buffer[start : start + size]

    monkeypatch.setattr(
        musa_device_ops,
        "construct_musa_tensor_from_data_pointer",
        _construct,
    )
    return calls


def test_lazy_gpu_ops_h2d_uses_musa_boundary_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public lazy H2D path delegates boundary splitting to MUSA ops."""
    chunk_size = 16
    start = chunk_size - 2
    size = 2 * chunk_size + 5
    host = torch.arange(4 * chunk_size, dtype=torch.uint8)
    device = torch.zeros(4 * chunk_size, dtype=torch.uint8)
    calls = _install_fake_musa_pointer_views(monkeypatch, device)
    memory_tensor = host[start : start + size]
    memory_obj = _FakeMemoryObj(
        memory_tensor.data_ptr(),
        memory_tensor.nbytes,
        start,
        tensor=memory_tensor,
    )
    monkeypatch.setattr(gpu_ops, "device_ops", MusaDeviceOps())
    monkeypatch.setattr(LazyMemoryAllocator, "PIN_CHUNK_SIZE", chunk_size)

    gpu_ops.lmcache_memcpy_async_h2d(
        cast(MemoryObj, memory_obj),
        device[:size],
    )

    assert [segment_size for _, segment_size in calls] == [2, 16, 16, 3]
    assert torch.equal(device[:size], host[start : start + size])


def test_lazy_gpu_ops_d2h_uses_musa_boundary_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public lazy D2H path delegates boundary splitting to MUSA ops."""
    chunk_size = 16
    start = chunk_size - 3
    size = chunk_size + 8
    host = torch.zeros(4 * chunk_size, dtype=torch.uint8)
    device = torch.arange(4 * chunk_size, dtype=torch.uint8)
    calls = _install_fake_musa_pointer_views(monkeypatch, device)
    memory_tensor = host[start : start + size]
    memory_obj = _FakeMemoryObj(
        memory_tensor.data_ptr(),
        memory_tensor.nbytes,
        start,
        tensor=memory_tensor,
    )
    monkeypatch.setattr(gpu_ops, "device_ops", MusaDeviceOps())
    monkeypatch.setattr(LazyMemoryAllocator, "PIN_CHUNK_SIZE", chunk_size)

    gpu_ops.lmcache_memcpy_async_d2h(
        device[:size],
        cast(MemoryObj, memory_obj),
    )

    assert [segment_size for _, segment_size in calls] == [3, 16, 5]
    assert torch.equal(host[start : start + size], device[:size])


def test_musa_device_ops_preserves_tensor_copy_mode() -> None:
    """MUSA ops retain the base operation's tensor-mode contract."""
    source = torch.arange(16, dtype=torch.uint8)
    destination = torch.zeros_like(source)

    MusaDeviceOps().lmcache_memcpy_async(
        destination,
        source,
        source.nbytes,
        TransferDirection.H2D,
        0,
        16,
    )

    assert torch.equal(destination, source)


def test_cachegen_uses_boundary_safe_copy_for_lazy_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CacheGen stages lazy inputs independently of the accelerator name."""
    source = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    memory_obj = _FakeMemoryObj(
        source.data_ptr(),
        source.nbytes,
        host_offset=14,
        tensor=source,
    )
    copy_calls: list[tuple[object, torch.Tensor]] = []
    encoded: dict[str, torch.Tensor] = {}
    original_empty_like = torch.empty_like

    def _cpu_empty_like(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        kwargs.pop("device", None)
        return original_empty_like(tensor, *args, **kwargs)

    def _copy_to_device(obj: object, destination: torch.Tensor) -> None:
        copy_calls.append((obj, destination))
        destination.copy_(source)

    def _encode(tensor: torch.Tensor, *_args: object) -> SimpleNamespace:
        encoded["tensor"] = tensor.clone()
        return SimpleNamespace(to_bytes=lambda: b"encoded")

    monkeypatch.setattr(cachegen_encoder, "torch_device_type", "cuda")
    monkeypatch.setattr(cachegen_encoder.torch, "empty_like", _cpu_empty_like)
    monkeypatch.setattr(
        cachegen_encoder,
        "lmcache_memcpy_async_h2d",
        _copy_to_device,
    )
    monkeypatch.setattr(cachegen_encoder, "encode_function", _encode)
    monkeypatch.setattr(cachegen_encoder.torch_dev, "current_device", lambda: None)

    serializer = cachegen_encoder.CacheGenSerializer.__new__(
        cachegen_encoder.CacheGenSerializer
    )
    serializer.cachegen_config = cast(
        cachegen_encoder.CacheGenConfig,
        SimpleNamespace(),
    )
    serializer.key_bins = torch.zeros(1)
    serializer.value_bins = torch.zeros(1)
    serializer.kv_shape = torch.Size([2, 1, 3, 2, 2])

    result = serializer.serialize(cast(MemoryObj, memory_obj))

    assert len(copy_calls) == 1
    assert copy_calls[0][0] is memory_obj
    assert torch.equal(
        encoded["tensor"],
        source.view(2, 1, 3, 2, 2).permute(1, 0, 2, 3, 4),
    )
    assert result.byte_array == b"encoded"


def _install_cpu_musa_transfer_shims(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, list[int]]:
    """Run public connector paths on CPU while recording each tensor copy."""
    segment_sizes: list[int] = []
    original_copy = torch.Tensor.copy_
    original_empty_like = torch.empty_like

    def _recording_copy(
        dest: torch.Tensor,
        src: torch.Tensor,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        segment_sizes.append(dest.nbytes)
        return original_copy(dest, src, non_blocking=non_blocking)

    def _stay_on_cpu(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        return tensor

    def _cpu_empty_like(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        kwargs.pop("device", None)
        return original_empty_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "musa", _FakeMusaModule(), raising=False)
    monkeypatch.setattr(torch.Tensor, "copy_", _recording_copy)
    monkeypatch.setattr(torch.Tensor, "to", _stay_on_cpu)
    monkeypatch.setattr(musa_connectors.torch, "empty_like", _cpu_empty_like)
    return _FAKE_MUSA_DEVICE, segment_sizes


def test_non_layerwise_musa_connector_splits_lazy_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public non-layerwise connector splits lazy H2D and D2H copies."""
    chunk_size = 16
    host_offset = chunk_size - 2
    num_tokens = 80
    _, segment_sizes = _install_cpu_musa_transfer_shims(monkeypatch)
    monkeypatch.setattr(LazyMemoryAllocator, "PIN_CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(vllm_detector, "torch_device_type", "musa")

    connector = VLLMPagedMemMUSAConnectorV2()

    host = torch.arange(2 * num_tokens, dtype=torch.uint8).reshape(
        2,
        1,
        num_tokens,
        1,
    )
    memory_obj = _FakeMemoryObj(
        host.data_ptr(),
        host.nbytes,
        host_offset,
        tensor=host,
        fmt=MemoryFormat.KV_2LTD,
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)
    kvcaches = [
        torch.zeros(2, 5, 16, 1, 1, dtype=torch.uint8).as_subclass(_FakeMusaTensor),
    ]

    connector.to_gpu(
        cast(MemoryObj, memory_obj),
        start=0,
        end=num_tokens,
        slot_mapping=slot_mapping,
        kvcaches=kvcaches,
    )

    assert segment_sizes == 2 * [2, 16, 16, 16, 16, 14]
    assert torch.equal(kvcaches[0][0].flatten(), host[0].flatten())
    assert torch.equal(kvcaches[0][1].flatten(), host[1].flatten())

    expected = host.clone()
    host.zero_()
    segment_sizes.clear()
    connector.from_gpu(
        cast(MemoryObj, memory_obj),
        start=0,
        end=num_tokens,
        slot_mapping=slot_mapping,
        kvcaches=kvcaches,
    )

    assert segment_sizes == 2 * [2] + 18 * [16] + 2 * [14]
    assert torch.equal(host, expected)


def test_layerwise_musa_connector_splits_lazy_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public layerwise connector splits lazy H2D and D2H copies."""
    chunk_size = 16
    host_offset = chunk_size - 2
    num_tokens = 20
    fake_device, segment_sizes = _install_cpu_musa_transfer_shims(monkeypatch)
    monkeypatch.setattr(LazyMemoryAllocator, "PIN_CHUNK_SIZE", chunk_size)

    connector = VLLMPagedMemLayerwiseMUSAConnector(
        hidden_dim_size=1,
        num_layers=1,
        use_musa=False,
        chunk_size=num_tokens,
        dtype=torch.uint8,
        device=fake_device,
    )
    host = torch.arange(2 * num_tokens, dtype=torch.uint8).reshape(
        num_tokens,
        2,
        1,
    )
    memory_obj = _FakeMemoryObj(
        host.data_ptr(),
        host.nbytes,
        host_offset,
        tensor=host,
        fmt=MemoryFormat.KV_T2D,
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)
    kvcaches = [
        torch.zeros(2, 5, 4, 1, 1, dtype=torch.uint8),
    ]

    consumer = connector.batched_to_gpu(
        starts=[0],
        ends=[num_tokens],
        slot_mapping=slot_mapping,
        sync=True,
        kvcaches=kvcaches,
    )
    next(consumer)
    consumer.send([cast(MemoryObj, memory_obj)])
    consumer.close()

    assert segment_sizes == [2, 16, 16, 6]
    assert torch.equal(kvcaches[0][0].flatten(), host[:, 0].flatten())
    assert torch.equal(kvcaches[0][1].flatten(), host[:, 1].flatten())

    expected = host.clone()
    segment_sizes.clear()
    host.zero_()
    producer = connector.batched_from_gpu(
        [[cast(MemoryObj, memory_obj)]],
        starts=[0],
        ends=[num_tokens],
        slot_mapping=slot_mapping,
        sync=True,
        kvcaches=kvcaches,
    )
    next(producer)
    producer.close()

    assert segment_sizes == [2, 16, 16, 6]
    assert torch.equal(host, expected)
