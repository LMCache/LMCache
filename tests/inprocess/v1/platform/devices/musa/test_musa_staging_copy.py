# SPDX-License-Identifier: Apache-2.0
"""In-process MUSA connector copies across pin-registration boundaries."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.gpu_connector import musa_connectors
from lmcache.inprocess.v1.gpu_connector.musa_connectors import (
    VLLMPagedMemLayerwiseMUSAConnector,
    VLLMPagedMemMUSAConnectorV2,
)
from lmcache.v1.gpu_connector.kv_format.detectors import vllm as vllm_detector
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from tests.v1.platform.devices.musa.staging_copy_utils import FakeMemoryObj


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
    memory_obj = FakeMemoryObj(
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
    memory_obj = FakeMemoryObj(
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
