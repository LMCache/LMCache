# SPDX-License-Identifier: Apache-2.0
"""GPU cache context: landing ring for GPU-direct borrowing (CUDA)."""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")

# First Party
from lmcache.v1.gpu_connector.gdr_context import (  # noqa: E402
    DeviceRegionRegistrar,
    set_device_region_registrar,
)
from lmcache.v1.platform.devices.cuda.cache_context import (  # noqa: E402
    GPUCacheContext,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires the CUDA backend"),
    pytest.mark.skipif(not torch_dev.is_available(), reason="Requires a GPU"),
]

_ALIGN = 65536


class _FakeIPCWrapper:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return None


class _Registrar(DeviceRegionRegistrar):
    def __init__(self) -> None:
        self.regions: list[tuple[int, int, int]] = []
        self.released: list[int] = []

    @property
    def align_bytes(self) -> int:
        return _ALIGN

    def register(self, ptr: int, size: int, device_index: int) -> int:
        self.regions.append((ptr, size, device_index))
        return 1

    def unregister(self, region_id: int) -> None:
        self.released.append(region_id)


def _context() -> GPUCacheContext:
    tensors = [
        torch.zeros([2, 8, 16, 4, 32], dtype=torch.bfloat16, device="cuda:0")
        for _ in range(2)
    ]
    return GPUCacheContext(
        [_FakeIPCWrapper(t) for t in tensors],  # type: ignore[misc]
        lmcache_tokens_per_chunk=32,
        engine_type=EngineType.VLLM,
    )


def test_context_has_no_ring_without_a_registrar() -> None:
    ctx = _context()
    try:
        assert ctx.gdr_landing is None
    finally:
        ctx.close()


def test_context_registers_an_aligned_ring_sized_for_its_objects() -> None:
    registrar = _Registrar()
    set_device_region_registrar(registrar)
    try:
        ctx = _context()
    finally:
        set_device_region_registrar(None)
    try:
        ring = ctx.gdr_landing
        assert ring is not None
        object_bytes = ctx.get_temp_object_group_buffer(0, 0).nbytes
        assert ring.slot_bytes >= object_bytes
        assert ring.slot_bytes % _ALIGN == 0
        assert ring.slots_per_bank == ctx.max_batch_size
        (ptr, size, device_index), *rest = registrar.regions
        assert not rest
        assert ptr % _ALIGN == 0
        assert size == 2 * ring.slots_per_bank * ring.slot_bytes
        assert device_index == 0
        slot = ring.slot(1, 1)
        assert slot.buffer.device == torch.device("cuda:0")
        assert slot.buffer.data_ptr() == ptr + slot.offset
    finally:
        ctx.close()
    assert registrar.released == [1]
