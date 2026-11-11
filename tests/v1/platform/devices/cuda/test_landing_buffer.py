# SPDX-License-Identifier: Apache-2.0
"""GPU cache context: the landing buffer a transfer channel reads into (CUDA)."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")

# First Party
from lmcache.v1.platform.devices.cuda.cache_context import (  # noqa: E402
    GPUCacheContext,
)
from lmcache.v1.platform.landing_region import (  # noqa: E402
    LandingRegionTarget,
    get_landing_region_target,
    set_landing_region_target,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires the CUDA backend"),
    pytest.mark.skipif(not torch_dev.is_available(), reason="Requires a GPU"),
]

_ALIGN = 65536
_REGION = 7


class _FakeIPCWrapper:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return None


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


@pytest.fixture
def registrar():
    """Install a landing-region target for the test; yield its registrar."""
    registrar = MagicMock()
    registrar.register_memory_region.return_value = _REGION
    set_landing_region_target(LandingRegionTarget(registrar, _ALIGN))
    yield registrar
    set_landing_region_target(None)


def test_no_target_means_no_landing_buffer() -> None:
    assert get_landing_region_target() is None
    ctx = _context()
    try:
        assert ctx.landing_banks == 0
        with pytest.raises(RuntimeError, match="landing buffer"):
            ctx.get_landing_slot(0, 0, 0)
    finally:
        ctx.close()


def test_landing_buffer_is_registered_as_one_aligned_gpu_region(registrar) -> None:
    ctx = _context()
    try:
        (registration,) = registrar.register_memory_region.call_args_list
        ptr, size, device_index = registration.args
        assert ptr % _ALIGN == 0
        assert size > 0 and size % _ALIGN == 0
        assert device_index == 0
        assert ctx.landing_banks == 2
        # The stream-ordered paths keep their own staging slots.
        assert ctx.max_batch_size == 4
        staged = ctx.get_temp_object_group_buffer(0, 0)
        assert not ptr <= staged.data_ptr() < ptr + size

        object_bytes = staged.nbytes
        spans = []
        for bank in range(ctx.landing_banks):
            for chunk_idx in range(ctx.max_batch_size):
                slot = ctx.get_landing_slot(bank, chunk_idx, 0)
                assert slot.region_id == _REGION
                assert slot.offset % _ALIGN == 0
                assert slot.buffer.device == torch.device("cuda:0")
                assert slot.buffer.data_ptr() == ptr + slot.offset
                assert slot.buffer.nbytes == object_bytes
                # A read covers whole pages, so the slot owns them all.
                spans.append((slot.offset, -(-object_bytes // _ALIGN) * _ALIGN))
        spans.sort()
        for (start, length), (next_start, _) in zip(spans, spans[1:], strict=False):
            assert start + length <= next_start
        assert spans[-1][0] + spans[-1][1] <= size
    finally:
        ctx.close()


@pytest.mark.parametrize("index", [(2, 0, 0), (0, 4, 0), (0, 0, 1), (-1, 0, 0)])
def test_out_of_range_slot_is_rejected(registrar, index) -> None:
    ctx = _context()
    try:
        with pytest.raises(ValueError, match="Invalid"):
            ctx.get_landing_slot(*index)
    finally:
        ctx.close()


def test_close_deregisters_the_region(registrar) -> None:
    ctx = _context()
    registrar.deregister_memory_region.assert_not_called()
    ctx.close()
    registrar.deregister_memory_region.assert_called_once_with(_REGION)
