# SPDX-License-Identifier: Apache-2.0
"""GPU cache context: the staging buffer as a transfer-channel device region
for GPU-direct borrowing (CUDA)."""

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
from lmcache.v1.distributed.transfer_channel import (  # noqa: E402
    DeviceRegionTarget,
    set_device_region_target,
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


def _target() -> tuple[MagicMock, DeviceRegionTarget]:
    context = MagicMock()
    context.register_device_region.return_value = 1
    return context, DeviceRegionTarget(context, _ALIGN)


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


def test_context_has_no_landing_banks_without_a_target() -> None:
    ctx = _context()
    try:
        assert ctx.landing_banks == 0
        assert ctx.max_batch_size == 4
        with pytest.raises(RuntimeError, match="device region"):
            ctx.get_landing_slot(0, 0, 0)
    finally:
        ctx.close()


def test_context_registers_its_staging_buffer_as_aligned_banks() -> None:
    context, target = _target()
    set_device_region_target(target)
    try:
        ctx = _context()
    finally:
        set_device_region_target(None)
    try:
        # The stream-ordered batch size is unchanged; a second bank is added.
        assert ctx.max_batch_size == 4
        assert ctx.landing_banks == 2
        (call_,) = context.register_device_region.call_args_list
        ptr, size, device_index = call_.args
        assert ptr % _ALIGN == 0
        assert size % _ALIGN == 0
        assert device_index == 0

        slot = ctx.get_landing_slot(1, 1, 0)
        assert slot.region_id == 1
        assert slot.offset % _ALIGN == 0
        assert slot.buffer.device == torch.device("cuda:0")
        assert slot.buffer.data_ptr() == ptr + slot.offset
        assert slot.offset + slot.buffer.nbytes <= size
        # Bank 1 is the batch slots after the stream-ordered paths' own.
        staged = ctx.get_temp_object_group_buffer(ctx.max_batch_size + 1, 0)
        assert staged.data_ptr() == slot.buffer.data_ptr()
        assert staged.nbytes == slot.buffer.nbytes
        # Bank 0 is those paths' own slots.
        first = ctx.get_landing_slot(0, 0, 0)
        assert first.offset == 0
        staged0 = ctx.get_temp_object_group_buffer(0, 0)
        assert first.buffer.data_ptr() == staged0.data_ptr()
        with pytest.raises(ValueError, match="bank"):
            ctx.get_landing_slot(2, 0, 0)
        with pytest.raises(ValueError, match="chunk_idx"):
            ctx.get_landing_slot(1, ctx.max_batch_size, 0)
    finally:
        ctx.close()
    context.unregister_device_region.assert_called_once_with(1)
