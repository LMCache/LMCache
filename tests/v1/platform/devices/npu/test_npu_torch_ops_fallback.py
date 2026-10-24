# SPDX-License-Identifier: Apache-2.0
"""NPU-gated tests for torch-fallback pointer views and memcpy.

``_tensor_from_ptr`` must reconstruct a non-owning view over an Ascend NPU
device pointer so the fallback ``multi_layer_block_kv_transfer`` path can
serve NPU paged buffers. Pointer-mode ``lmcache_memcpy_async`` copies via
those views, not libcudart. These tests need real Ascend hardware and skip
cleanly everywhere else.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.devices.npu.device_ops import NpuDeviceOps
import lmcache.lmcache_native as lmcache_native

pytestmark = [
    pytest.mark.npu,
    pytest.mark.no_shared_allocator,
]

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU hardware is required",
)


@requires_npu
def test_tensor_from_ptr_npu_pointer_view() -> None:
    """A raw NPU pointer round-trips into an aliasing tensor view."""
    # Third Party
    import torch_npu  # noqa: F401

    original = torch.arange(24, dtype=torch.float32, device="npu").reshape(4, 6)
    view = torch_ops._tensor_from_ptr(
        original.data_ptr(), original.shape, original.dtype, original.device
    )

    assert view.data_ptr() == original.data_ptr()
    assert view.shape == original.shape
    assert view.dtype == original.dtype
    assert view.device.type == "npu"

    # The view must alias the original buffer, not copy it: a mutation
    # through the view has to be observed by the caller's tensor.
    view.fill_(7.0)
    assert torch.equal(original, torch.full_like(original, 7.0))


@requires_npu
def test_pointer_mode_memcpy_roundtrip_on_device() -> None:
    """lmcache_memcpy_async pointer mode copies via tensor views, not CUDA."""
    ops = NpuDeviceOps()
    device = torch.device("npu:0")
    # int32: aclnnArange does not implement uint8 on this CANN build.
    src = torch.arange(64, dtype=torch.int32, device=device)
    host = torch.zeros(64, dtype=torch.int32, pin_memory=True)
    ops.lmcache_memcpy_async(
        host.data_ptr(),
        src.data_ptr(),
        256,
        lmcache_native.TransferDirection.D2H,
        0,
        1,
    )
    torch.npu.current_stream().synchronize()
    assert torch.equal(host, src.cpu())

    mutated = torch.full((64,), 7, dtype=torch.int32)
    ops.lmcache_memcpy_async(
        src.data_ptr(),
        mutated.data_ptr(),
        256,
        lmcache_native.TransferDirection.H2D,
        0,
        1,
    )
    torch.npu.current_stream().synchronize()
    assert torch.equal(src.cpu(), mutated)
