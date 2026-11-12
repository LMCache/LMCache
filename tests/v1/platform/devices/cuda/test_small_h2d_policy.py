# SPDX-License-Identifier: Apache-2.0
"""CUDA copy-policy contract through the public cache-context interface."""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.multiprocess.object_group_transfer import downsample_and_stage_block_ids
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.devices.cuda.cache_context import GPUCacheContext
import lmcache.lmcache_native as native

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        torch_device_type != "cuda" or not torch_dev.is_available(),
        reason="Requires the CUDA cache context and a GPU",
    ),
]


class TensorWrapper:
    """Provide a local tensor without reopening same-process CUDA IPC."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        """Return the locally owned KV tensor."""
        return self.tensor

    def close(self) -> None:
        """Keep local tensor ownership in the fixture."""


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("head_width", [64, 256, 320, 512])
@pytest.mark.parametrize(
    "direction", [native.TransferDirection.H2D, native.TransferDirection.D2H]
)
@pytest.mark.parametrize("hip", [False, True])
def test_fused_copy_policy(
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
    head_width: int,
    direction: native.TransferDirection,
    hip: bool,
) -> None:
    """Only CUDA small fused H2D with host IDs suppresses direct dispatch."""
    shape = (8, 16, 4, head_width) if fused else (2, 8, 16, 4, head_width)
    tensors = [torch.empty(shape, dtype=torch.uint8, device="cuda") for _ in range(2)]
    context = GPUCacheContext(
        [TensorWrapper(t) for t in tensors],  # type: ignore[misc]
        lmcache_tokens_per_chunk=32,
    )
    calls: list[dict[str, object]] = []

    def record(self: BaseCacheContext, *args: object, **kwargs: object) -> None:
        calls.append(kwargs)

    try:
        expected_format = (
            native.EngineKVFormat.NL_X_NB_BS_NH_CS
            if fused
            else native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
        )
        assert context.get_engine_kv_format(0) == expected_format
        monkeypatch.setattr(BaseCacheContext, "transfer_object_group", record)
        monkeypatch.setattr(torch.version, "hip", "test" if hip else None)
        ids = [[3, 2]]
        context.transfer_object_group(
            [],
            [],
            object_group_id=0,
            batch_size=4,
            skip_first_n_tokens=16,
            direction=direction,
            transfer_key="policy-test",
            block_ids_host=ids,
        )
        expected_staging = (
            fused
            and not hip
            and direction == native.TransferDirection.H2D
            and 16 * 4 * head_width <= 16384
        )
        assert calls[0]["block_ids_host"] == (() if expected_staging else ids)
        assert calls[0]["skip_first_n_tokens"] == 16
        assert calls[0]["direction"] == direction
        assert calls[0]["transfer_key"] == "policy-test"
        context.transfer_object_group(
            [],
            [],
            object_group_id=0,
            batch_size=4,
            skip_first_n_tokens=0,
            direction=direction,
            transfer_key="no-host-ids",
        )
        assert calls[1]["block_ids_host"] == ()
    finally:
        context.close()


@pytest.mark.parametrize("skip", [0, 16])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_fused_retrieve_preserves_paged_bytes(skip: int, dtype: torch.dtype) -> None:
    """Descending page IDs and a skipped prefix retain exact bytes on both policies."""
    tensors = [
        torch.randint(0, 100, (8, 16, 4, 256), device="cuda").to(dtype)
        for _ in range(2)
    ]
    context = GPUCacheContext(
        [TensorWrapper(t) for t in tensors],  # type: ignore[misc]
        lmcache_tokens_per_chunk=32,
    )
    allocator = LazyMemoryAllocator(init_size=32 << 20, final_size=32 << 20)
    objects = []
    try:
        shape, host_dtype = context.get_kernel_group_shape_dtype(32, 0)
        ids = [[3, 2, 5, 4]]
        expected = [t.cpu().view(torch.uint8).clone() for t in tensors]
        for chunk in range(2):
            obj = allocator.allocate(shape, host_dtype)
            assert obj is not None
            raw = obj.raw_tensor
            assert raw is not None
            raw.view(torch.uint8).copy_(
                torch.randint(0, 256, raw.view(torch.uint8).shape, dtype=torch.uint8)
            )
            payload = raw.view(torch.uint8)[: obj.get_size()].view(
                2, 32, 4, 256 * dtype.itemsize
            )
            for layer in range(2):
                for block in range(2):
                    if chunk == 0 and block * 16 < skip:
                        continue
                    expected[layer][ids[0][2 * chunk + block]] = payload[
                        layer, block * 16 : (block + 1) * 16
                    ]
            objects.append(obj)
        torch_dev.synchronize(context.device)
        with torch_dev.stream(context.stream):
            gpu_ids = downsample_and_stage_block_ids(context, ids)
            context.transfer_object_group(
                gpu_ids,
                objects,
                object_group_id=0,
                batch_size=4,
                skip_first_n_tokens=skip,
                direction=native.TransferDirection.H2D,
                transfer_key="byte-policy",
                block_ids_host=ids,
            )
        context.stream.synchronize()
        for actual, reference in zip(tensors, expected, strict=True):
            assert torch.equal(actual.cpu().view(torch.uint8), reference)
    finally:
        context.stream.synchronize()
        for obj in objects:
            allocator.free(obj)
        context.close()
        allocator.close()
