# SPDX-License-Identifier: Apache-2.0
"""XPU hardware tests for the SYCL ``execute_object_group_transfer``.

The planner in ``transfer_kv_per_object_group`` hands a whole object group to
the native executor when ``device_ops`` exports it, and otherwise walks the
same batches in Python, issuing ``lmcache_memcpy_async`` and
``multi_layer_block_kv_transfer`` one by one. The fallback loop is the
reference: the native plan must produce bit-identical object bytes (D2H) and
paged KV (H2D) through a real ``XPUCacheContext`` and real
``LazyMemoryAllocator`` objects. The cases cover batching (batch size 4),
``skip_first_n_tokens``, MLA and NHD layouts, and objects that straddle the
allocator's 64 MB pin chunk.
"""

# Standard
from typing import Any
import random

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops, torch_dev, torch_device_type

if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)

# First Party
from lmcache.v1.gpu_connector.gpu_ops import build_staging_copies  # noqa: E402
from lmcache.v1.memory_allocators.lazy_memory_allocator import (  # noqa: E402
    LazyMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryObj  # noqa: E402
from lmcache.v1.multiprocess.object_group_transfer import (  # noqa: E402
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.platform.devices.xpu.cache_context import (  # noqa: E402
    XPUCacheContext,
)
import lmcache.lmcache_native as lmcache_native  # noqa: E402
import lmcache.v1.multiprocess.object_group_transfer as ogt  # noqa: E402

pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not hasattr(device_ops, "execute_object_group_transfer"),
        reason="lmcache.xpu_ops was built without execute_object_group_transfer",
    ),
]

H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H
_DEVICE = torch.device("xpu:0")
_DTYPE = torch.bfloat16
_NUM_LAYERS = 4


class _FakeIPCWrapper:
    """Hands ``XPUCacheContext`` a local XPU tensor instead of an IPC import."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return None


# name -> (per-layer paged shape, expected format, chunk tokens, num chunks)
_CASES = {
    # Kimi-Linear-like MLA: one 1024-token block per chunk; 16 chunks of
    # 4.7 MB straddle the allocator's 64 MB pin chunk.
    "mla_1024": (
        [20, 1024, 576],
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        1024,
        16,
    ),
    # vLLM NHD flash-attention layout, 16 blocks of 16 tokens per chunk.
    "nhd_16": (
        [2, 96, 16, 8, 128],
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        256,
        6,
    ),
}


def _make_context(name: str) -> tuple[XPUCacheContext, list[torch.Tensor]]:
    shape, expected_fmt, chunk_tokens, _ = _CASES[name]
    tensors = [
        torch.rand(shape, dtype=_DTYPE, device=_DEVICE) for _ in range(_NUM_LAYERS)
    ]
    ctx = XPUCacheContext(
        [_FakeIPCWrapper(t) for t in tensors],  # type: ignore[misc]
        lmcache_tokens_per_chunk=chunk_tokens,
    )
    assert ctx.get_engine_kv_format(0) == expected_fmt
    return ctx, tensors


def _allocate_objects(
    allocator: LazyMemoryAllocator, ctx: XPUCacheContext, num_chunks: int, fill: bool
) -> list[MemoryObj]:
    shape, dtype = ctx.get_kernel_group_shape_dtype(ctx.lmcache_tokens_per_chunk, 0)
    objs: list[MemoryObj] = []
    for _ in range(num_chunks):
        obj = allocator.allocate(shape, dtype)
        if obj is None:
            raise RuntimeError("allocator out of space")
        view = _object_view(obj)
        if fill:
            view.copy_(torch.randint(0, 256, (obj.get_size(),), dtype=torch.uint8))
        else:
            view.zero_()
        objs.append(obj)
    return objs


def _object_view(obj: MemoryObj) -> torch.Tensor:
    raw = obj.raw_tensor
    assert raw is not None
    return raw.view(torch.uint8)[: obj.get_size()]


def _object_bytes(obj: MemoryObj) -> torch.Tensor:
    return _object_view(obj).clone()


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _run(
    monkeypatch: pytest.MonkeyPatch,
    ctx: XPUCacheContext,
    block_ids: list[list[int]],
    objs: list[MemoryObj],
    direction: "lmcache_native.TransferDirection",
    mode: str,
    batch_size: int,
    skip_first_n_tokens: int,
) -> int:
    """Run one transfer through the planner; return native executor calls."""
    calls = 0
    native = device_ops.execute_object_group_transfer

    def _counting_executor(*args, **kwargs):
        nonlocal calls
        calls += 1
        return native(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(ogt, "_HAS_NATIVE_OBJECT_GROUP_TRANSFER", mode == "native")
        mp.setattr(device_ops, "execute_object_group_transfer", _counting_executor)
        with torch_dev.device(ctx.device), torch_dev.stream(ctx.stream):
            block_ids_gpu = downsample_and_stage_block_ids(ctx, block_ids)
            transfer_kv_per_object_group(
                ctx,
                block_ids_gpu,
                objs,
                object_group_id=0,
                batch_size=batch_size,
                skip_first_n_tokens=skip_first_n_tokens,
                direction=direction,
                transfer_key=f"test-{mode}",
            )
        ctx.stream.synchronize()
    return calls


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(7)
    torch.manual_seed(7)


@pytest.fixture
def allocator():
    alloc = LazyMemoryAllocator(init_size=256 << 20, final_size=256 << 20)
    yield alloc
    alloc.close()


def test_xpu_takes_native_plan_not_direct_copy():
    """XPU has no batch memcpy: the planner must use the native plan and
    never the direct-copy path."""
    assert ogt._HAS_NATIVE_OBJECT_GROUP_TRANSFER is True
    assert ogt._HAS_BATCH_MEMCPY_ASYNC is False
    assert not hasattr(device_ops, "execute_direct_copy_transfer")


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("name", sorted(_CASES))
def test_store_native_matches_fallback(
    monkeypatch: pytest.MonkeyPatch,
    allocator: LazyMemoryAllocator,
    name: str,
    batch_size: int,
):
    """D2H: identical object bytes from the native plan and the Python loop."""
    ctx, _tensors = _make_context(name)
    _, _, chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(chunk, 0)
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    try:
        fallback_objs = _allocate_objects(allocator, ctx, num_chunks, fill=False)
        native_objs = _allocate_objects(allocator, ctx, num_chunks, fill=False)
        assert (
            _run(
                monkeypatch,
                ctx,
                block_ids,
                fallback_objs,
                D2H,
                "fallback",
                batch_size,
                0,
            )
            == 0
        )
        assert (
            _run(monkeypatch, ctx, block_ids, native_objs, D2H, "native", batch_size, 0)
            == 1
        )
        for f, n in zip(fallback_objs, native_objs, strict=True):
            assert torch.equal(_object_bytes(f), _object_bytes(n))
            assert int(_object_bytes(n).sum()) != 0  # something was copied
        for obj in fallback_objs + native_objs:
            allocator.free(obj)
    finally:
        ctx.close()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_retrieve_native_matches_fallback(
    monkeypatch: pytest.MonkeyPatch, allocator: LazyMemoryAllocator, name: str
):
    """H2D with batch size 4 and a skipped token prefix: identical pages."""
    ctx, tensors = _make_context(name)
    _, _, chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(chunk, 0)
    block_size = chunk // blocks_per_chunk
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    # Skip one whole chunk plus one block of the next one.
    skip_first_n_tokens = chunk + block_size
    try:
        objs = _allocate_objects(allocator, ctx, num_chunks, fill=True)
        before = [t.clone() for t in tensors]

        _run(
            monkeypatch,
            ctx,
            block_ids,
            objs,
            H2D,
            "fallback",
            ctx.max_batch_size,
            skip_first_n_tokens,
        )
        after_fallback = [t.clone() for t in tensors]
        for t, snap in zip(tensors, before, strict=True):
            t.copy_(snap)

        assert (
            _run(
                monkeypatch,
                ctx,
                block_ids,
                objs,
                H2D,
                "native",
                ctx.max_batch_size,
                skip_first_n_tokens,
            )
            == 1
        )
        for f, n in zip(after_fallback, tensors, strict=True):
            assert _bitwise_equal(f, n)
        assert any(
            not _bitwise_equal(f, snap)
            for f, snap in zip(after_fallback, before, strict=True)
        )
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()


def _hand_built_plan(
    ctx: XPUCacheContext, block_ids_gpu: torch.Tensor
) -> tuple[list[Any], Any]:
    """One kernel-group spec plus a valid single-object H2D launch."""
    spec = device_ops.KernelGroupSpec(
        ctx.get_kernel_group_kv_pointers(0).data_ptr(),
        [
            ctx.get_temp_kernel_group_buffer(slot, 0).data_ptr()
            for slot in range(ctx.max_batch_size)
        ],
        ctx.get_shape_desc(0),
        ctx.get_slots_per_chunk_in_sw(0),
        ctx.get_engine_kv_format(0),
        block_ids_gpu.data_ptr(),
        block_ids_gpu.numel(),
    )
    blocks = ctx.calculate_num_blocks(ctx.lmcache_tokens_per_chunk, 0)
    good = device_ops.LaunchVar(
        group_idx=0,
        block_ids_offset=0,
        total_blocks=blocks,
        num_objects=1,
        skip_prefix_n_blocks=0,
    )
    return [spec], good


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"group_idx": 1}, id="group_idx-out-of-range"),
        pytest.param({"num_objects": 5}, id="num_objects-over-slots"),
        pytest.param({"num_objects": 0}, id="num_objects-zero"),
        pytest.param({"block_ids_offset": 16}, id="block_ids-past-capacity"),
        pytest.param({"total_blocks": 32}, id="total_blocks-over-objects"),
        pytest.param({"skip_prefix_n_blocks": -1}, id="negative-skip"),
    ],
)
def test_executor_rejects_bad_launch_before_enqueuing(
    allocator: LazyMemoryAllocator, bad: dict
):
    """A bad launch anywhere in the plan raises before any staging copy or
    kernel of an earlier, valid step is enqueued."""
    ctx, tensors = _make_context("nhd_16")
    blocks = ctx.calculate_num_blocks(ctx.lmcache_tokens_per_chunk, 0)
    try:
        objs = _allocate_objects(allocator, ctx, 1, fill=True)
        block_ids_gpu = torch.arange(blocks, dtype=torch.long, device=_DEVICE)
        specs, good = _hand_built_plan(ctx, block_ids_gpu)
        staging = build_staging_copies(
            objs, [ctx.get_temp_object_group_buffer(0, 0)], True
        )
        fields = {
            "group_idx": 0,
            "block_ids_offset": 0,
            "total_blocks": blocks,
            "num_objects": 1,
            "skip_prefix_n_blocks": 0,
        }
        fields.update(bad)
        steps = [
            device_ops.BatchStep(staging, [good]),
            device_ops.BatchStep(staging, [device_ops.LaunchVar(**fields)]),
        ]
        before = [t.clone() for t in tensors]
        with pytest.raises(RuntimeError):
            device_ops.execute_object_group_transfer(
                int(H2D), ctx.device, LazyMemoryAllocator.PIN_CHUNK_SIZE, specs, steps
            )
        ctx.stream.synchronize()
        torch_dev.synchronize()
        for t, snap in zip(tensors, before, strict=True):
            assert _bitwise_equal(t, snap)

        # The valid step alone does write the pages (the check above is not
        # vacuous).
        device_ops.execute_object_group_transfer(
            int(H2D), ctx.device, LazyMemoryAllocator.PIN_CHUNK_SIZE, specs, steps[:1]
        )
        torch_dev.synchronize()
        assert any(
            not _bitwise_equal(t, snap) for t, snap in zip(tensors, before, strict=True)
        )
        allocator.free(objs[0])
    finally:
        ctx.close()


def test_executor_rejects_host_paged_pointer(allocator: LazyMemoryAllocator):
    """The paged-pointer array must be device USM; a host tensor is refused."""
    ctx, _tensors = _make_context("nhd_16")
    blocks = ctx.calculate_num_blocks(ctx.lmcache_tokens_per_chunk, 0)
    try:
        objs = _allocate_objects(allocator, ctx, 1, fill=True)
        block_ids_gpu = torch.arange(blocks, dtype=torch.long, device=_DEVICE)
        host_ptrs = ctx.get_kernel_group_kv_pointers(0).cpu()
        spec = device_ops.KernelGroupSpec(
            host_ptrs.data_ptr(),
            [
                ctx.get_temp_kernel_group_buffer(slot, 0).data_ptr()
                for slot in range(ctx.max_batch_size)
            ],
            ctx.get_shape_desc(0),
            ctx.get_slots_per_chunk_in_sw(0),
            ctx.get_engine_kv_format(0),
            block_ids_gpu.data_ptr(),
            block_ids_gpu.numel(),
        )
        _, good = _hand_built_plan(ctx, block_ids_gpu)
        with pytest.raises(RuntimeError, match="device"):
            device_ops.execute_object_group_transfer(
                int(H2D),
                ctx.device,
                LazyMemoryAllocator.PIN_CHUNK_SIZE,
                [spec],
                [device_ops.BatchStep([], [good])],
            )
        allocator.free(objs[0])
    finally:
        ctx.close()
