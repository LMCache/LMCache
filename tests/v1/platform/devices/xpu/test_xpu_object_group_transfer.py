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

The hybrid cases register several engine groups at once (full attention,
sliding windows shorter and longer than a chunk, a recurrent-state group, a
padded MLA pool and a compressed group) and check the round trip against the
pages themselves, so they do not depend on the object layout either path
picks. The padded-stride cases compare every block-axis format against its
tight twin.
"""

# Standard
from collections.abc import Callable
from typing import Any, NamedTuple
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
from lmcache.v1.multiprocess.group_view import EngineGroupInfo  # noqa: E402
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (  # noqa: E402
    get_layout_desc,
)
from lmcache.v1.multiprocess.object_group_transfer import (  # noqa: E402
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.platform.devices.xpu.cache_context import (  # noqa: E402
    XPUCacheContext,
    list_to_xpu_tensor,
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
    object_group_id: int = 0,
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
            # Downsampling rewrites the lists in place; keep the caller's.
            block_ids_gpu = downsample_and_stage_block_ids(
                ctx, [list(ids) for ids in block_ids]
            )
            transfer_kv_per_object_group(
                ctx,
                block_ids_gpu,
                objs,
                object_group_id=object_group_id,
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


# ------------------------------------------------------------------ #
#  Hybrid KV cache groups                                              #
# ------------------------------------------------------------------ #

_HYBRID_CHUNK = 64
_HYBRID_NUM_CHUNKS = 5  # one full batch of 4 plus a partial one
_TWO_NB = lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
_NB_TWO = lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
_NB_BS_HS = lmcache_native.EngineKVFormat.NL_X_NB_BS_HS


class _Layer(NamedTuple):
    """A registered KV view plus the allocation it lives in."""

    backing: torch.Tensor
    view: Callable[[torch.Tensor], torch.Tensor]


def _hybrid_layers() -> tuple[list[_Layer], list[EngineGroupInfo]]:
    """One engine group per hybrid feature; each becomes one kernel group."""

    def plain(shape: list[int]) -> _Layer:
        return _Layer(torch.rand(shape, dtype=_DTYPE, device=_DEVICE), lambda t: t)

    attn = [2, 64, 16, 2, 64]
    # Padded MLA pool: 576 payload elements per token, 640 allocated.
    padded = _Layer(
        torch.rand(64, 16 * 640, dtype=_DTYPE, device=_DEVICE),
        lambda t: t.as_strided((64, 16, 576), (16 * 640, 576, 1)),
    )
    layers = [
        plain(attn),
        plain(attn),
        plain(attn),
        plain(attn),
        plain(attn),
        # Mamba/GDN state re-viewed as a [NB, 2, BS, NH, HS] page
        # (_MambaStateViewEdit), one block per chunk.
        plain([16, 2, _HYBRID_CHUNK, 1, 32]),
        padded,
        # 4x compressed: 16 slots stand for 64 tokens.
        plain([64, 16, 512]),
    ]
    groups = [
        EngineGroupInfo(0, (0, 1), tokens_per_block=16),
        EngineGroupInfo(1, (2, 3), tokens_per_block=16, sw_size_tokens=32),
        EngineGroupInfo(2, (4,), tokens_per_block=16, sw_size_tokens=128),
        EngineGroupInfo(
            3,
            (5,),
            tokens_per_block=_HYBRID_CHUNK,
            sw_size_tokens=_HYBRID_CHUNK,
            recurrent_state=True,
        ),
        EngineGroupInfo(4, (6,), tokens_per_block=16),
        EngineGroupInfo(5, (7,), tokens_per_block=_HYBRID_CHUNK),
    ]
    return layers, groups


def _make_hybrid_context(
    separate_object_groups: bool,
) -> tuple[XPUCacheContext, list[_Layer], list[EngineGroupInfo]]:
    layers, groups = _hybrid_layers()
    ctx = XPUCacheContext(
        [_FakeIPCWrapper(layer.view(layer.backing)) for layer in layers],  # type: ignore[misc]
        lmcache_tokens_per_chunk=_HYBRID_CHUNK,
        engine_group_infos=groups,
        separate_object_groups=separate_object_groups,
    )
    manager = ctx.kv_layer_groups_manager
    # The expectations below index kernel groups by engine group.
    assert [g.engine_group_idx for g in manager.kernel_groups] == list(range(6))
    assert [ctx.get_engine_kv_format(kg) for kg in range(6)] == [
        _TWO_NB,
        _TWO_NB,
        _TWO_NB,
        _NB_TWO,
        _NB_BS_HS,
        _NB_BS_HS,
    ]
    assert manager.get_shape_desc(4).block_stride_elems == 16 * 640
    expected_groups = (
        [[0, 4, 5], [1], [2], [3]] if separate_object_groups else [list(range(6))]
    )
    assert [og.kernel_group_indices for og in manager.object_groups] == expected_groups
    return ctx, layers, groups


def _hybrid_block_ids(
    ctx: XPUCacheContext, num_chunks: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Disjoint source (lower half) and destination (upper half) block ids."""
    src, dst = [], []
    for kg in range(ctx.kv_layer_groups_manager.num_kernel_groups):
        count = num_chunks * ctx.calculate_num_blocks(_HYBRID_CHUNK, kg)
        nb = ctx.get_shape_desc(kg).nb
        src.append(random.sample(range(nb // 2), count))
        dst.append(random.sample(range(nb // 2, nb), count))
    return src, dst


def _allocate_group_objects(
    allocator: LazyMemoryAllocator, ctx: XPUCacheContext, object_group_id: int
) -> list[MemoryObj]:
    layout = get_layout_desc(ctx, _HYBRID_CHUNK, object_group_id)
    objs: list[MemoryObj] = []
    for _ in range(_HYBRID_NUM_CHUNKS):
        obj = allocator.allocate(layout.shapes, layout.dtypes)
        if obj is None:
            raise RuntimeError("allocator out of space")
        _object_view(obj).zero_()
        objs.append(obj)
    return objs


def _store_all_groups(
    monkeypatch: pytest.MonkeyPatch,
    allocator: LazyMemoryAllocator,
    ctx: XPUCacheContext,
    block_ids: list[list[int]],
    mode: str,
    batch_size: int,
) -> list[list[MemoryObj]]:
    objs_per_group = []
    for og in range(ctx.kv_layer_groups_manager.num_object_groups):
        objs = _allocate_group_objects(allocator, ctx, og)
        calls = _run(monkeypatch, ctx, block_ids, objs, D2H, mode, batch_size, 0, og)
        assert calls == (1 if mode == "native" else 0)
        objs_per_group.append(objs)
    return objs_per_group


def _retrieve_all_groups(
    monkeypatch: pytest.MonkeyPatch,
    ctx: XPUCacheContext,
    block_ids: list[list[int]],
    objs_per_group: list[list[MemoryObj]],
    mode: str,
    skip_first_n_tokens: int,
) -> None:
    for og, objs in enumerate(objs_per_group):
        _run(
            monkeypatch,
            ctx,
            block_ids,
            objs,
            H2D,
            mode,
            ctx.max_batch_size,
            skip_first_n_tokens,
            og,
        )


def _block_slice(fmt: Any, ids: list[int]) -> tuple:
    return (slice(None), ids) if fmt == _TWO_NB else (ids,)


def _expected_after_round_trip(
    ctx: XPUCacheContext,
    layers: list[_Layer],
    groups: list[EngineGroupInfo],
    snapshot: list[torch.Tensor],
    src: list[list[int]],
    dst: list[list[int]],
) -> tuple[list[torch.Tensor], list[int]]:
    """Copy the restorable source pages onto their destination ids.

    Per kernel group, a chunk restores only the blocks inside the sub-chunk
    window, and an object group with a cross-chunk window restores only its
    last ``num_chunks_in_sw`` chunks. Returns the expected backing tensors
    and the number of restored blocks per kernel group.
    """
    manager = ctx.kv_layer_groups_manager
    windows = manager.get_attn_desc().num_chunks_in_sw
    expected = [t.clone() for t in snapshot]
    restored = [0] * manager.num_kernel_groups
    for og_id, og in enumerate(manager.object_groups):
        window = windows[og_id]
        first_chunk = 0 if window == -1 else max(0, _HYBRID_NUM_CHUNKS - window)
        for kg in og.kernel_group_indices:
            per_chunk = ctx.calculate_num_blocks(_HYBRID_CHUNK, kg)
            keep = ctx.calculate_num_blocks(
                min(_HYBRID_CHUNK, manager.get_subchunk_sw_size_tokens(kg)), kg
            )
            positions = [
                c * per_chunk + j
                for c in range(first_chunk, _HYBRID_NUM_CHUNKS)
                for j in range(per_chunk - keep, per_chunk)
            ]
            src_ids = [src[kg][p] for p in positions]
            dst_ids = [dst[kg][p] for p in positions]
            restored[kg] = len(positions)
            fmt = ctx.get_engine_kv_format(kg)
            for layer_idx in groups[kg].layer_indices:
                layer = layers[layer_idx]
                target = layer.view(expected[layer_idx])
                source = layer.view(snapshot[layer_idx])
                target[_block_slice(fmt, dst_ids)] = source[_block_slice(fmt, src_ids)]
    return expected, restored


def _free_all(allocator: LazyMemoryAllocator, groups: list[list[MemoryObj]]):
    for objs in groups:
        for obj in objs:
            allocator.free(obj)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("separate", [False, True], ids=["merged", "separate"])
def test_hybrid_store_native_matches_fallback(
    monkeypatch: pytest.MonkeyPatch,
    allocator: LazyMemoryAllocator,
    separate: bool,
    batch_size: int,
):
    """D2H of every object group of a hybrid model: identical object bytes."""
    ctx, _layers, _groups = _make_hybrid_context(separate)
    src, _ = _hybrid_block_ids(ctx, _HYBRID_NUM_CHUNKS)
    try:
        fallback = _store_all_groups(
            monkeypatch, allocator, ctx, src, "fallback", batch_size
        )
        native = _store_all_groups(
            monkeypatch, allocator, ctx, src, "native", batch_size
        )
        for f_objs, n_objs in zip(fallback, native, strict=True):
            for f, n in zip(f_objs, n_objs, strict=True):
                assert torch.equal(_object_bytes(f), _object_bytes(n))
                assert int(_object_bytes(n).sum()) != 0
        _free_all(allocator, fallback + native)
    finally:
        ctx.close()


@pytest.mark.parametrize("mode", ["native", "fallback"])
@pytest.mark.parametrize("separate", [False, True], ids=["merged", "separate"])
def test_hybrid_round_trip_restores_window(
    monkeypatch: pytest.MonkeyPatch,
    allocator: LazyMemoryAllocator,
    separate: bool,
    mode: str,
):
    """Store from one set of blocks, retrieve into another: exactly the
    in-window source pages land on their destinations, and nothing else
    (other blocks, the padded tail of the MLA pool) changes."""
    ctx, layers, groups = _make_hybrid_context(separate)
    src, dst = _hybrid_block_ids(ctx, _HYBRID_NUM_CHUNKS)
    try:
        objs = _store_all_groups(
            monkeypatch, allocator, ctx, src, mode, ctx.max_batch_size
        )
        snapshot = [layer.backing.clone() for layer in layers]
        _retrieve_all_groups(monkeypatch, ctx, dst, objs, mode, 0)
        expected, restored = _expected_after_round_trip(
            ctx, layers, groups, snapshot, src, dst
        )
        for layer_idx, layer in enumerate(layers):
            assert _bitwise_equal(layer.backing, expected[layer_idx]), layer_idx
        assert all(n > 0 for n in restored)
        # The windows did cut something. Merged, only the sub-chunk window
        # applies (2 of 4 blocks per chunk). Separate, each window group also
        # restores just its trailing chunks: the 32-token window the last
        # chunk's last 2 blocks, the 128-token window the last 2 chunks, the
        # recurrent group the last state block.
        if separate:
            assert restored[:4] == [_HYBRID_NUM_CHUNKS * 4, 2, 2 * 4, 1]
        else:
            assert restored[:4] == [_HYBRID_NUM_CHUNKS * n for n in (4, 2, 4, 1)]
        _free_all(allocator, objs)
    finally:
        ctx.close()


@pytest.mark.parametrize("separate", [False, True], ids=["merged", "separate"])
def test_hybrid_retrieve_native_matches_fallback_with_skip(
    monkeypatch: pytest.MonkeyPatch,
    allocator: LazyMemoryAllocator,
    separate: bool,
):
    """H2D with a skipped prefix across mixed block sizes: identical pages."""
    ctx, layers, _groups = _make_hybrid_context(separate)
    src, dst = _hybrid_block_ids(ctx, _HYBRID_NUM_CHUNKS)
    # One chunk plus one 16-token block: splits the 16-token groups mid-chunk
    # and covers the 64-token-block groups' whole first chunk.
    skip = _HYBRID_CHUNK + 16
    try:
        objs = _store_all_groups(
            monkeypatch, allocator, ctx, src, "native", ctx.max_batch_size
        )
        before = [layer.backing.clone() for layer in layers]
        _retrieve_all_groups(monkeypatch, ctx, dst, objs, "fallback", skip)
        after_fallback = [layer.backing.clone() for layer in layers]
        for layer, snap in zip(layers, before, strict=True):
            layer.backing.copy_(snap)
        _retrieve_all_groups(monkeypatch, ctx, dst, objs, "native", skip)
        for layer, f, snap in zip(layers, after_fallback, before, strict=True):
            assert _bitwise_equal(layer.backing, f)
            assert not _bitwise_equal(f, snap)
        _free_all(allocator, objs)
    finally:
        ctx.close()


# ------------------------------------------------------------------ #
#  Padded block stride                                                 #
# ------------------------------------------------------------------ #

# Every format whose dim-0 is the block axis, i.e. every format the
# connector may report a padded ``block_stride_elems`` for
# (``_BLOCK_AXIS_FORMATS``, minus the plane tuple the SYCL kernel refuses).
# name -> (format, bs, nh, hs, dtype)
_PADDED_CASES = {
    "mla": (_NB_BS_HS, 16, 1, 576, torch.bfloat16),
    "blocked_scale": (
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        64,
        1,
        132,
        torch.uint8,
    ),
    "cs_hnd": (
        lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS,
        16,
        2,
        96,
        torch.bfloat16,
    ),
    "cs_nhd": (
        lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS,
        16,
        2,
        96,
        torch.bfloat16,
    ),
    "nhd_single": (
        lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_HS,
        16,
        4,
        64,
        torch.bfloat16,
    ),
}
_PAD_NB = 32
_PAD_LAYERS = 2
_PAD_CHUNK = 64
_PAD_OBJECTS = 3
_PAD_EXTRA = 64  # padding elements per block


def _random_like(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, device=_DEVICE)
    return torch.rand(shape, dtype=dtype, device=_DEVICE)


def _pad_transfer(
    pools: list[torch.Tensor],
    objects: list[torch.Tensor],
    ids: torch.Tensor,
    direction: "lmcache_native.TransferDirection",
    name: str,
    block_stride_elems: int,
) -> None:
    fmt, bs, nh, hs, dtype = _PADDED_CASES[name]
    desc = lmcache_native.PageBufferShapeDesc()
    desc.kv_size, desc.nl, desc.nb = 1, _PAD_LAYERS, _PAD_NB
    desc.bs, desc.nh, desc.hs = bs, nh, hs
    desc.element_size = torch.empty(0, dtype=dtype).element_size()
    desc.block_stride_elems = block_stride_elems
    device_ops.multi_layer_block_kv_transfer(
        list_to_xpu_tensor([p.data_ptr() for p in pools], _DEVICE),
        [o.data_ptr() for o in objects],
        ids,
        _DEVICE,
        direction,
        desc,
        _PAD_CHUNK,
        fmt,
        0,
    )
    torch_dev.synchronize()


@pytest.mark.parametrize("name", sorted(_PADDED_CASES))
def test_padded_block_stride_matches_tight_pool(name: str):
    """A padded pool moves the same bytes as its tight twin, and H2D never
    writes the padding."""
    _, bs, nh, hs, dtype = _PADDED_CASES[name]
    tight_block = bs * nh * hs
    stride = tight_block + _PAD_EXTRA
    padded = [_random_like((_PAD_NB, stride), dtype) for _ in range(_PAD_LAYERS)]
    tight = [p[:, :tight_block].contiguous() for p in padded]
    ids = torch.tensor(
        random.sample(range(_PAD_NB), _PAD_OBJECTS * _PAD_CHUNK // bs),
        dtype=torch.long,
        device=_DEVICE,
    )
    object_elems = _PAD_LAYERS * _PAD_CHUNK * nh * hs

    # D2H: same object bytes from both pools.
    from_padded = [
        torch.zeros(object_elems, dtype=dtype, device=_DEVICE)
        for _ in range(_PAD_OBJECTS)
    ]
    from_tight = [torch.zeros_like(o) for o in from_padded]
    _pad_transfer(padded, from_padded, ids, D2H, name, stride)
    _pad_transfer(tight, from_tight, ids, D2H, name, 0)
    for p, t in zip(from_padded, from_tight, strict=True):
        assert torch.equal(p, t)
        assert bool(t.ne(0).any())

    # H2D: same payload in both pools; the padding keeps its old bytes.
    objects = [_random_like((object_elems,), dtype) for _ in range(_PAD_OBJECTS)]
    before = [p.clone() for p in padded]
    _pad_transfer(padded, objects, ids, H2D, name, stride)
    _pad_transfer(tight, objects, ids, H2D, name, 0)
    for p, t, snap in zip(padded, tight, before, strict=True):
        assert torch.equal(p[:, :tight_block], t)
        assert torch.equal(p[:, tight_block:], snap[:, tight_block:])
        assert not torch.equal(p, snap)
