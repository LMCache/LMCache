# SPDX-License-Identifier: Apache-2.0
"""Public-planner roundtrips and observed bytes for covered-prefix staging."""

# Standard
from collections.abc import Iterator
import random
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops, torch_dev, torch_device_type
from lmcache.v1.common.group_view import EngineGroupInfo
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.ops_types import TransferPhase
import lmcache.lmcache_native as native
import lmcache.v1.multiprocess.object_group_transfer as transfer

pytest.importorskip("cupy", reason="GPU cache context requires cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires CUDA extension")

# First Party
from lmcache.v1.platform.devices.cuda.cache_context import GPUCacheContext  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires CUDA backend"),
    pytest.mark.skipif(
        not hasattr(device_ops, "pop_completed_phase_timings"),
        reason="Requires native phase byte accounting",
    ),
]


class _LocalTensor(DeviceIPCWrapper):
    """Expose a local tensor through the IPC wrapper's public interface."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = tensor.storage_offset()
        self.device_uuid = str(torch_dev.get_device_properties(tensor.device).uuid)

    def to_tensor(self) -> torch.Tensor:
        return self.tensor

    def close(self) -> None:
        pass


@pytest.fixture
def timing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enable native public byte samples without starting a global event bus."""
    bus = EventBus(EventBusConfig(enabled=True))
    bus.subscribe(EventType.MP_TRANSFER_PHASE_SAMPLES, lambda event: None)
    monkeypatch.setattr(transfer, "is_observability_enabled", lambda: True)
    monkeypatch.setattr(transfer, "get_event_bus", lambda: bus)
    device_ops.pop_completed_phase_timings()
    yield
    device_ops.pop_completed_phase_timings()
    bus.stop()


def _allocate(
    allocator: LazyMemoryAllocator,
    ctx: GPUCacheContext,
    count: int,
) -> list[MemoryObj]:
    """Wait for the lazy allocator's background pin expansion, then fill objects."""
    layouts = [
        ctx.get_kernel_group_shape_dtype(ctx.lmcache_tokens_per_chunk, kg)
        for kg in ctx.kv_layer_groups_manager.object_groups[0].kernel_group_indices
    ]
    deadline = time.monotonic() + 10
    while True:
        objects = allocator.batched_allocate(
            [shape for shape, _ in layouts], [dtype for _, dtype in layouts], count
        )
        if objects is not None:
            break
        if time.monotonic() >= deadline:
            pytest.fail("lazy allocator did not finish expanding")
        time.sleep(0.001)
    generator = torch.Generator().manual_seed(23)
    for obj in objects:
        raw = obj.raw_tensor
        assert raw is not None
        raw.copy_(
            torch.randint(0, 256, raw.shape, dtype=torch.uint8, generator=generator)
        )
    return objects


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    ("chunk", "skip", "count", "heads", "padding", "expected_bytes"),
    [
        (4096, 0, 2, (4,), False, 16 << 20),
        (4096, 16, 2, (4,), False, 16 << 20),
        (4096, 2032, 2, (4,), False, 16 << 20),
        (4096, 2048, 2, (4,), False, 12 << 20),
        (4096, 4080, 2, (4,), False, (8 << 20) + (32 << 10)),
        (4096, 4096, 2, (4,), False, 8 << 20),
        (4096, 4112, 2, (4,), False, 8 << 20),
        (4096, 6144, 2, (4,), False, 4 << 20),
        (4096, 8192, 2, (4,), False, 0),
        (4096, 22528, 6, (4,), False, 4 << 20),
        (1024, 1008, 2, (4,), False, 4 << 20),
        (16384, 2048, 2, (4,), False, 64 << 20),
        (16384, 8192, 2, (4,), False, 48 << 20),
        (4096, 2048, 2, (4,), True, 12 << 20),
        (4096, 2048, 2, (4, 8), False, 36 << 20),
        (4096, 2048, 2, (4, 1), False, 20 << 20),
    ],
)
def test_retrieve_preserves_pages_and_accounts_staging(
    timing: None,
    chunk: int,
    skip: int,
    count: int,
    heads: tuple[int, ...],
    padding: bool,
    expected_bytes: int,
    batch_size: int,
) -> None:
    """Only requested suffix pages change; phase samples report actual H2D bytes.

    The pin-boundary case starts an object at 60 MiB + 4 KiB in a pool with
    separate initial and expanded 64 MiB registrations. A row tail crosses
    the registration boundary. Multiple groups exercise nonzero region offsets.
    """
    blocks_per_chunk = chunk // 16
    num_blocks = count * blocks_per_chunk + 7
    tensors = [
        torch.full(
            (2, num_blocks, 16, nh, 128),
            165,
            dtype=torch.uint8,
            device=torch.device(f"{torch_device_type}:0"),
        )
        for nh in heads
        for _ in range(2)
    ]
    ctx = GPUCacheContext(
        [_LocalTensor(t) for t in tensors],
        lmcache_tokens_per_chunk=chunk,
        engine_group_infos=[
            EngineGroupInfo(i, (2 * i, 2 * i + 1)) for i in range(len(heads))
        ],
        separate_object_groups=False,
    )
    allocator = LazyMemoryAllocator(init_size=64 << 20, final_size=128 << 20)
    pad = None
    objects: list[MemoryObj] = []
    try:
        if padding:
            pad = allocator.allocate(torch.Size([(60 << 20) + 4096]), torch.uint8)
            assert pad is not None
        objects = _allocate(allocator, ctx, count)
        sources = []
        for obj in objects:
            raw = obj.raw_tensor
            assert raw is not None
            sources.append(raw.clone())
        ids = [
            random.Random(7 + kg).sample(range(num_blocks), count * blocks_per_chunk)
            for kg in range(len(heads))
        ]
        expected = [torch.full(t.shape, 165, dtype=torch.uint8) for t in tensors]
        for obj_index, obj in enumerate(objects):
            for kg, group in enumerate(ctx.kv_layer_groups_manager.kernel_groups):
                source = obj.get_tensor(kg)
                assert source is not None
                for block in range(blocks_per_chunk):
                    flat_block = obj_index * blocks_per_chunk + block
                    if flat_block < skip // 16:
                        continue
                    for layer, tensor_index in enumerate(group.layer_indices):
                        expected[tensor_index][:, ids[kg][flat_block]].copy_(
                            source[:, layer, block * 16 : (block + 1) * 16].reshape(
                                2, 16, heads[kg], 128
                            )
                        )
        # Initial page fills use the default stream; the planner uses ctx.stream.
        torch_dev.synchronize()
        with torch_dev.stream(ctx.stream):
            gpu_ids = transfer.downsample_and_stage_block_ids(
                ctx, [list(x) for x in ids]
            )
            transfer.transfer_kv_per_object_group(
                ctx,
                gpu_ids,
                objects,
                0,
                batch_size,
                skip,
                native.TransferDirection.H2D,
                transfer_key="prefix-staging-test",
            )
        ctx.stream.synchronize()
        assert all(
            torch.equal(t.cpu(), e) for t, e in zip(tensors, expected, strict=True)
        )
        assert all(
            torch.equal(obj.raw_tensor, original)
            for obj, original in zip(objects, sources, strict=True)
        )
        samples = device_ops.pop_completed_phase_timings()
        assert (
            sum(
                sample[4]
                for sample in samples
                if sample[0] == int(TransferPhase.STAGING)
                and sample[5] == "prefix-staging-test"
            )
            == expected_bytes
        )
    finally:
        ctx.stream.synchronize()
        for obj in objects:
            allocator.free(obj)
        if pad is not None:
            allocator.free(pad)
        allocator.close()
        ctx.close()


def test_compressed_window_prefix(timing: None) -> None:
    """Logical skip positions map to the retained physical SWA slots."""
    tensors = [
        torch.full(
            (2, 519, 16, 4, 128),
            165,
            dtype=torch.uint8,
            device=torch.device(f"{torch_device_type}:0"),
        )
        for _ in range(2)
    ]
    ctx = GPUCacheContext(
        [_LocalTensor(t) for t in tensors],
        lmcache_tokens_per_chunk=16384,
        engine_group_infos=[
            EngineGroupInfo(0, (0, 1), tokens_per_block=32, sw_size_tokens=8192)
        ],
    )
    allocator = LazyMemoryAllocator(init_size=64 << 20, final_size=64 << 20)
    objects: list[MemoryObj] = []
    try:
        objects = _allocate(allocator, ctx, 1)
        source = objects[0].get_tensor(0)
        assert source is not None
        before = source.clone()
        ids = random.Random(19).sample(range(519), 512)
        expected = [torch.full(t.shape, 165, dtype=torch.uint8) for t in tensors]
        for block in range(448, 512):
            for layer in range(2):
                expected[layer][:, ids[block]].copy_(
                    source[:, layer, (block - 256) * 16 : (block - 255) * 16].reshape(
                        2, 16, 4, 128
                    )
                )
        torch_dev.synchronize()
        with torch_dev.stream(ctx.stream):
            gpu_ids = transfer.downsample_and_stage_block_ids(ctx, [ids])
            transfer.transfer_kv_per_object_group(
                ctx,
                gpu_ids,
                objects,
                0,
                4,
                14336,
                native.TransferDirection.H2D,
                transfer_key="compressed-prefix-test",
            )
        ctx.stream.synchronize()
        assert all(
            torch.equal(t.cpu(), e) for t, e in zip(tensors, expected, strict=True)
        )
        assert torch.equal(source, before)
        assert (
            sum(
                sample[4]
                for sample in device_ops.pop_completed_phase_timings()
                if sample[0] == int(TransferPhase.STAGING)
                and sample[5] == "compressed-prefix-test"
            )
            == 2 << 20
        )
    finally:
        ctx.stream.synchronize()
        for obj in objects:
            allocator.free(obj)
        allocator.close()
        ctx.close()


@pytest.mark.parametrize("invalid", ["none", "size"])
def test_covered_objects_still_validate(invalid: str) -> None:
    """A covered object inside a live batch still obeys H2D validation."""
    tensors = [
        torch.full(
            (2, 512, 16, 4, 128),
            165,
            dtype=torch.uint8,
            device=torch.device(f"{torch_device_type}:0"),
        )
        for _ in range(2)
    ]
    ctx = GPUCacheContext(
        [_LocalTensor(t) for t in tensors],
        lmcache_tokens_per_chunk=4096,
    )
    allocator = LazyMemoryAllocator(init_size=64 << 20, final_size=64 << 20)
    objects: list[MemoryObj] = []
    bad = None
    try:
        objects = _allocate(allocator, ctx, 1)
        if invalid == "size":
            bad = allocator.allocate(torch.Size([16]), torch.uint8)
            assert bad is not None
        torch_dev.synchronize()
        with torch_dev.stream(ctx.stream):
            gpu_ids = transfer.downsample_and_stage_block_ids(ctx, [list(range(512))])
            with pytest.raises(ValueError, match="MemoryObj is None|Size mismatch"):
                transfer.transfer_kv_per_object_group(
                    ctx,
                    gpu_ids,
                    [bad, objects[0]],
                    0,
                    4,
                    6144,
                    native.TransferDirection.H2D,
                    transfer_key="invalid-prefix-test",
                )
        ctx.stream.synchronize()
        assert all(bool((t == 165).all()) for t in tensors)
    finally:
        ctx.stream.synchronize()
        for obj in objects:
            allocator.free(obj)
        if bad is not None:
            allocator.free(bad)
        allocator.close()
        ctx.close()
