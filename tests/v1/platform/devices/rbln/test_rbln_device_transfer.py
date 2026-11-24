# SPDX-License-Identifier: Apache-2.0
"""RBLN block transfer on a real NPU.

``test_rbln_kv_ops.py`` pins the chunk layout with CPU tensors, which never
exercises the device dispatch: the device staging buffers only diverge
from the CPU behaviour once the
paged KV lives on ``rbln``. These tests store from device-resident paged KV
into host chunks and retrieve them into a zeroed device cache, then compare
the chunks against the shared torch path on the CPU and the restored cache
against the source.

The paged KV must really be on the device. torch-rbln allocates device memory
lazily and keeps a tensor made with ``.to("rbln")`` host-latest, so a
transfer over it is served from the host copy and never crosses to the NPU.
The tests therefore bind each paged layer's device memory up front, as a
vLLM-RBLN device-tensor KV cache is, and run the transfer under
``torch.rbln.explain()`` to fail if any copy is served on the host instead.

Each test prepares its KV geometry first, as an engine does before serving,
so the staging and swap programs exist before any transfer.

They skip unless ``torch.rbln`` reports a usable NPU, so they run only on the
``rbln-mp-test`` Buildkite lane.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.client.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.platform.devices.rbln import RblnDeviceSpec, warm_up_kv_transfer
from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops import multi_layer_block_kv_transfer
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 64
MLA_HEAD_SIZE = 16
BLOCKS_PER_CHUNK = 2
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.bfloat16

pytestmark = [
    pytest.mark.rbln,
    pytest.mark.skipif(
        not RblnDeviceSpec().is_available(), reason="RBLN NPU is required"
    ),
]


def _shape_desc(
    kv_size: int, num_heads: int, head_size: int, num_layers: int = NUM_LAYERS
) -> PageBufferShapeDesc:
    """Descriptor for ``num_layers`` x ``NUM_BLOCKS`` paged layers."""
    desc = PageBufferShapeDesc()
    desc.kv_size = kv_size
    desc.nl = num_layers
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = num_heads
    desc.hs = head_size
    desc.element_size = DTYPE.itemsize
    return desc


def _bound_device_tensor(source: torch.Tensor) -> torch.Tensor:
    """Copy ``source`` into a new ``rbln:0`` tensor with bound device memory."""
    tensor = torch.empty(source.shape, dtype=source.dtype, device="rbln:0")
    torch.rbln.bind_device_memory(tensor)
    tensor.copy_(source)
    return tensor


def _transfer(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
) -> None:
    """Prepare the layers' geometry and move every block to or from ``chunks``."""
    ops = RblnDeviceOps()
    ops.prepare_kv_transfer(layers, engine_kv_format)
    ops.multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)),
        layers[0].device,
        direction,
        shape_desc,
        CHUNK_TOKENS,
        engine_kv_format,
        0,
    )


def _canonical_chunks(
    layers: list[torch.Tensor],
    chunk_shape: tuple[int, ...],
    canonical_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
) -> list[torch.Tensor]:
    """Store every block through the shared torch path: the bytes to match."""
    chunks = [
        torch.zeros(chunk_shape, dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]
    multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        TransferDirection.D2H,
        shape_desc,
        CHUNK_TOKENS,
        canonical_format,
        0,
    )
    return chunks


def _assert_device_round_trip(
    layer_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    engine_kv_format: EngineKVFormat,
    canonical_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
) -> None:
    """Store and retrieve on ``rbln:0`` and compare against the shared path."""
    torch.manual_seed(7)
    cpu_layers = [torch.randn(layer_shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]
    num_chunks = NUM_BLOCKS // BLOCKS_PER_CHUNK
    # The shared path addresses HND without RBLN's singleton axis.
    canonical_layers = (
        [layer.squeeze(3) for layer in cpu_layers]
        if len(layer_shape) == 6
        else cpu_layers
    )
    expected_chunks = _canonical_chunks(
        canonical_layers, chunk_shape, canonical_format, shape_desc
    )

    device_layers = [_bound_device_tensor(layer) for layer in cpu_layers]
    restored = [_bound_device_tensor(torch.zeros_like(layer)) for layer in cpu_layers]
    RblnDeviceOps().prepare_kv_transfer(device_layers, engine_kv_format)
    torch.rbln.synchronize()
    chunks = [torch.zeros(chunk_shape, dtype=DTYPE) for _ in range(num_chunks)]
    with torch.rbln.explain() as region:
        _transfer(
            device_layers, chunks, TransferDirection.D2H, engine_kv_format, shape_desc
        )
        _transfer(restored, chunks, TransferDirection.H2D, engine_kv_format, shape_desc)
    assert region.verdict()["clean"], region.report()

    for chunk, expected in zip(chunks, expected_chunks, strict=True):
        assert torch.equal(chunk, expected)
    for layer, expected in zip(restored, cpu_layers, strict=True):
        assert torch.equal(layer.cpu(), expected)


def test_hnd_round_trip_on_device() -> None:
    """The 6-D HND cache survives store and retrieve on a real NPU."""
    _assert_device_round_trip(
        (NUM_BLOCKS, 2, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE),
        (2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE),
        EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS,
        EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        _shape_desc(kv_size=2, num_heads=NUM_HEADS, head_size=HEAD_SIZE),
    )


def test_kv_first_hnd_round_trip_on_device() -> None:
    """The K/V-first 6-D HND cache survives store and retrieve on a real NPU."""
    _assert_device_round_trip(
        (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE),
        (2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE),
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        _shape_desc(kv_size=2, num_heads=NUM_HEADS, head_size=HEAD_SIZE),
    )


def test_hnd_gather_scatter_on_device() -> None:
    """The raw 6-D cache round-trips through MP gather/scatter on a real NPU.

    Adds the link the round trips above skip: ``gather_paged_kv_to_cpu`` /
    ``scatter_cpu_to_paged_kv`` detect vLLM-RBLN's ``[NB, 2, NH, 1, BS, HS]``
    as its own format and dispatch to ``RblnDeviceOps``. The chunks must match
    the shared torch path on the same KV.
    """
    torch.manual_seed(13)
    cpu_layers = [
        torch.randn((NUM_BLOCKS, 2, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_LAYERS)
    ]
    expected = _canonical_chunks(
        [layer.squeeze(3) for layer in cpu_layers],
        (2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE),
        EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        _shape_desc(kv_size=2, num_heads=NUM_HEADS, head_size=HEAD_SIZE),
    )

    block_ids = list(range(NUM_BLOCKS))
    paged = {f"layer.{i}": _bound_device_tensor(t) for i, t in enumerate(cpu_layers)}
    restored = {
        name: _bound_device_tensor(torch.zeros_like(t))
        for name, t in zip(paged, cpu_layers, strict=True)
    }
    warm_up_kv_transfer(paged)
    torch.rbln.synchronize()
    with torch.rbln.explain() as region:
        chunks = gather_paged_kv_to_cpu(paged, block_ids, BLOCKS_PER_CHUNK)
        torch.rbln.synchronize()
        scatter_cpu_to_paged_kv(restored, block_ids, chunks, BLOCKS_PER_CHUNK)
        torch.rbln.synchronize()
    assert region.verdict()["clean"], region.report()

    for chunk, want in zip(chunks, expected, strict=True):
        assert torch.equal(chunk, want)
    for layer, want in zip(restored.values(), cpu_layers, strict=True):
        assert torch.equal(layer.cpu(), want)


def test_transfers_after_warm_up_build_no_program() -> None:
    """After warm-up, a worker-thread store and a restore build no device program.

    Warm-up runs on this thread outside inference mode, where vLLM-RBLN calls
    it. The store then runs as LMCache's async store does: on another thread,
    under ``torch.inference_mode()``, on a side stream. A program built there
    would race the model thread on torch's process-wide export state.
    ``torch._dynamo.reset()`` after warm-up drops the compiled copies
    torch-rbln would otherwise look up again, so only kept programs avoid a
    rebuild. ``torch.rbln.capture_programs()`` is per thread, so each transfer
    opens its own scope.
    """
    torch.manual_seed(29)
    cpu_layers = [
        torch.randn((NUM_BLOCKS, 2, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_LAYERS)
    ]
    paged = {f"layer.{i}": _bound_device_tensor(t) for i, t in enumerate(cpu_layers)}
    restored = {
        name: _bound_device_tensor(torch.zeros_like(t))
        for name, t in zip(paged, cpu_layers, strict=True)
    }
    block_ids = list(range(NUM_BLOCKS))
    warm_up_kv_transfer(paged)
    torch.rbln.synchronize()
    torch._dynamo.reset()

    def store() -> tuple[list[torch.Tensor], int]:
        side = torch.rbln.Stream()
        with (
            torch.inference_mode(),
            torch.rbln.stream(side),
            torch.rbln.capture_programs() as built,
        ):
            chunks = gather_paged_kv_to_cpu(paged, block_ids, BLOCKS_PER_CHUNK)
            side.synchronize()
        return chunks, len(built)

    with ThreadPoolExecutor(max_workers=1) as pool:
        chunks, store_built = pool.submit(store).result(timeout=60)
    with torch.inference_mode(), torch.rbln.capture_programs() as built:
        scatter_cpu_to_paged_kv(restored, block_ids, chunks, BLOCKS_PER_CHUNK)
        torch.rbln.synchronize()

    assert (store_built, len(built)) == (0, 0)
    for layer, want in zip(restored.values(), cpu_layers, strict=True):
        assert torch.equal(layer.cpu(), want)


def test_mla_round_trip_on_device() -> None:
    """The MLA cache survives store and retrieve on a real NPU."""
    _assert_device_round_trip(
        (NUM_BLOCKS, BLOCK_SIZE, MLA_HEAD_SIZE),
        (NUM_LAYERS, CHUNK_TOKENS, MLA_HEAD_SIZE),
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NB_BS_HS,
        _shape_desc(kv_size=1, num_heads=1, head_size=MLA_HEAD_SIZE),
    )


@pytest.mark.parametrize("kv_first", [False, True])
def test_hnd_back_to_back_without_synchronization(kv_first: bool) -> None:
    """Fresh bit patterns survive repeated transfers and staging reuse on the NPU."""
    shape = (
        (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
        if kv_first
        else (NUM_BLOCKS, 2, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    )
    fmt = (
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS
        if kv_first
        else EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS
    )
    desc = _shape_desc(2, NUM_HEADS, HEAD_SIZE)
    layers = [
        _bound_device_tensor(torch.zeros(shape, dtype=DTYPE)) for _ in range(NUM_LAYERS)
    ]
    other = [
        _bound_device_tensor(torch.zeros(shape, dtype=DTYPE)) for _ in range(NUM_LAYERS)
    ]
    chunk_shape = (2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE)
    for seed in range(4):
        generator = torch.Generator().manual_seed(173 + seed)
        source = [
            torch.randn(chunk_shape, dtype=DTYPE, generator=generator)
            for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
        ]
        stored = [torch.empty_like(chunk) for chunk in source]
        _transfer(layers, source, TransferDirection.H2D, fmt, desc)
        _transfer(other, source, TransferDirection.H2D, fmt, desc)
        _transfer(layers, stored, TransferDirection.D2H, fmt, desc)
        for got, expected in zip(stored, source, strict=True):
            assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
        _transfer(other, stored, TransferDirection.D2H, fmt, desc)
        for got, expected in zip(stored, source, strict=True):
            assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))


_FORMATS = {
    "blocks_first": (
        EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS,
        EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ),
    "kv_first": (
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ),
}


def _hnd_layers(
    num_layers: int, layout: str, seed: int, num_heads: int = NUM_HEADS
) -> list[torch.Tensor]:
    """Random host HND layers in ``layout``, as vLLM-RBLN allocates them."""
    generator = torch.Generator().manual_seed(seed)
    layers = [
        torch.randn(
            (NUM_BLOCKS, 2, num_heads, 1, BLOCK_SIZE, HEAD_SIZE),
            dtype=DTYPE,
            generator=generator,
        )
        for _ in range(num_layers)
    ]
    if layout == "kv_first":
        layers = [layer.transpose(0, 1).contiguous() for layer in layers]
    return layers


@pytest.mark.parametrize("layout", ["blocks_first", "kv_first"])
@pytest.mark.parametrize(
    "direction,skip",
    [
        (TransferDirection.D2H, 0),
        (TransferDirection.H2D, 0),
        (TransferDirection.H2D, 1),
        (TransferDirection.H2D, 3),
        (TransferDirection.H2D, 7),
    ],
)
@pytest.mark.parametrize("blocks_per_chunk", [1, 3])
def test_hnd_matches_the_shared_path_on_device(
    layout: str, direction: TransferDirection, skip: int, blocks_per_chunk: int
) -> None:
    """Each direction writes the shared path's bytes and leaves the rest alone.

    Three layers make six staging rows, so on a multi-chiplet NPU one
    chiplet's slice of the staging spans the K/V boundary. Block IDs are
    permuted, chunks start with random
    bytes, and the last multi-block chunk is partial. A restore skips a
    global block prefix, including one that ends inside a chunk.
    """
    num_layers, ids = 3, [5, 2, 7, 1, 6, 0, 3]
    engine_kv_format, canonical_format = _FORMATS[layout]
    desc = _shape_desc(2, NUM_HEADS, HEAD_SIZE, num_layers=num_layers)
    chunk_tokens = blocks_per_chunk * BLOCK_SIZE
    generator = torch.Generator().manual_seed(18)
    chunks = [
        torch.randn(
            (2, num_layers, chunk_tokens, NUM_HEADS * HEAD_SIZE),
            dtype=DTYPE,
            generator=generator,
        )
        for _ in range(-(-len(ids) // blocks_per_chunk))
    ]
    layers = _hnd_layers(num_layers, layout, seed=19)
    expected_layers = [layer.clone() for layer in layers]
    expected_chunks = [chunk.clone() for chunk in chunks]
    multi_layer_block_kv_transfer(
        [layer.squeeze(3) for layer in expected_layers],
        expected_chunks,
        ids,
        torch.device("cpu"),
        direction,
        desc,
        chunk_tokens,
        canonical_format,
        skip,
    )

    device_layers = [_bound_device_tensor(layer) for layer in layers]
    ops = RblnDeviceOps()
    ops.prepare_kv_transfer(device_layers, engine_kv_format)
    torch.rbln.synchronize()
    ops.multi_layer_block_kv_transfer(
        device_layers,
        chunks,
        ids,
        device_layers[0].device,
        direction,
        desc,
        chunk_tokens,
        engine_kv_format,
        skip,
    )
    torch.rbln.synchronize()

    for got, want in zip(
        [layer.cpu() for layer in device_layers] + chunks,
        expected_layers + expected_chunks,
        strict=True,
    ):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_prepared_geometries_are_kept_on_device() -> None:
    """Preparing a second geometry keeps the first one's staging and programs.

    Each KV cache group with its own layer count is a geometry, and a
    transfer may only use what warm-up prepared, so preparing one must not
    evict another.
    """
    engine_kv_format, canonical_format = _FORMATS["blocks_first"]
    three = [_bound_device_tensor(t) for t in _hnd_layers(3, "blocks_first", 23)]
    five = [_bound_device_tensor(t) for t in _hnd_layers(5, "blocks_first", 24)]
    ops = RblnDeviceOps()
    for layers in (three, five, three):
        ops.prepare_kv_transfer(layers, engine_kv_format)
    torch.rbln.synchronize()
    desc = _shape_desc(2, NUM_HEADS, HEAD_SIZE, num_layers=3)
    chunk_shape = (2, 3, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE)
    expected = _canonical_chunks(
        [layer.cpu().squeeze(3) for layer in three],
        chunk_shape,
        canonical_format,
        desc,
    )
    chunks = [
        torch.zeros(chunk_shape, dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]

    with torch.rbln.capture_programs() as built:
        ops.multi_layer_block_kv_transfer(
            three,
            chunks,
            list(range(NUM_BLOCKS)),
            three[0].device,
            TransferDirection.D2H,
            desc,
            CHUNK_TOKENS,
            engine_kv_format,
            0,
        )
        torch.rbln.synchronize()

    assert len(built) == 0
    for got, want in zip(chunks, expected, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize(
    "dtype,head_size",
    [(torch.uint8, HEAD_SIZE), (DTYPE, 32)],
    ids=["uint8-head-size-64", "head-size-32"],
)
def test_kv_without_a_swap_program_fails_at_warm_up(
    dtype: torch.dtype, head_size: int
) -> None:
    """KV torch-rbln builds no swap program for fails before serving.

    No torch-rbln builds the swap for a head size under 128 bytes, and
    releases that build it only for bf16 build none for uint8 KV at all. For
    such KV warm-up raises, and transfers of it keep failing before any copy
    instead of building on a commit worker.
    """
    shape = (NUM_BLOCKS, 2, NUM_HEADS, 1, BLOCK_SIZE, head_size)
    layers = [
        _bound_device_tensor(torch.randn(shape).to(dtype)) for _ in range(NUM_LAYERS)
    ]
    torch.rbln.synchronize()

    with pytest.raises(RuntimeError, match="built no program"):
        warm_up_kv_transfer({f"layer.{i}": t for i, t in enumerate(layers)})


def _mla_layers(num_layers: int, seed: int) -> list[torch.Tensor]:
    """Random host MLA layers ``[NB, BS, HS]``."""
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(
            (NUM_BLOCKS, BLOCK_SIZE, MLA_HEAD_SIZE), dtype=DTYPE, generator=generator
        )
        for _ in range(num_layers)
    ]


def _case(
    layout: str, num_layers: int, seed: int, num_heads: int = NUM_HEADS
) -> tuple[
    list[torch.Tensor],
    EngineKVFormat,
    EngineKVFormat,
    list[torch.Tensor],
    tuple[int, ...],
    PageBufferShapeDesc,
]:
    """Host layers of ``layout`` and what the shared path needs to match them.

    Returns:
        The layers, their engine format, the shared path's format, the
        layers as the shared path addresses them, one chunk's shape and the
        shape descriptor.
    """
    if layout == "mla":
        layers = _mla_layers(num_layers, seed)
        return (
            layers,
            EngineKVFormat.NL_X_NB_BS_HS,
            EngineKVFormat.NL_X_NB_BS_HS,
            layers,
            (num_layers, CHUNK_TOKENS, MLA_HEAD_SIZE),
            _shape_desc(1, 1, MLA_HEAD_SIZE, num_layers=num_layers),
        )
    layers = _hnd_layers(num_layers, layout, seed, num_heads)
    engine_kv_format, canonical_format = _FORMATS[layout]
    return (
        layers,
        engine_kv_format,
        canonical_format,
        [layer.squeeze(3) for layer in layers],
        (2, num_layers, CHUNK_TOKENS, num_heads * HEAD_SIZE),
        _shape_desc(2, num_heads, HEAD_SIZE, num_layers=num_layers),
    )


@pytest.mark.parametrize(
    "direction,skip",
    [
        (TransferDirection.D2H, 0),
        (TransferDirection.H2D, 0),
        (TransferDirection.H2D, 1),
        (TransferDirection.H2D, 3),
        (TransferDirection.H2D, 7),
    ],
)
@pytest.mark.parametrize("blocks_per_chunk", [1, 3])
def test_mla_matches_the_shared_path_on_device(
    direction: TransferDirection, skip: int, blocks_per_chunk: int
) -> None:
    """MLA writes the shared path's bytes and leaves the rest alone.

    Block IDs are permuted, chunks start with random bytes, and the last
    multi-block chunk is partial. A restore skips a global block prefix,
    including one that ends inside a chunk.
    """
    num_layers, ids = 3, [5, 2, 7, 1, 6, 0, 3]
    desc = _shape_desc(1, 1, MLA_HEAD_SIZE, num_layers=num_layers)
    chunk_tokens = blocks_per_chunk * BLOCK_SIZE
    generator = torch.Generator().manual_seed(41)
    chunks = [
        torch.randn(
            (num_layers, chunk_tokens, MLA_HEAD_SIZE), dtype=DTYPE, generator=generator
        )
        for _ in range(-(-len(ids) // blocks_per_chunk))
    ]
    layers = _mla_layers(num_layers, seed=42)
    expected_layers = [layer.clone() for layer in layers]
    expected_chunks = [chunk.clone() for chunk in chunks]
    multi_layer_block_kv_transfer(
        expected_layers,
        expected_chunks,
        ids,
        torch.device("cpu"),
        direction,
        desc,
        chunk_tokens,
        EngineKVFormat.NL_X_NB_BS_HS,
        skip,
    )

    device_layers = [_bound_device_tensor(layer) for layer in layers]
    ops = RblnDeviceOps()
    ops.prepare_kv_transfer(device_layers, EngineKVFormat.NL_X_NB_BS_HS)
    torch.rbln.synchronize()
    ops.multi_layer_block_kv_transfer(
        device_layers,
        chunks,
        ids,
        device_layers[0].device,
        direction,
        desc,
        chunk_tokens,
        EngineKVFormat.NL_X_NB_BS_HS,
        skip,
    )
    torch.rbln.synchronize()

    for got, want in zip(
        [layer.cpu() for layer in device_layers] + chunks,
        expected_layers + expected_chunks,
        strict=True,
    ):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_warm_up_prepares_every_layout_it_is_given() -> None:
    """One warm-up call prepares HND and MLA caches alike.

    The entry point groups the caches by shape and dtype, so each layout is
    prepared on its own; both then store without building a program.
    """
    hnd = _case("blocks_first", 2, 51)
    mla = _case("mla", 2, 52)
    device_layers = {
        name: [_bound_device_tensor(t) for t in case[0]]
        for name, case in (("hnd", hnd), ("mla", mla))
    }
    warm_up_kv_transfer(
        {
            f"{name}.{i}": layer
            for name, layers in device_layers.items()
            for i, layer in enumerate(layers)
        }
    )
    torch.rbln.synchronize()

    for name, case in (("hnd", hnd), ("mla", mla)):
        _, engine_kv_format, canonical_format, canonical, chunk_shape, desc = case
        chunks = [
            torch.zeros(chunk_shape, dtype=DTYPE)
            for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
        ]
        with torch.rbln.capture_programs() as built:
            RblnDeviceOps().multi_layer_block_kv_transfer(
                device_layers[name],
                chunks,
                list(range(NUM_BLOCKS)),
                device_layers[name][0].device,
                TransferDirection.D2H,
                desc,
                CHUNK_TOKENS,
                engine_kv_format,
                0,
            )
            torch.rbln.synchronize()
        assert len(built) == 0
        expected = _canonical_chunks(canonical, chunk_shape, canonical_format, desc)
        for got, want in zip(chunks, expected, strict=True):
            assert torch.equal(got, want)


@pytest.mark.parametrize("layout", ["blocks_first", "mla"])
def test_unprepared_transfer_fails_before_any_copy_on_device(layout: str) -> None:
    """A transfer of a geometry warm-up never saw fails and changes no byte.

    The engine-driven store then drops the store, and the restore reports a
    miss, so the engine recomputes instead of reading partly written KV.
    """
    # Four layers: a geometry no other test prepares.
    host_layers, engine_kv_format, _, _, chunk_shape, desc = _case(layout, 4, 61)
    layers = [_bound_device_tensor(layer) for layer in host_layers]
    chunks = [
        torch.full(chunk_shape, 7, dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]
    torch.rbln.synchronize()

    for direction in (TransferDirection.D2H, TransferDirection.H2D):
        with pytest.raises(RuntimeError, match="warm_up_kv_transfer"):
            RblnDeviceOps().multi_layer_block_kv_transfer(
                layers,
                chunks,
                list(range(NUM_BLOCKS)),
                layers[0].device,
                direction,
                desc,
                CHUNK_TOKENS,
                engine_kv_format,
                0,
            )

    assert all(torch.all(chunk == 7) for chunk in chunks)
    for got, want in zip(layers, host_layers, strict=True):
        assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize(
    "layout,num_heads",
    [("blocks_first", NUM_HEADS), ("blocks_first", BLOCK_SIZE), ("mla", NUM_HEADS)],
    ids=["hnd", "hnd-as-many-heads-as-block-tokens", "mla"],
)
def test_concurrent_stores_and_a_restore_move_independent_bytes(
    layout: str, num_heads: int
) -> None:
    """Stores on commit workers and a restore running together keep their bytes.

    Stores share one staging set and take turns; the restore uses its own.
    With as many heads as tokens per block, both HND swaps have one shape and
    run one program, which no two threads may run at once. Three stores and a
    restore start together and repeat, so their swaps overlap; each must move
    exactly its own blocks.
    """
    cases = [_case(layout, NUM_LAYERS, 71 + i, num_heads) for i in range(4)]
    _, engine_kv_format, canonical_format, _, chunk_shape, desc = cases[0]
    expected = [
        _canonical_chunks(case[3], chunk_shape, canonical_format, desc)
        for case in cases
    ]
    stored_layers = [
        [_bound_device_tensor(layer) for layer in case[0]] for case in cases[:3]
    ]
    restored = [_bound_device_tensor(torch.zeros_like(t)) for t in cases[3][0]]
    RblnDeviceOps().prepare_kv_transfer(restored, engine_kv_format)
    torch.rbln.synchronize()
    start = threading.Barrier(4)

    def store(layers: list[torch.Tensor]) -> list[torch.Tensor]:
        chunks = [
            torch.zeros(chunk_shape, dtype=DTYPE)
            for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
        ]
        start.wait(timeout=30)
        for _ in range(100):
            _transfer(layers, chunks, TransferDirection.D2H, engine_kv_format, desc)
        torch.rbln.synchronize()
        return chunks

    def restore() -> None:
        start.wait(timeout=30)
        for _ in range(300):
            _transfer(
                restored, expected[3], TransferDirection.H2D, engine_kv_format, desc
            )
        torch.rbln.synchronize()

    with ThreadPoolExecutor(max_workers=4) as pool:
        stores = [pool.submit(store, layers) for layers in stored_layers]
        restoring = pool.submit(restore)
        stored = [future.result(timeout=120) for future in stores]
        restoring.result(timeout=120)

    for got_chunks, want_chunks in zip(stored, expected[:3], strict=True):
        for got, want in zip(got_chunks, want_chunks, strict=True):
            assert torch.equal(got, want)
    for layer, want in zip(restored, cases[3][0], strict=True):
        assert torch.equal(layer.cpu(), want)
