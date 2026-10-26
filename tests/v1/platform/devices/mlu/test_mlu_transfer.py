# SPDX-License-Identifier: Apache-2.0
"""MLU block-transfer contracts, runnable on CPU and on real MLU hardware."""

# Standard
import math

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.platform.devices.mlu import MluDeviceSpec
from lmcache.v1.platform.devices.mlu.device_ops import MluDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

pytestmark = pytest.mark.no_shared_allocator

FORMATS = [
    EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
    EngineKVFormat.NL_X_NB_BS_HS,
]


@pytest.fixture(
    params=[("cpu", False), ("mlu", False), ("mlu", True)],
    ids=["cpu", "mlu-pageable", "mlu-pinned"],
)
def transfer_case(request: pytest.FixtureRequest) -> tuple[torch.device, bool]:
    """Exercise pageable and pinned host buffers on real MLU hardware."""
    device, pin_memory = request.param
    if device == "mlu" and not MluDeviceSpec().is_available():
        pytest.skip("requires torch_mlu and an MLU device")
    return torch.device(device), pin_memory


@pytest.mark.parametrize("fmt", FORMATS, ids=["nhd", "hnd", "mla"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("skip", [0, 1, 3, 4])
def test_store_reload_preserves_layout_and_skipped_blocks(
    transfer_case: tuple[torch.device, bool],
    fmt: EngineKVFormat,
    dtype: torch.dtype,
    skip: int,
) -> None:
    """Reordered blocks match an independent canonical reference in both directions."""
    transfer_device, pin_memory = transfer_case
    nl, nb, bs, nh, hs, chunk_tokens = 2, 6, 4, 3, 8, 8
    is_mla = fmt == EngineKVFormat.NL_X_NB_BS_HS
    is_hnd = fmt == EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    shape = (nb, bs, hs) if is_mla else (2, nb, bs, nh, hs)
    # Small integral values are represented exactly even in bfloat16.
    layers = [
        (torch.arange(math.prod(shape)).reshape(shape) % 97 + i).to(dtype)
        for i in range(nl)
    ]
    for layer in layers:
        layer.flatten()[:6] = torch.tensor(
            [
                -0.0,
                0.0,
                torch.finfo(dtype).tiny / 4,
                -torch.finfo(dtype).tiny / 4,
                0.375,
                -0.625,
            ],
            dtype=dtype,
        )
    bit_dtype = torch.int16 if dtype.itemsize == 2 else torch.int32
    canonical = [layer.clone() for layer in layers]
    if is_hnd:
        layers = [layer.permute(0, 1, 3, 2, 4).contiguous() for layer in layers]
    source = [layer.to(transfer_device) for layer in layers]
    target = [torch.full_like(layer, -2) for layer in source]
    chunk_shape = (nl, chunk_tokens, hs) if is_mla else (2, nl, chunk_tokens, nh * hs)
    chunks = [
        torch.full(chunk_shape, -1, dtype=dtype, pin_memory=pin_memory)
        for _ in range(2)
    ]
    expected = [chunk.clone() for chunk in chunks]
    ids = [4, 1, 5, 0]
    for pos, block in enumerate(ids):
        if pos < skip:
            continue
        obj, offset = divmod(pos, 2)
        for layer_idx, layer in enumerate(canonical):
            if is_mla:
                expected[obj][layer_idx, offset * bs : (offset + 1) * bs] = layer[block]
            else:
                expected[obj][:, layer_idx, offset * bs : (offset + 1) * bs] = layer[
                    :, block
                ].reshape(2, bs, nh * hs)
    desc = PageBufferShapeDesc()
    desc.nl, desc.nb, desc.bs = nl, nb, bs
    desc.nh, desc.hs, desc.kv_size = (1 if is_mla else nh), hs, (1 if is_mla else 2)
    desc.element_size, desc.dtype = dtype.itemsize, dtype
    ops = MluDeviceOps()
    for direction, pages in (
        (TransferDirection.D2H, source),
        (TransferDirection.H2D, target),
    ):
        ops.multi_layer_block_kv_transfer(
            pages,
            chunks,
            ids,
            transfer_device,
            direction,
            desc,
            chunk_tokens,
            fmt,
            skip,
        )
        if transfer_device.type == "mlu":
            torch.mlu.synchronize()
        if direction == TransferDirection.D2H:
            for actual, reference in zip(chunks, expected, strict=True):
                assert torch.equal(actual.view(bit_dtype), reference.view(bit_dtype))

    for actual, original in zip(target, layers, strict=True):
        reference = torch.full_like(original, -2)
        if is_mla:
            reference[ids[skip:]] = original[ids[skip:]]
        else:
            reference[:, ids[skip:]] = original[:, ids[skip:]]
        assert torch.equal(actual.cpu().view(bit_dtype), reference.view(bit_dtype))


@pytest.mark.parametrize("fmt", FORMATS, ids=["nhd", "hnd", "mla"])
def test_d2h_temporary_memory_is_bounded(fmt: EngineKVFormat) -> None:
    """Multi-chunk D2H preserves bits without retaining full per-layer gathers."""
    if not MluDeviceSpec().is_available():
        pytest.skip("requires torch_mlu and an MLU device")
    nl, nb, bs, nh, hs, chunk_tokens = 16, 64, 16, 8, 64, 512
    is_mla = fmt == EngineKVFormat.NL_X_NB_BS_HS
    is_hnd = fmt == EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    shape = (nb, bs, hs) if is_mla else (2, nb, bs, nh, hs)
    host_layers = [
        ((torch.arange(math.prod(shape)).reshape(shape) + i) % 2048 - 1024)
        .to(torch.float16)
        .div_(1024)
        for i in range(nl)
    ]
    source = [
        (layer.permute(0, 1, 3, 2, 4).contiguous() if is_hnd else layer).to("mlu")
        for layer in host_layers
    ]
    chunk_shape = (nl, chunk_tokens, hs) if is_mla else (2, nl, chunk_tokens, nh * hs)
    chunks = [
        torch.empty(chunk_shape, dtype=torch.float16, pin_memory=True) for _ in range(2)
    ]
    ids = list(reversed(range(nb)))
    desc = PageBufferShapeDesc()
    desc.nl, desc.nb, desc.bs = nl, nb, bs
    desc.nh, desc.hs, desc.kv_size = (1 if is_mla else nh), hs, (1 if is_mla else 2)
    desc.element_size, desc.dtype = 2, torch.float16
    ops = MluDeviceOps()

    def store() -> None:
        ops.multi_layer_block_kv_transfer(
            source,
            chunks,
            ids,
            "mlu",
            TransferDirection.D2H,
            desc,
            chunk_tokens,
            fmt,
            0,
        )
        torch.mlu.synchronize()

    store()
    allocated = torch.mlu.memory_allocated()
    torch.mlu.reset_peak_memory_stats()
    store()
    temporary_peak = torch.mlu.max_memory_allocated() - allocated
    # Allow scratch/allocator overhead, but not another full chunk of live
    # per-layer results or a previous chunk retained into the next allocation.
    assert temporary_peak < 2 * chunks[0].nbytes
    blocks_per_chunk = chunk_tokens // bs
    for index, chunk in enumerate(chunks):
        selected = ids[index * blocks_per_chunk : (index + 1) * blocks_per_chunk]
        expected = torch.stack(
            [
                layer[selected].reshape(chunk_tokens, hs)
                if is_mla
                else layer[:, selected].reshape(2, chunk_tokens, nh * hs)
                for layer in host_layers
            ],
            dim=0 if is_mla else 1,
        )
        assert torch.equal(chunk.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("fmt", FORMATS, ids=["nhd", "hnd", "mla"])
def test_engine_driven_dispatch_on_mlu(fmt: EngineKVFormat) -> None:
    """The public MP gather/scatter path resolves MLU ops from tensor devices."""
    if not MluDeviceSpec().is_available():
        pytest.skip("requires torch_mlu and an MLU device")
    is_mla = fmt == EngineKVFormat.NL_X_NB_BS_HS
    is_hnd = fmt == EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    shape = (6, 4, 8) if is_mla else ((2, 6, 3, 4, 8) if is_hnd else (2, 6, 4, 3, 8))
    source = {
        str(i): torch.randn(shape, dtype=torch.float16).to("mlu") for i in range(2)
    }
    target = {key: torch.zeros_like(value) for key, value in source.items()}
    hints: LayoutHints = {"kv_layout": "HND" if is_hnd else "NHD"}
    ids = [3, 0, 5, 1]
    stream = torch.mlu.Stream()
    with torch.mlu.stream(stream):
        stream.wait_stream(torch.mlu.default_stream())
        chunks = gather_paged_kv_to_cpu(source, ids, 2, layout_hints=hints)
        stream.synchronize()
        scatter_cpu_to_paged_kv(target, ids, chunks, 2, layout_hints=hints)
        stream.synchronize()
    for key, actual in target.items():
        expected = torch.zeros_like(source[key])
        if is_mla:
            expected[ids] = source[key][ids]
        else:
            expected[:, ids] = source[key][:, ids]
        assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("record_event", [False, True])
def test_stream_completion_after_device_to_host_copy(
    default: bool, record_event: bool
) -> None:
    """Default and owned queues publish only after copying and remain usable."""
    if not MluDeviceSpec().is_available():
        pytest.skip("requires torch_mlu and an MLU device")
    spec = MluDeviceSpec()
    ops = spec.get_ops()
    ops.drain_recorded_completions()
    ops.drain_recorded_events()
    device = torch.device("mlu")
    stream = torch.mlu.default_stream() if default else torch.mlu.Stream()
    host = torch.empty(1024 * 1024, dtype=torch.float32, pin_memory=True)
    with torch.mlu.stream(stream):
        source = torch.full(host.shape, 17, dtype=host.dtype, device=device)
        host.copy_(source, non_blocking=True)
        event = spec.create_stream_event(device)
        spec.record_stream_event(event, stream)
    handle = 0 if default else spec.get_stream_handle(stream)
    if record_event:
        ops.record_event_on_stream(handle, "store", "session", {}, {})
        assert len(ops.drain_recorded_events()) == 1
    else:
        ops.record_completion_on_stream(handle, "store", b"done")
        assert ops.drain_recorded_completions() == [("store", b"done")]
    assert spec.is_stream_event_complete(event)
    assert torch.equal(host, torch.full_like(host, 17))
    # Wrapping a native queue for completion must not destroy the owner's queue.
    with torch.mlu.stream(stream):
        source.fill_(23)
        host.copy_(source, non_blocking=True)
    stream.synchronize()
    assert torch.equal(host, torch.full_like(host, 23))


def test_multichunk_transfer_survives_allocator_reuse() -> None:
    """Temporary tensors stay safe under allocation pressure on another stream."""
    if not MluDeviceSpec().is_available():
        pytest.skip("requires torch_mlu and an MLU device")
    shape = (2, 32, 16, 4, 64)
    host_layers = {
        str(i): (torch.arange(math.prod(shape)).reshape(shape) % 97 + i).half()
        for i in range(4)
    }
    source = {key: layer.to("mlu") for key, layer in host_layers.items()}
    target = {key: torch.zeros_like(layer) for key, layer in source.items()}
    ids = list(reversed(range(32)))
    stream = torch.mlu.Stream()
    producer = torch.mlu.current_stream()
    with torch.mlu.stream(stream):
        stream.wait_stream(producer)
        chunks = gather_paged_kv_to_cpu(
            source, ids, 4, layout_hints={"kv_layout": "NHD"}
        )
    pressure = [
        torch.full(shape, -7, device="mlu", dtype=torch.float16) for _ in range(16)
    ]
    stream.synchronize()
    for index, chunk in enumerate(chunks):
        selected = ids[index * 4 : (index + 1) * 4]
        expected = torch.stack(
            [layer[:, selected].reshape(2, 64, 256) for layer in host_layers.values()],
            dim=1,
        )
        assert torch.equal(chunk, expected)
    with torch.mlu.stream(stream):
        scatter_cpu_to_paged_kv(
            target, ids, chunks, 4, layout_hints={"kv_layout": "NHD"}
        )
    for tensor in pressure:
        tensor.fill_(11)
    stream.synchronize()
    for key, layer in target.items():
        assert torch.equal(layer.cpu(), host_layers[key])
