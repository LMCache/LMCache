# SPDX-License-Identifier: Apache-2.0
"""Wire and device contracts for kernels called by LMCache_inprocess connectors."""

# Standard
from types import ModuleType
from typing import NamedTuple
import re

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform import synchronize_device, torch_ops

BLOCKS, BLOCK_SIZE, LAYERS = 3, 4, 2
# Unordered slots in three different blocks. Every other slot keeps its sentinel.
SLOTS = (9, 1, 7)
BIT_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}


class PageLayout(NamedTuple):
    """How one engine KV format stores a logical page in memory.

    A logical page is indexed ``[plane, block, slot, head, channel]``. ``axes``
    is the order of those dimensions in the engine tensor, and ``page_shape``
    folds the size-one dimensions that an MLA format does not store.
    """

    fmt: EngineKVFormat
    axes: tuple[int, ...] = (0, 1, 2, 3, 4)
    planes: int = 2
    heads: int = 2
    width: int = 8
    dtype: torch.dtype = torch.float32
    page_shape: tuple[int, ...] | None = None


LAYOUTS = {
    "flash-attn-nhd": PageLayout(EngineKVFormat.NL_X_TWO_NB_BS_NH_HS),
    "flashinfer-nhd": PageLayout(
        EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, axes=(1, 0, 2, 3, 4)
    ),
    "flash-attn-hnd": PageLayout(
        EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, axes=(0, 1, 3, 2, 4)
    ),
    "flashinfer-hnd": PageLayout(
        EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, axes=(1, 0, 3, 2, 4)
    ),
    "cross-layer": PageLayout(EngineKVFormat.NB_NL_TWO_BS_NH_HS),
    "mla": PageLayout(
        EngineKVFormat.NL_X_NB_BS_HS,
        planes=1,
        heads=1,
        page_shape=(BLOCKS, BLOCK_SIZE, 8),
    ),
    # A 132-byte BF16 row is copied in 4-byte units instead of 8-byte ones.
    "mla-66": PageLayout(
        EngineKVFormat.NL_X_NB_BS_HS,
        planes=1,
        heads=1,
        width=66,
        dtype=torch.bfloat16,
        page_shape=(BLOCKS, BLOCK_SIZE, 66),
    ),
    "sglang-mla": PageLayout(
        EngineKVFormat.NL_X_NBBS_ONE_HS,
        planes=1,
        heads=1,
        page_shape=(BLOCKS * BLOCK_SIZE, 1, 8),
    ),
}
NHD = ("flash-attn-nhd", "flashinfer-nhd")
HND = ("flash-attn-hnd", "flashinfer-hnd")
# Layouts each backend implements. Only the CUDA kernels address HND pages, and
# the 4-byte MLA unit is checked on CUDA and the torch fallback.
MULTI_LAYER_LAYOUTS = {
    "cpu": (*NHD, "cross-layer", "mla", "mla-66", "sglang-mla"),
    "cuda": tuple(LAYOUTS),
    "xpu": (*NHD, "cross-layer", "mla", "sglang-mla"),
}
SINGLE_LAYER_LAYOUTS = {
    "cpu": (*NHD, "mla"),
    "cuda": (*NHD, *HND, "mla"),
    "xpu": (*NHD, "mla"),
}
BACKEND_MARKS = {"cpu": (), "cuda": pytest.mark.cuda, "xpu": pytest.mark.xpu}
MULTI_LAYER_CASES = [
    pytest.param(
        backend, LAYOUTS[name], id=f"{backend}-{name}", marks=BACKEND_MARKS[backend]
    )
    for backend, names in MULTI_LAYER_LAYOUTS.items()
    for name in names
]
SINGLE_LAYER_CASES = [
    pytest.param(
        backend,
        LAYOUTS[name],
        token_major,
        id=f"{backend}-{name}-{order}",
        marks=BACKEND_MARKS[backend],
    )
    for backend, names in SINGLE_LAYER_LAYOUTS.items()
    for name in names
    for token_major, order in ((True, "token-major"), (False, "plane-major"))
]


def distinct_payload(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Fill ``shape`` with distinct bit patterns so every swapped element shows."""
    count = torch.Size(shape).numel()
    bits = torch.arange(count, dtype=BIT_VIEW[dtype], device=device)
    return bits.view(dtype).reshape(shape)


def sentinel(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Fill ``shape`` with the all-ones pattern, which no payload element uses."""
    return torch.full(shape, -1, dtype=BIT_VIEW[dtype], device=device).view(dtype)


def page_pointers(pages: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Device vector of page addresses, as the engine connectors build it.

    Level Zero USM addresses can set the top bit, so an XPU vector is ``uint64``
    and is filled through NumPy, as the downstream XPU connector does. CUDA and
    the fallback keep ``int64``.
    """
    addresses = np.array([page.data_ptr() for page in pages], dtype=np.uint64)
    pointers = torch.from_numpy(addresses)
    if device.type != "xpu":
        pointers = pointers.to(torch.int64)
    return pointers.to(device)


def same_bits(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Compare raw bit patterns, so NaN payloads cannot hide a difference."""
    return torch.equal(
        actual.view(BIT_VIEW[actual.dtype]), expected.view(BIT_VIEW[expected.dtype])
    )


def reference_pages(layout: PageLayout, chunk: torch.Tensor) -> list[torch.Tensor]:
    """Build the pages a scatter of ``chunk`` must produce, one per layer.

    The tokens are placed by indexing logical dimensions and only then arranged
    in the physical order, independently of the kernels' address arithmetic.

    Args:
        layout: Engine format and physical dimension order of each page.
        chunk: Transfer buffer shaped ``[plane, layer, token, head * channel]``.

    Returns:
        One page per layer. Unmapped slots hold the sentinel.
    """
    layers = chunk.shape[1]
    logical = sentinel(
        (layout.planes, layers, BLOCKS, BLOCK_SIZE, layout.heads, layout.width),
        layout.dtype,
        chunk.device,
    )
    tokens = chunk.reshape(
        layout.planes, layers, len(SLOTS), layout.heads, layout.width
    )
    for token, slot in enumerate(SLOTS):
        logical[:, :, slot // BLOCK_SIZE, slot % BLOCK_SIZE] = tokens[:, :, token]
    pages = [page.permute(layout.axes) for page in logical.unbind(1)]
    if layout.page_shape is not None:
        pages = [page.reshape(layout.page_shape) for page in pages]
    return pages


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("xpu", marks=pytest.mark.xpu),
    ]
)
def backend(request: pytest.FixtureRequest) -> tuple[ModuleType, torch.device]:
    """Run the torch fallback on CPU and the compiled module on its device."""
    name = request.param
    if name == "cpu":
        return torch_ops, torch.device("cpu")
    if torch_device_type != name or not torch_dev.is_available():
        pytest.skip(f"requires {name} hardware")
    return pytest.importorskip(f"lmcache.{name}_ops"), torch.device(name)


@pytest.mark.parametrize(("backend", "layout"), MULTI_LAYER_CASES, indirect=["backend"])
def test_multi_layer_transfer_addresses_each_layout(
    backend: tuple[ModuleType, torch.device], layout: PageLayout
) -> None:
    """A scatter writes only the mapped slots of each layout and a gather reads them."""
    ops, device = backend
    chunk = distinct_payload(
        (layout.planes, LAYERS, len(SLOTS), layout.heads * layout.width),
        layout.dtype,
        device,
    )
    expected = reference_pages(layout, chunk)
    pages = [sentinel(tuple(page.shape), layout.dtype, device) for page in expected]
    args = (
        page_pointers(pages, device),
        torch.tensor(SLOTS, device=device),
        device,
        BLOCKS * BLOCK_SIZE,
    )
    ops.multi_layer_kv_transfer(
        chunk, *args, TransferDirection.H2D, layout.fmt, BLOCK_SIZE, layout.width
    )
    synchronize_device(device)
    for page, reference in zip(pages, expected, strict=True):
        assert same_bits(page, reference)
    gathered = sentinel(tuple(chunk.shape), layout.dtype, device)
    ops.multi_layer_kv_transfer(
        gathered, *args, TransferDirection.D2H, layout.fmt, BLOCK_SIZE, layout.width
    )
    synchronize_device(device)
    assert same_bits(gathered, chunk)


@pytest.mark.parametrize(
    ("backend", "layout", "token_major"), SINGLE_LAYER_CASES, indirect=["backend"]
)
def test_single_layer_transfer_addresses_each_layout(
    backend: tuple[ModuleType, torch.device], layout: PageLayout, token_major: bool
) -> None:
    """Layerwise connectors move one page through either transfer-buffer order."""
    ops, device = backend
    chunk = distinct_payload(
        (layout.planes, 1, len(SLOTS), layout.heads * layout.width),
        layout.dtype,
        device,
    )
    (expected,) = reference_pages(layout, chunk)
    # MLA has no plane axis; otherwise the buffer is token-major or plane-major.
    buffer = chunk[:, 0]
    if layout.planes == 1:
        buffer = buffer[0]
    elif token_major:
        buffer = buffer.transpose(0, 1).contiguous()
    page = sentinel(tuple(expected.shape), layout.dtype, device)
    slots = torch.tensor(SLOTS, device=device)
    ops.single_layer_kv_transfer(
        buffer, page, slots, TransferDirection.H2D, layout.fmt, token_major
    )
    synchronize_device(device)
    assert same_bits(page, expected)
    gathered = sentinel(tuple(buffer.shape), layout.dtype, device)
    ops.single_layer_kv_transfer(
        gathered, page, slots, TransferDirection.D2H, layout.fmt, token_major
    )
    synchronize_device(device)
    assert same_bits(gathered, buffer)


def test_cachegen_encoded_bytes_and_both_decoders(
    backend: tuple[ModuleType, torch.device],
) -> None:
    """The padded and packed decoders read the same stored two-channel bitstream."""
    ops, device = backend
    symbols = (
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int8, device=device)
        .repeat(16, 1)
        .unsqueeze(0)
    )
    cdf = ops.calculate_cdf(symbols, 2)
    assert cdf.tolist() == [[[0, 32767, -1], [0, 32767, -1]]]
    encoded = torch.zeros((1, 2, 256), dtype=torch.uint8, device=device)
    lengths = torch.zeros((1, 2), dtype=torch.int32, device=device)
    ops.encode_fast_new(cdf, symbols, encoded, lengths)
    assert lengths.tolist() == [[5, 5]]
    assert encoded[0, :, :5].tolist() == [[85, 84, 0, 1, 0], [170, 169, 85, 84, 0]]
    decoded = torch.empty_like(symbols, dtype=torch.uint8)
    ops.decode_fast_new(cdf, encoded, lengths, decoded)
    assert torch.equal(decoded, symbols)
    decoded.fill_(255)
    packed = encoded[0, :, :5].contiguous().flatten()
    ops.decode_fast_prefsum(cdf, packed, lengths.cumsum(1), decoded)
    assert torch.equal(decoded, symbols)


@pytest.mark.parametrize(
    ("neox", "rotated"),
    [(True, [-3, -4, 1, 2]), (False, [-2, 1, -4, 3])],
)
def test_rotary_embedding_k_fused(
    backend: tuple[ModuleType, torch.device], neox: bool, rotated: list[int]
) -> None:
    """A quarter turn verifies the head-pair ordering in both RoPE conventions."""
    ops, device = backend
    key = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device=device)
    cache = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], device=device)
    ops.rotary_embedding_k_fused(
        torch.tensor([0], device=device),
        torch.tensor([1], device=device),
        key,
        4,
        cache,
        neox,
    )
    assert key.flatten().tolist() == rotated


@pytest.mark.cuda
def test_pci_bus_id_identifies_visible_gpu() -> None:
    """NUMA discovery consumes the PCI address of a CUDA-visible device."""
    if torch_device_type != "cuda" or not torch_dev.is_available():
        pytest.skip("requires CUDA hardware")
    ops = pytest.importorskip("lmcache.cuda_ops")
    assert re.fullmatch(
        r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",
        ops.get_gpu_pci_bus_id(0),
    )
