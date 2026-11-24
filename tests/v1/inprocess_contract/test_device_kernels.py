# SPDX-License-Identifier: Apache-2.0
"""Wire and device contracts for kernels called by LMCache_inprocess connectors."""

# Standard
from types import ModuleType
import re

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform import torch_ops


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


def test_single_and_multi_layer_transfer(
    backend: tuple[ModuleType, torch.device],
) -> None:
    """Both transfer APIs gather and scatter the same nonconsecutive KV slots."""
    ops, device = backend
    blocks, block_size, layers, heads, width = 3, 4, 2, 2, 8
    fmt = EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    slots = torch.tensor([9, 1, 7], device=device)
    pages = [
        torch.full((2, blocks, block_size, heads, width), -1.0, device=device)
        for _ in range(layers)
    ]
    ptrs = torch.tensor([page.data_ptr() for page in pages], device=device)
    chunk = torch.arange(
        2 * layers * len(slots) * heads * width, device=device, dtype=torch.float32
    ).reshape(2, layers, len(slots), heads * width)
    args = (ptrs, slots, device, blocks * block_size)
    ops.multi_layer_kv_transfer(
        chunk, *args, TransferDirection.H2D, fmt, block_size, width
    )
    for layer, page in enumerate(pages):
        expected = torch.full_like(page, -1)
        expected.flatten(1, 2)[:, slots] = chunk[:, layer].reshape(
            2, len(slots), heads, width
        )
        assert torch.equal(page, expected)
        # Layerwise connectors also use a token-major transfer buffer.
        single = torch.empty((len(slots), 2, heads * width), device=device)
        ops.single_layer_kv_transfer(
            single, page, slots, TransferDirection.D2H, fmt, True
        )
        assert torch.equal(single.transpose(0, 1), chunk[:, layer])
        single.add_(1000)
        ops.single_layer_kv_transfer(
            single, page, slots, TransferDirection.H2D, fmt, True
        )
    retrieved = torch.empty_like(chunk)
    ops.multi_layer_kv_transfer(
        retrieved, *args, TransferDirection.D2H, fmt, block_size, width
    )
    assert torch.equal(retrieved, chunk + 1000)


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
