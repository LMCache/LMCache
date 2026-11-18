# SPDX-License-Identifier: Apache-2.0
"""Native MUSA CacheGen bounds, bitstream compatibility and device regressions."""

# Third Party
import pytest
import torch

pytest.importorskip("torch_musa")
if not torch.musa.is_available():
    pytest.skip("MUSA device required", allow_module_level=True)
native = pytest.importorskip("lmcache.musa_ops")

# First Party
from lmcache.v1.platform.torch_ops import cachegen_kernels as reference  # noqa: E402

pytestmark = [pytest.mark.musa, pytest.mark.no_shared_allocator]


def check_roundtrip(tokens: int, channels: int, skew: bool, device: str) -> None:
    """Compare both native decoder layouts and encoded bytes to the CPU codec.

    Args:
        tokens: Number of tokens per layer.
        channels: Number of channels.
        skew: When true, use all-zero symbols and the worst-case CDF.
        device: MUSA device string for the native kernels.
    """
    torch.manual_seed(19)
    symbols = torch.zeros((2, tokens, channels), dtype=torch.int8)
    if skew:
        cdf = torch.tensor([0, 1, -1], dtype=torch.int16).repeat(2, channels, 1)
    else:
        symbols.random_(0, 32)
        cdf = reference.calculate_cdf(symbols, 32)
    expected = torch.zeros((2, channels, 513), dtype=torch.uint8)
    lengths = torch.zeros((2, channels), dtype=torch.int32)
    reference.encode_fast_new(cdf, symbols, expected, lengths)
    gpu_cdf, gpu_symbols = cdf.to(device), symbols.to(device)
    buffer = torch.zeros_like(expected, device=device)
    gpu_lengths = torch.empty_like(lengths, device=device)
    native.encode_fast_new(gpu_cdf, gpu_symbols, buffer, gpu_lengths)
    assert torch.equal(gpu_lengths.cpu(), lengths)
    assert torch.equal(buffer.cpu(), expected)
    if skew and tokens == 256:
        assert lengths.unique().tolist() == [513]
    output = torch.empty_like(gpu_symbols, dtype=torch.uint8)
    native.decode_fast_new(gpu_cdf, buffer, gpu_lengths, output)
    assert torch.equal(output.cpu(), symbols.to(torch.uint8))
    packed = torch.cat(
        [
            row[: int(n)]
            for row, n in zip(expected.flatten(0, 1), lengths.flatten(), strict=True)
        ]
    )
    prefix = lengths.flatten().cumsum(0).reshape(2, channels).to(device)
    native.decode_fast_prefsum(gpu_cdf, packed.to(device), prefix, output)
    assert torch.equal(output.cpu(), symbols.to(torch.uint8))


@pytest.mark.parametrize("tokens", [1, 127, 128, 255, 256])
@pytest.mark.parametrize("channels", [1, 3, 128])
@pytest.mark.parametrize("skew", [False, True])
def test_roundtrip(tokens: int, channels: int, skew: bool) -> None:
    """Exercise short padded tails and streams crossing the old 256-byte limit."""
    check_roundtrip(tokens, channels, skew, "musa:0")


@pytest.mark.parametrize("capacity", [1, 256, 512])
def test_capacity_error_preserves_canary(capacity: int) -> None:
    """Undersized output raises without touching adjacent storage or the context."""
    cdf = torch.tensor([[[0, 1, -1]]], dtype=torch.int16, device="musa")
    symbols = torch.zeros((1, 256, 1), dtype=torch.int8, device="musa")
    allocation = torch.full(
        (1, 1, capacity + 16), 165, dtype=torch.uint8, device="musa"
    )
    lengths = torch.empty((1, 1), dtype=torch.int32, device="musa")
    with pytest.raises(RuntimeError, match="insufficient output capacity"):
        native.encode_fast_new(cdf, symbols, allocation[..., :capacity], lengths)
    assert torch.all(allocation[..., capacity:].cpu() == 165)
    assert lengths.item() == -1
    check_roundtrip(256, 1, True, "musa:0")


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("bad_length", [-1, 9, 2**32])
def test_invalid_lengths(prefix: bool, bad_length: int) -> None:
    """Invalid lengths fail before a global read, including int64 prefix overflow."""
    if not prefix and bad_length == 2**32:
        pytest.skip("int32 layout")
    cdf = torch.tensor([[[0, 1, -1]]], dtype=torch.int16, device="musa")
    data = torch.zeros((8,) if prefix else (1, 1, 8), dtype=torch.uint8, device="musa")
    lengths = torch.tensor(
        [[bad_length]], dtype=torch.int64 if prefix else torch.int32, device="musa"
    )
    output = torch.full((1, 256, 1), 165, dtype=torch.uint8, device="musa")
    decoder = native.decode_fast_prefsum if prefix else native.decode_fast_new
    with pytest.raises(RuntimeError, match="invalid stream lengths"):
        decoder(cdf, data, lengths, output)
    assert torch.all(output.cpu() == 165)


def test_non_default_stream_and_non_current_device() -> None:
    """All four host entry points honor the tensor device's current stream."""
    if torch.musa.device_count() < 2:
        pytest.skip("two MUSA devices required")
    previous = torch.musa.current_device()
    stream = torch.musa.Stream(device=1)
    try:
        with torch.musa.stream(stream):
            torch.musa.set_device(0)
            symbols = torch.randint(
                0, 32, (2, 256, 3), dtype=torch.int8, device="musa:1"
            )
            cdf = native.calculate_cdf(symbols, 32)
            expected = reference.calculate_cdf(symbols.cpu(), 32)
            assert torch.equal(cdf.cpu(), expected)
            check_roundtrip(256, 3, True, "musa:1")
            assert torch.musa.current_device() == 0
        stream.synchronize()
    finally:
        torch.musa.set_device(previous)


@pytest.mark.parametrize("entry", ["encode", "decode", "prefix"])
def test_mixed_device_rejected(entry: str) -> None:
    """Mixed CPU/MUSA arguments must be rejected before a launch."""
    cdf = torch.tensor([[[0, 1, -1]]], dtype=torch.int16, device="musa")
    output = torch.zeros((1, 1, 1), dtype=torch.uint8, device="musa")
    lengths = torch.ones((1, 1), dtype=torch.int32, device="musa")
    with pytest.raises(RuntimeError, match="same device"):
        if entry == "encode":
            native.encode_fast_new(
                cdf, torch.zeros((1, 1, 1), dtype=torch.int8), output, lengths
            )
        elif entry == "decode":
            native.decode_fast_new(cdf, output.cpu(), lengths, output)
        else:
            native.decode_fast_prefsum(
                cdf, output.flatten().cpu(), lengths.long(), output
            )
