# SPDX-License-Identifier: Apache-2.0
"""Kernel-level tests for the KVWeave native modules (Intel only).

These drive ``lmcache.kvweave_quant`` (CPU) and ``lmcache.kvweave_quant_xpu``
(XPU) directly, through the entry points the ``kvweave`` serde calls, so a
kernel regression is reported against the kernel rather than through the
serde. Adapted from the native-layer tests of the standalone KVWeave project.

Payload layouts produced by the native modules:

* ``kvweave_serialize_chunk``: ``header | BE32 len | K scales | BE32 len |
  V scales | q_K | q_V``.
* ``kvweave_serialize_chunk_state`` / ``kvweave_serialize_fused_chunk_direct``:
  ``header | BE32 len | scales | q``.
"""

# Standard
import struct

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type

kvweave_quant = pytest.importorskip("lmcache.kvweave_quant")

try:
    # First Party
    import lmcache.kvweave_quant_xpu as kvweave_quant_xpu
except ImportError:
    kvweave_quant_xpu = None

_HAS_XPU = (
    kvweave_quant_xpu is not None
    and torch_device_type == "xpu"
    and torch_dev.is_available()
)
_DEVICES = ["cpu"] + (["xpu"] if _HAS_XPU else [])
requires_xpu = pytest.mark.skipif(not _HAS_XPU, reason="requires an Intel XPU")

_HEADER = b"KVWT"
_LAYERS, _TOKENS, _HIDDEN = 3, 64, 256

# Mean absolute error bounds for unit-variance inputs.
_MEAN_ERR = {(4, False): 0.15, (8, False): 0.02, (4, True): 0.5, (8, True): 0.05}


def _native(device: str):
    return kvweave_quant if device == "cpu" else kvweave_quant_xpu


def _precond(n: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    perm = torch.randperm(n, generator=g).to(torch.int32)
    return signs, perm


def _data(shape, dtype=torch.float16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype)


def _split_chunk_payload(payload: bytes) -> tuple[bytes, bytes, bytes]:
    off = len(_HEADER)
    blobs = []
    for _ in range(2):
        (n,) = struct.unpack_from(">I", payload, off)
        off += 4
        blobs.append(payload[off : off + n])
        off += n
    return blobs[0], blobs[1], payload[off:]


def _split_state_payload(payload: bytes) -> tuple[bytes, bytes]:
    off = len(_HEADER)
    (n,) = struct.unpack_from(">I", payload, off)
    off += 4
    return payload[off : off + n], payload[off + n :]


def _q(raw: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(raw), dtype=torch.int8)


def _parse_unit_scales(blob: bytes, units: int, nchunks: int) -> torch.Tensor:
    """Per-unit framed scale blob -> ``[units, nchunks, 2]`` (scale, zp)."""
    out = torch.empty(units, nchunks, 2)
    off = 0
    for u in range(units):
        (count,) = struct.unpack_from("<I", blob, off)
        assert count == nchunks
        off += 4
        for _ in range(nchunks):
            idx, scale, zp = struct.unpack_from("<Iff", blob, off)
            out[u, idx, 0], out[u, idx, 1] = scale, zp
            off += 12
    assert off == len(blob)
    return out


# ---------------------------------------------------------------------------
# Split K/V chunks: kvweave_serialize_chunk / kvweave_dequantize_chunk
# ---------------------------------------------------------------------------


def _chunk_kwargs(qbit, scaling, rh, asym, signs, perm, tokens=_TOKENS):
    # blocks_num * block_size is the per_token row count; head_num=1 /
    # head_dim=H is how the serde calls it (the scaling ignores heads).
    return dict(
        qbit=qbit,
        blocks_num=1,
        block_size=tokens,
        head_num=1,
        head_dim=_HIDDEN,
        rh=rh,
        asym=asym,
        scaling_method=scaling,
        signs=signs,
        perm=perm,
        num_threads=0,
    )


def _encode_chunk(native, x, qbit, scaling, rh, asym, signs, perm):
    tokens = x.shape[2]
    return bytes(
        native.kvweave_serialize_chunk(
            x,
            _HEADER,
            1,
            2,
            num_layers=x.shape[1],
            **_chunk_kwargs(qbit, scaling, rh, asym, signs, perm, tokens),
        )
    )


def _decode_chunk(native, payload, shape, dtype, qbit, scaling, rh, asym, signs, perm):
    k_blob, v_blob, q_raw = _split_chunk_payload(payload)
    _, layers, tokens, hidden = shape
    return native.kvweave_dequantize_chunk(
        _q(q_raw),
        k_blob,
        v_blob,
        layers,
        tokens,
        hidden,
        output_dtype=dtype,
        **_chunk_kwargs(qbit, scaling, rh, asym, signs, perm, tokens),
    )


# (scaling, rh, precond) combinations the serde produces for split chunks.
_SPLIT_MODES = [
    ("per_token", False, False),
    ("per_token", True, False),
    ("per_token", True, True),
    ("per_tensor", False, False),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("qbit", [4, 8])
@pytest.mark.parametrize(
    "scaling,rh,precond", _SPLIT_MODES, ids=["pt", "pt-rh", "pt-rh-pre", "tensor"]
)
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_chunk_round_trip(device, qbit, scaling, rh, precond, asym, dtype):
    """Quantize -> dequantize stays within the accuracy bound of the scaling."""
    native = _native(device)
    x = _data((2, _LAYERS, _TOKENS, _HIDDEN), dtype)
    signs, perm = _precond(_HIDDEN) if precond else (None, None)
    payload = _encode_chunk(native, x, qbit, scaling, rh, asym, signs, perm)
    assert payload[: len(_HEADER)] == _HEADER
    out = _decode_chunk(
        native, payload, x.shape, dtype, qbit, scaling, rh, asym, signs, perm
    )
    assert out.shape == x.shape and out.dtype == dtype
    err = (out.float() - x.float()).abs().mean().item()
    assert err < _MEAN_ERR[(qbit, scaling == "per_tensor")], err


@pytest.mark.parametrize("device", _DEVICES)
def test_split_chunk_is_deterministic_and_into_4d_matches(device):
    """Same input -> same bytes; ``into_4d`` writes exactly what
    ``kvweave_dequantize_chunk`` returns."""
    native = _native(device)
    x = _data((2, _LAYERS, _TOKENS, _HIDDEN))
    signs, perm = _precond(_HIDDEN)
    args = (4, "per_token", True, True, signs, perm)
    payload = _encode_chunk(native, x, *args)
    assert payload == _encode_chunk(native, x, *args)

    expected = _decode_chunk(native, payload, x.shape, x.dtype, *args)
    k_blob, v_blob, q_raw = _split_chunk_payload(payload)
    dst = torch.full_like(x, 7.0)
    native.kvweave_dequantize_chunk_into_4d(
        _q(q_raw),
        k_blob,
        v_blob,
        dst,
        _LAYERS,
        _TOKENS,
        _HIDDEN,
        output_dtype=x.dtype,
        **_chunk_kwargs(4, "per_token", True, True, signs, perm),
    )
    assert torch.equal(dst, expected)


def test_cpu_split_chunk_out_matches_bytes():
    """CPU ``kvweave_serialize_chunk(out=...)`` writes exactly the bytes the
    bytes-returning call produces, nothing past them, and rejects an undersized
    or mistyped buffer."""
    x = _data((2, _LAYERS, _TOKENS, _HIDDEN))
    signs, perm = _precond(_HIDDEN)
    kwargs = _chunk_kwargs(4, "per_token", True, True, signs, perm)
    expected = _encode_chunk(kvweave_quant, x, 4, "per_token", True, True, signs, perm)
    out = torch.full((len(expected) + 55,), 0xAB, dtype=torch.uint8)
    n = kvweave_quant.kvweave_serialize_chunk(
        x, _HEADER, 1, 2, num_layers=_LAYERS, out=out, **kwargs
    )
    assert n == len(expected)
    assert bytes(out[:n].numpy()) == expected
    assert bool((out[n:] == 0xAB).all())
    for bad in (
        torch.empty(n - 1, dtype=torch.uint8),
        torch.empty(n, dtype=torch.int8),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            kvweave_quant.kvweave_serialize_chunk(
                x, _HEADER, 1, 2, num_layers=_LAYERS, out=bad, **kwargs
            )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("tokens", [16, 48])
def test_split_chunk_with_tokens_not_a_multiple_of_64(device, tokens):
    """The per_token row count is ``blocks_num * block_size``; any token count
    works when it equals the chunk's tokens."""
    native = _native(device)
    x = _data((2, _LAYERS, tokens, _HIDDEN))
    signs, perm = _precond(_HIDDEN)
    args = (4, "per_token", True, True, signs, perm)
    out = _decode_chunk(
        native, _encode_chunk(native, x, *args), x.shape, x.dtype, *args
    )
    assert (out.float() - x.float()).abs().mean().item() < _MEAN_ERR[(4, False)]


# ---------------------------------------------------------------------------
# Preconditioner validation (signs / perm must cover exactly one chunk)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_mismatched_preconditioner_is_rejected(device):
    """A preconditioner that does not cover exactly one quantization chunk, or
    a perm entry outside it, is rejected instead of indexing out of bounds."""
    native = _native(device)
    x = _data((2, _LAYERS, _TOKENS, _HIDDEN))
    signs, perm = _precond(_HIDDEN)
    payload = _encode_chunk(native, x, 4, "per_token", True, True, signs, perm)
    bad_perm = perm.clone()
    bad_perm[0] = _HIDDEN

    bad = [
        (signs[:-1], perm),  # short signs
        (signs, perm[:-1]),  # short perm
        (signs, bad_perm),  # perm entry out of range
    ]
    for s, p in bad:
        with pytest.raises((ValueError, RuntimeError)):
            _encode_chunk(native, x, 4, "per_token", True, True, s, p)
        with pytest.raises((ValueError, RuntimeError)):
            _decode_chunk(
                native, payload, x.shape, x.dtype, 4, "per_token", True, True, s, p
            )
        with pytest.raises((ValueError, RuntimeError)):
            native.kvweave_serialize_chunk_state(
                x[0],
                _HEADER,
                3,
                num_layers=_LAYERS,
                **_chunk_kwargs(4, "per_token", True, True, s, p),
            )

    # The original bug: a per_token row count that does not match the tokens
    # changes the chunk size, so a hidden-sized preconditioner no longer fits.
    wrong_rows = _chunk_kwargs(4, "per_token", True, True, signs, perm)
    wrong_rows["block_size"] = 4 * _TOKENS
    with pytest.raises((ValueError, RuntimeError)):
        native.kvweave_serialize_chunk(
            x, _HEADER, 1, 2, num_layers=_LAYERS, **wrong_rows
        )


# ---------------------------------------------------------------------------
# Fused K/V chunks: kvweave_serialize_fused_chunk_direct (per_channel)
# ---------------------------------------------------------------------------

_HEAD_NUM, _HEAD_DIM = 2, 128  # fused [L, T, 2 * 128]: the serde's two halves


def _split_heads(x: torch.Tensor) -> torch.Tensor:
    layers, tokens, _ = x.shape
    return (
        x.reshape(layers, tokens, _HEAD_NUM, _HEAD_DIM).permute(0, 2, 1, 3).contiguous()
    )


def _merge_heads(y: torch.Tensor) -> torch.Tensor:
    return (
        y.view(_LAYERS, _HEAD_NUM, _TOKENS, _HEAD_DIM)
        .permute(0, 2, 1, 3)
        .reshape(_LAYERS, _TOKENS, _HEAD_NUM * _HEAD_DIM)
    )


def _state_kwargs(qbit, scaling, rh, asym, signs, perm):
    return dict(
        qbit=qbit,
        blocks_num=1,
        block_size=_TOKENS,
        head_num=_HEAD_NUM,
        head_dim=_HEAD_DIM,
        rh=rh,
        asym=asym,
        scaling_method=scaling,
        signs=signs,
        perm=perm,
        num_threads=0,
    )


def _direct(native, x, qbit, rh, asym, signs, perm, scaling="per_channel", out=None):
    return native.kvweave_serialize_fused_chunk_direct(
        x,
        _HEADER,
        qbit,
        _HEAD_NUM,
        _HEAD_DIM,
        rh=rh,
        asym=asym,
        scaling_method=scaling,
        signs=signs,
        perm=perm,
        num_threads=0,
        out=out,
    )


def _direct_dequant(native, q, blob, dst, qbit, rh, asym, signs, perm, scaling):
    native.kvweave_dequantize_fused_chunk_direct(
        q,
        blob,
        dst,
        _LAYERS,
        _TOKENS,
        _HEAD_NUM,
        _HEAD_DIM,
        qbit,
        rh=rh,
        asym=asym,
        scaling_method=scaling,
        signs=signs,
        perm=perm,
        num_threads=0,
    )


_FUSED_MODES = [(False, False), (True, False), (True, True)]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("qbit", [4, 8])
@pytest.mark.parametrize("rh,precond", _FUSED_MODES, ids=["no-rh", "rh", "rh-pre"])
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_fused_direct_matches_split_heads(device, dtype, qbit, rh, precond, asym):
    """The direct kernel reads the fused layout in place and produces the same
    payload as head-splitting first; its dequant matches the split path
    exactly and stays within the per_channel accuracy bound."""
    native = _native(device)
    x = _data((_LAYERS, _TOKENS, _HEAD_NUM * _HEAD_DIM), dtype)
    signs, perm = _precond(_TOKENS) if precond else (None, None)
    units = _LAYERS * _HEAD_NUM
    kwargs = _state_kwargs(qbit, "per_channel", rh, asym, signs, perm)

    direct = bytes(_direct(native, x, qbit, rh, asym, signs, perm))
    ref = bytes(
        native.kvweave_serialize_chunk_state(
            _split_heads(x).view(-1), _HEADER, 4, num_layers=units, **kwargs
        )
    )
    assert len(direct) == len(ref)
    d_blob, d_q = _split_state_payload(direct)
    r_blob, r_q = _split_state_payload(ref)
    torch.testing.assert_close(
        _parse_unit_scales(d_blob, units, _HEAD_DIM),
        _parse_unit_scales(r_blob, units, _HEAD_DIM),
        rtol=1e-5,
        atol=1e-6,
    )
    assert (_q(d_q) != _q(r_q)).float().mean().item() < 1e-2

    expected = _merge_heads(
        native.kvweave_dequantize_chunk_state(
            _q(d_q), d_blob, units, _TOKENS, _HEAD_DIM, output_dtype=dtype, **kwargs
        )
    )
    dst = torch.full_like(x, 7.0)
    _direct_dequant(
        native, _q(d_q), d_blob, dst, qbit, rh, asym, signs, perm, "per_channel"
    )
    assert torch.equal(dst, expected)
    assert (dst.float() - x.float()).abs().mean().item() < _MEAN_ERR[(qbit, False)]


@pytest.mark.parametrize("device", _DEVICES)
def test_fused_direct_buffer_contract(device):
    """``out=`` writes exactly the returned bytes and nothing past them,
    undersized / mistyped buffers are rejected, trailing payload padding is
    ignored on decode and a truncated payload is rejected."""
    native = _native(device)
    x = _data((_LAYERS, _TOKENS, _HEAD_NUM * _HEAD_DIM))
    signs, perm = _precond(_TOKENS)
    expected = bytes(_direct(native, x, 4, True, True, signs, perm))

    out = torch.full((len(expected) + 77,), 0xAB, dtype=torch.uint8)
    n = _direct(native, x, 4, True, True, signs, perm, out=out)
    assert n == len(expected)
    assert bytes(out[:n].numpy()) == expected
    assert bool((out[n:] == 0xAB).all())
    for bad_out in (
        torch.empty(n - 1, dtype=torch.uint8),
        torch.empty(n, dtype=torch.int8),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            _direct(native, x, 4, True, True, signs, perm, out=bad_out)

    blob, q_raw = _split_state_payload(expected)
    exact, padded = torch.empty_like(x), torch.empty_like(x)
    _direct_dequant(
        native, _q(q_raw), blob, exact, 4, True, True, signs, perm, "per_channel"
    )
    q_padded = torch.cat([_q(q_raw), torch.full((4096,), -17, dtype=torch.int8)])
    _direct_dequant(
        native, q_padded, blob, padded, 4, True, True, signs, perm, "per_channel"
    )
    assert torch.equal(exact, padded)
    with pytest.raises((ValueError, RuntimeError)):
        _direct_dequant(
            native,
            _q(q_raw)[:-1],
            blob,
            exact,
            4,
            True,
            True,
            signs,
            perm,
            "per_channel",
        )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("scaling", ["per_token", "per_tensor"])
def test_fused_direct_is_per_channel_only(device, scaling):
    """Fused chunks are only quantized per channel; other scalings raise."""
    native = _native(device)
    x = _data((_LAYERS, _TOKENS, _HEAD_NUM * _HEAD_DIM))
    with pytest.raises((ValueError, RuntimeError)):
        _direct(native, x, 4, False, True, None, None, scaling=scaling)
    blob, q_raw = _split_state_payload(
        bytes(_direct(native, x, 4, False, True, None, None))
    )
    with pytest.raises((ValueError, RuntimeError)):
        _direct_dequant(
            native,
            _q(q_raw),
            blob,
            torch.empty_like(x),
            4,
            False,
            True,
            None,
            None,
            scaling,
        )


# ---------------------------------------------------------------------------
# Single-plane state API (fused per_tensor and the XPU fused decode path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("qbit", [4, 8])
def test_state_per_tensor_round_trip(device, qbit):
    """Fused chunks with ``per_tensor=true`` go through the state API."""
    native = _native(device)
    x = _data((_LAYERS, _TOKENS, _HEAD_NUM * _HEAD_DIM))
    kwargs = _state_kwargs(qbit, "per_tensor", False, True, None, None)
    payload = bytes(
        native.kvweave_serialize_chunk_state(
            x, _HEADER, 5, num_layers=_LAYERS, **kwargs
        )
    )
    blob, q_raw = _split_state_payload(payload)
    out = native.kvweave_dequantize_chunk_state(
        _q(q_raw),
        blob,
        _LAYERS,
        _TOKENS,
        _HEAD_NUM * _HEAD_DIM,
        output_dtype=x.dtype,
        **kwargs,
    )
    assert (out.float().view_as(x) - x.float()).abs().mean().item() < _MEAN_ERR[
        (qbit, True)
    ]


@requires_xpu
@pytest.mark.parametrize("merge_head_num", [0, _HEAD_NUM])
def test_xpu_state_into_variants_match_the_tensor_api(merge_head_num):
    """``*_chunk_state_into`` write exactly what the bytes / tensor returning
    entry points produce (with heads merged back when asked)."""
    x = _data((_LAYERS * _HEAD_NUM, _TOKENS, _HEAD_DIM))
    kwargs = _state_kwargs(4, "per_channel", False, True, None, None)
    planes = _LAYERS * _HEAD_NUM
    expected = bytes(
        kvweave_quant_xpu.kvweave_serialize_chunk_state(
            x, _HEADER, 6, num_layers=planes, **kwargs
        )
    )
    dst = torch.full((len(expected) + 33,), 0xAB, dtype=torch.uint8)
    n = kvweave_quant_xpu.kvweave_serialize_chunk_state_into(
        x, dst, _HEADER, 6, num_layers=planes, **kwargs
    )
    assert n == len(expected) and bytes(dst[:n].numpy()) == expected
    assert bool((dst[n:] == 0xAB).all())

    blob, q_raw = _split_state_payload(expected)
    ref = kvweave_quant_xpu.kvweave_dequantize_chunk_state(
        _q(q_raw), blob, planes, _TOKENS, _HEAD_DIM, output_dtype=x.dtype, **kwargs
    )
    if merge_head_num:
        ref = _merge_heads(ref)
    out = torch.full(ref.shape, 7.0, dtype=x.dtype)
    kvweave_quant_xpu.kvweave_dequantize_chunk_state_into(
        _q(q_raw),
        blob,
        out.view(-1),
        planes,
        _TOKENS,
        _HEAD_DIM,
        output_dtype=x.dtype,
        merge_head_num=merge_head_num,
        **kwargs,
    )
    assert torch.equal(out, ref)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def test_split_and_merge_fused_heads_are_inverse():
    """The CPU head reorder used around the fused codec round-trips exactly."""
    x = _data((_LAYERS, _TOKENS, _HEAD_NUM * _HEAD_DIM))
    split = kvweave_quant.kvweave_split_fused_heads(x, _HEAD_NUM, _HEAD_DIM)
    assert torch.equal(split.view_as(_split_heads(x)), _split_heads(x))
    merged = kvweave_quant.kvweave_merge_fused_heads(
        split, _LAYERS, _HEAD_NUM, _HEAD_DIM
    )
    assert torch.equal(merged.view_as(x), x)
