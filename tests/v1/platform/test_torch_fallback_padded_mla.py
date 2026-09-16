# SPDX-License-Identifier: Apache-2.0
"""Torch-fallback block transfer for dim-0-padded MLA pools and the DSA indexer cache.

Regression for the DeepSeek V4 reload failure seen with vLLM + LMCache MP when
``lmcache.cuda_ops`` cannot load (``index_copy_(): ... Source dimensionality
(3), destination dimensionality (5)``). In pointer mode the fallback rebuilt
every ``NL_X_NB_BSV_BSS`` layer with the generic 5-D NHD shape, and ignored
``PageBufferShapeDesc.block_stride_elems`` for the per-layer MLA formats, so
stores gathered the wrong blocks and reloads either crashed or scattered into
the wrong pages. These tests run on CPU; the ``cuda`` parametrisations only
add the same checks on device tensors when a GPU happens to be present.
"""

# Standard
import ctypes
import struct
import unittest.mock
import warnings

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import torch_ops
import lmcache.lmcache_native as lmcache_native

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
TRANSFER_CASES = [
    pytest.param("cpu", False, id="cpu"),
    pytest.param("cuda", False, marks=cuda_only, id="cuda-host-chunks"),
    pytest.param("cuda", True, marks=cuda_only, id="cuda-device-chunks"),
]

H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H
MLA = lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
BSV_BSS = lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS


def make_pool(
    nb: int,
    nl: int,
    bs: int,
    hs: int,
    pad_elems: int | None,
    dtype: torch.dtype,
    device: str,
    fill: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Blocks-first pool like vLLM's hybrid allocator: one page per block holds
    ``nl`` layer slices plus ``pad_elems`` of padding owned by other groups.
    Each layer is the dim-0-padded view ``raw[:, l * bs * hs : ...]``.

    ``pad_elems=None`` builds the classic layout instead: every layer is its
    own contiguous ``[NB, BS, HS]`` tensor (tight ``stride(0) == bs * hs``);
    ``raw`` is then the stacked ``[NB, NL * BS * HS]`` mirror with no padding."""

    def _fill(shape):
        if fill != "random":
            return torch.zeros(shape, dtype=dtype, device=device)
        if dtype == torch.uint8:
            return torch.randint(0, 256, shape, dtype=dtype, device=device)
        return torch.randn(*shape, dtype=dtype, device=device)

    if pad_elems is None:
        # Twice the rows, then slice: a rebuild that over-reads the layer
        # (the old 5-D default) lands in owned memory and fails by value.
        views = [_fill((2 * nb, bs, hs))[:nb] for _ in range(nl)]
        assert all(v.stride(0) == bs * hs for v in views)
        raw = torch.cat([v.reshape(nb, bs * hs) for v in views], dim=1)
        return raw, views
    page = nl * bs * hs + pad_elems
    raw = _fill((nb, page))
    views = [
        raw[:, layer * bs * hs : (layer + 1) * bs * hs].view(nb, bs, hs)
        for layer in range(nl)
    ]
    return raw, views


def make_shape_desc(
    nl: int, nb: int, bs: int, hs: int, dtype: torch.dtype, block_stride: int
):
    sd = lmcache_native.PageBufferShapeDesc()
    sd.kv_size = 1
    sd.nl = nl
    sd.nb = nb
    sd.bs = bs
    sd.nh = 1
    sd.hs = hs
    sd.element_size = dtype.itemsize
    sd.block_stride_elems = block_stride
    sd.dtype = dtype
    return sd


def ptr_tensor(views: list[torch.Tensor], device: str) -> torch.Tensor:
    return torch.tensor(
        [v.data_ptr() for v in views], dtype=torch.uint64, device=device
    )


def alloc_chunks(
    count: int,
    nl: int,
    tokens: int,
    hs: int,
    dtype: torch.dtype,
    device: str,
    device_chunks: bool = False,
) -> list[torch.Tensor]:
    chunks = [torch.zeros(nl, tokens, hs, dtype=dtype) for _ in range(count)]
    if device_chunks:
        return [c.to(device) for c in chunks]
    if device == "cuda":
        chunks = [c.pin_memory() for c in chunks]
    return chunks


def run(
    views, chunks, block_ids, device, direction, shape_desc, chunk_tokens, fmt
) -> None:
    torch_ops.multi_layer_block_kv_transfer(
        ptr_tensor(views, device),
        [c.data_ptr() for c in chunks],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        direction,
        shape_desc,
        chunk_tokens,
        fmt,
        0,
    )
    if device == "cuda":
        torch.cuda.synchronize()


@pytest.mark.parametrize("device,device_chunks", TRANSFER_CASES)
def test_pointer_mode_honours_block_stride_roundtrip(device, device_chunks):
    """Padded per-layer MLA pool: D2H must gather the right blocks, H2D must
    write them back without touching the padding owned by other groups."""
    torch.manual_seed(7)
    nb, nl, bs, hs, pad = 8, 3, 4, 16, 40
    dtype = torch.float32
    raw, views = make_pool(nb, nl, bs, hs, pad, dtype, device, "random")
    assert not views[0].is_contiguous()
    sd = make_shape_desc(nl, nb, bs, hs, dtype, views[0].stride(0))
    assert sd.block_stride_elems != bs * hs

    blocks_per_chunk = 2
    chunk_tokens = blocks_per_chunk * bs
    block_ids = [5, 2, 7, 0, 3, 6, 1, 4]
    chunks = alloc_chunks(
        nb // blocks_per_chunk, nl, chunk_tokens, hs, dtype, device, device_chunks
    )

    run(views, chunks, block_ids, device, D2H, sd, chunk_tokens, MLA)
    for k, chunk in enumerate(chunks):
        ids = block_ids[k * blocks_per_chunk : (k + 1) * blocks_per_chunk]
        for layer in range(nl):
            expected = views[layer][ids].reshape(chunk_tokens, hs).cpu()
            assert torch.equal(chunk[layer].cpu(), expected), f"chunk {k} layer {layer}"

    raw2, views2 = make_pool(nb, nl, bs, hs, pad, dtype, device, "zeros")
    sd2 = make_shape_desc(nl, nb, bs, hs, dtype, views2[0].stride(0))
    run(views2, chunks, block_ids, device, H2D, sd2, chunk_tokens, MLA)
    for layer in range(nl):
        assert torch.equal(views2[layer], views[layer]), f"layer {layer}"
    padding = raw2[:, nl * bs * hs :]
    assert torch.count_nonzero(padding) == 0, "H2D wrote into the page padding"


def reference_rows(pages: torch.Tensor, val_bytes: int) -> torch.Tensor:
    """Pure-python repack of ``[N, BS, HS]`` uint8 engine pages into
    token-major rows ``[N*BS, HS]`` = ``[vals | scale]`` per token."""
    n, bs, hs = pages.shape
    scale_bytes = hs - val_bytes
    rows = []
    for page in pages.reshape(n, bs * hs).cpu().tolist():
        vals = page[: bs * val_bytes]
        scales = page[bs * val_bytes :]
        for t in range(bs):
            rows.append(
                vals[t * val_bytes : (t + 1) * val_bytes]
                + scales[t * scale_bytes : (t + 1) * scale_bytes]
            )
    return torch.tensor(rows, dtype=torch.uint8)


@pytest.mark.parametrize("device,device_chunks", TRANSFER_CASES)
@pytest.mark.parametrize("pad", [None, 24], ids=["contiguous", "padded"])
def test_pointer_mode_bsv_bss_repacks_blocked_scale_pages(device, device_chunks, pad):
    """DSA indexer k-cache: pointer mode must rebuild rank-3 ``[NB, BS, 132]``
    layers (not the 5-D NHD default) and convert between the blocked
    ``[BS x 128 vals][BS x 4 scales]`` page and token-major chunk rows in
    both directions, exactly like the native kernel. ``contiguous`` is the
    tight layout where the old code reached ``index_copy_`` with a 5-D
    destination; ``padded`` is the DeepSeek V4 shared-page layout."""
    torch.manual_seed(11)
    nb, nl, bs, hs = 6, 2, 4, 132
    val_bytes = hs - 4
    dtype = torch.uint8
    raw, views = make_pool(nb, nl, bs, hs, pad, dtype, device, "random")
    sd = make_shape_desc(nl, nb, bs, hs, dtype, views[0].stride(0))

    blocks_per_chunk = 3
    chunk_tokens = blocks_per_chunk * bs
    block_ids = [4, 1, 5, 0, 2, 3]
    chunks = alloc_chunks(
        nb // blocks_per_chunk, nl, chunk_tokens, hs, dtype, device, device_chunks
    )

    with warnings.catch_warnings():
        # A resized ``index_select(out=)`` would detach the staging view and
        # only show up as a UserWarning; treat that as the failure it is.
        warnings.simplefilter("error")
        run(views, chunks, block_ids, device, D2H, sd, chunk_tokens, BSV_BSS)
    for k, chunk in enumerate(chunks):
        ids = block_ids[k * blocks_per_chunk : (k + 1) * blocks_per_chunk]
        for layer in range(nl):
            expected = reference_rows(views[layer][ids], val_bytes)
            assert torch.equal(chunk[layer].cpu(), expected), f"chunk {k} layer {layer}"

    raw2, views2 = make_pool(nb, nl, bs, hs, pad, dtype, device, "zeros")
    sd2 = make_shape_desc(nl, nb, bs, hs, dtype, views2[0].stride(0))
    run(views2, chunks, block_ids, device, H2D, sd2, chunk_tokens, BSV_BSS)
    for layer in range(nl):
        assert torch.equal(views2[layer], views[layer]), f"layer {layer}"
    if pad is not None:
        assert torch.count_nonzero(raw2[:, nl * bs * hs :]) == 0


def test_bsv_bss_repack_is_an_exact_inverse():
    pages = torch.randint(0, 256, (3, 4, 132), dtype=torch.uint8)
    rows = torch_ops._blocked_scale_pages_to_rows(pages)
    assert rows.shape == (12, 132)
    assert torch.equal(rows, reference_rows(pages, 128))
    assert torch.equal(torch_ops._blocked_scale_rows_to_pages(rows, 4), pages)


@pytest.mark.parametrize("direction", [D2H, H2D], ids=["d2h", "h2d"])
def test_mla_rejects_wrong_rank_layers(direction):
    """A layer that is not ``[NB, BS, HS]`` must fail loudly in both
    directions instead of being resized away from the staging buffer (D2H)
    or surfacing as a torch rank error deep in ``index_copy_`` (H2D)."""
    nb, nl, bs, hs = 4, 2, 2, 8
    layers = [torch.randn(nb, 2, bs, 1, hs) for _ in range(nl)]
    chunks = [torch.zeros(nl, nb * bs, hs)]
    sd = make_shape_desc(nl, nb, bs, hs, torch.float32, 0)
    with pytest.raises(ValueError):
        torch_ops.multi_layer_block_kv_transfer(
            layers,
            chunks,
            torch.arange(nb, dtype=torch.int64),
            torch.device("cpu"),
            direction,
            sd,
            nb * bs,
            MLA,
            0,
        )


def test_pointer_mode_rejects_padding_on_non_block_axis_formats():
    """Only the block-axis MLA formats can carry ``block_stride_elems``; a
    padded pointer reconstruction of any other format stays an honest
    NotImplementedError rather than a silently wrong view."""
    nb, bs, nh, hs = 4, 2, 2, 8
    layer = torch.zeros(2, nb, bs, nh, hs)
    sd = make_shape_desc(1, nb, bs, hs, torch.float32, bs * nh * hs + 8)
    sd.nh = nh
    sd.kv_size = 2
    with pytest.raises(NotImplementedError):
        torch_ops._normalize_paged_layers(
            torch.tensor([layer.data_ptr()], dtype=torch.uint64),
            lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            shape_desc=sd,
            device="cpu",
            dtype=torch.float32,
        )


class _FakeRuntime:
    """Exercise the full ABI write without loading or calling CUDA."""

    def __init__(self, mem_type, dev, err=0, version=13000, version_err=0):
        class _Version:
            def __call__(self, destination):
                destination._obj.value = version
                return version_err

        class _Query:
            def __call__(self, destination, _ptr):
                # Check capacity before copying: the former 64-byte allocation
                # must fail safely rather than letting this regression corrupt
                # the Python process.
                size = 2 * ctypes.sizeof(ctypes.c_int) + 2 * ctypes.sizeof(
                    ctypes.c_void_p
                )
                if version >= 13000:
                    size += 8 * ctypes.sizeof(ctypes.c_long)
                assert ctypes.sizeof(destination._obj) >= size
                packed = struct.pack("ii", mem_type, dev) + bytes(size - 8)
                ctypes.memmove(destination, packed, size)
                return err

        self.cudaRuntimeGetVersion = _Version()
        self.cudaPointerGetAttributes = _Query()


@pytest.mark.parametrize("version", [12000, 13000])
@pytest.mark.parametrize(
    "mem_type,dev,expected",
    [
        (2, 1, torch.device("cuda", 1)),
        (3, 0, torch.device("cuda", 0)),
        (1, 0, torch.device("cpu")),
        (0, 0, torch.device("cpu")),
    ],
    ids=["device", "managed", "pinned-host", "unregistered"],
)
def test_resolve_ptr_device_maps_runtime_memory_types(version, mem_type, dev, expected):
    fake = _FakeRuntime(mem_type, dev, version=version)
    with (
        unittest.mock.patch.object(torch.cuda, "is_available", return_value=True),
        unittest.mock.patch.object(torch_ops, "_get_copy_lib", return_value=fake),
    ):
        assert (
            torch_ops._resolve_ptr_device(0x1000, torch.device("cuda", 0)) == expected
        )


@pytest.mark.parametrize("error", [1, 100, 700, 999])
def test_pointer_query_errors_never_become_cpu_pointers(error):
    fake = _FakeRuntime(2, 0, err=error)
    with (
        unittest.mock.patch.object(torch.cuda, "is_available", return_value=True),
        unittest.mock.patch.object(torch_ops, "_get_copy_lib", return_value=fake),
        pytest.raises(RuntimeError),
    ):
        torch_ops._resolve_ptr_device(0x1000, torch.device("cuda", 0))


@pytest.mark.parametrize(
    "runtime",
    [
        None,
        object(),
        _FakeRuntime(2, 0, version_err=999),
        _FakeRuntime(2, 0, version=11000),
        _FakeRuntime(2, 0, version=14000),
        _FakeRuntime(99, 0),
        _FakeRuntime(2, -1),
    ],
)
def test_unqualified_pointer_residency_fails_closed(runtime):
    with (
        unittest.mock.patch.object(torch.cuda, "is_available", return_value=True),
        unittest.mock.patch.object(torch_ops, "_get_copy_lib", return_value=runtime),
        pytest.raises((RuntimeError, NotImplementedError)),
    ):
        torch_ops._resolve_ptr_device(0x1000, torch.device("cuda", 0))


def test_resolve_ptr_device_without_cuda():
    with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False):
        cpu = torch.device("cpu")
        assert torch_ops._resolve_ptr_device(0x1000, cpu) == cpu
        with pytest.raises(RuntimeError):
            torch_ops._resolve_ptr_device(0x1000, torch.device("cuda", 0))


def test_cuda_alias_failure_is_not_replaced_by_a_copy():
    with (
        unittest.mock.patch.object(
            torch, "as_tensor", side_effect=RuntimeError("alias unavailable")
        ),
        pytest.raises(RuntimeError),
    ):
        torch_ops._tensor_from_cuda_ptr(
            0x1000, (8,), torch.float32, torch.device("cuda", 0), 8
        )
