# SPDX-License-Identifier: Apache-2.0
"""NL_X_NB_BSV_BSS: blocked-scale DSA indexer cache transfers.

vLLM lays an indexer-cache page out blocked — per 64-token block, all tokens'
128-byte fp8 values first, then all tokens' 4-byte fp32 scales. LMCache's
chunk rows are canonical token-major ``[vals | scale]`` (132 B). These tests
build a synthetic blocked page, gather it token-granular (D2H), scatter to a
DIFFERENTLY-ALIGNED destination (H2D), and require value-exactness — the
mod-block_size misalignment that silently garbled GLM CacheBlend reuse.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type  # noqa: E402

if torch_device_type != "cuda" or not torch_dev.is_available():
    pytest.skip(
        f"blocked-scale format test requires CUDA runtime, got {torch_device_type}",
        allow_module_level=True,
    )

# First Party
import lmcache.cuda_ops as cuda_ops  # noqa: E402
import lmcache.lmcache_native as lmcache_native

if not hasattr(lmcache_native.EngineKVFormat, "NL_X_NB_BSV_BSS"):
    pytest.skip("cuda_ops build lacks NL_X_NB_BSV_BSS", allow_module_level=True)

_BS = 64  # tokens per block
_NB = 8
_NL = 3
# (value bytes, scale bytes) per token: the fp8 DSA indexer row (fp32 scale,
# 4-byte transfer word) and DeepSeek-V4's packed main KV on SM90 (8 B of UE8M0
# scales, 8-byte transfer word). Module globals are rebound per record.
_RECORDS = {132: (128, 4), 584: (576, 8)}
_HD = 128
_SC = 4
_ROW = _HD + _SC


@pytest.fixture(params=sorted(_RECORDS), ids=lambda r: f"row{r}")
def record(request):
    global _HD, _SC, _ROW
    _HD, _SC = _RECORDS[request.param]
    _ROW = _HD + _SC
    return request.param


def _blocked_page_write(cache, layer, slot, row_bytes):
    """Write one token's row into the BLOCKED page layout (the ground truth
    from vllm's indexer_k_quant_and_cache / quantize_and_insert_k_cache)."""
    b, off = slot // _BS, slot % _BS
    flat = cache[layer].view(-1)
    base = b * _BS * _ROW
    flat[base + off * _HD : base + off * _HD + _HD] = row_bytes[:_HD]
    sbase = base + _BS * _HD
    flat[sbase + off * _SC : sbase + off * _SC + _SC] = row_bytes[_HD:]


def _blocked_page_read(cache, layer, slot):
    b, off = slot // _BS, slot % _BS
    flat = cache[layer].view(-1)
    base = b * _BS * _ROW
    vals = flat[base + off * _HD : base + off * _HD + _HD]
    sbase = base + _BS * _HD
    scale = flat[sbase + off * _SC : sbase + off * _SC + _SC]
    return torch.cat([vals, scale])


def _make_paged():
    """Per-layer [NB, BS, 132] uint8 tensors + a pointer array."""
    caches = [
        torch.zeros(_NB, _BS, _ROW, dtype=torch.uint8, device=torch_device_type)
        for _ in range(_NL)
    ]
    ptrs = torch.tensor(
        [c.data_ptr() for c in caches], dtype=torch.uint64, device=torch_device_type
    )
    return caches, ptrs


def _write_all(caches, slots, rows):
    for layer in range(_NL):
        for i, slot in enumerate(slots.tolist()):
            _blocked_page_write(caches, layer, slot, rows[layer, i])


def _read_all(caches, slots):
    out = []
    for layer in range(_NL):
        out.append(
            torch.stack([_blocked_page_read(caches, layer, s) for s in slots.tolist()])
        )
    return torch.stack(out)


def test_blocked_roundtrip_value_exact_across_alignments(record):
    torch.manual_seed(0)
    n_tok = 96
    src, src_ptrs = _make_paged()
    rows = torch.randint(
        0, 255, (_NL, n_tok, _ROW), dtype=torch.uint8, device=torch_device_type
    )
    src_slots = torch.arange(
        32, 32 + n_tok, dtype=torch.int64, device=torch_device_type
    )
    _write_all(src, src_slots, rows)

    # D2H-style gather into a token-major chunk buffer (kv=1, NL, n_tok, 132).
    chunk = torch.zeros(
        1, _NL, n_tok, _ROW, dtype=torch.uint8, device=torch_device_type
    )
    cuda_ops.multi_layer_kv_transfer(
        chunk,
        src_ptrs,
        src_slots,
        torch.device(torch_device_type),
        _NB * _BS,
        lmcache_native.TransferDirection.D2H,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        block_size=_BS,
        head_size=0,
    )
    torch_dev.synchronize()
    assert torch.equal(chunk[0], rows), "gather must produce token-major rows"

    # H2D scatter to a DIFFERENT intra-block alignment (delta 17 mod 64 != 0).
    dst, dst_ptrs = _make_paged()
    dst_slots = torch.arange(
        17, 17 + n_tok, dtype=torch.int64, device=torch_device_type
    )
    cuda_ops.multi_layer_kv_transfer(
        chunk,
        dst_ptrs,
        dst_slots,
        torch.device(torch_device_type),
        _NB * _BS,
        lmcache_native.TransferDirection.H2D,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        block_size=_BS,
        head_size=0,
    )
    torch_dev.synchronize()
    got = _read_all(dst, dst_slots)
    assert torch.equal(got, rows), (
        "misaligned scatter must land every token's values AND scale at the "
        "blocked-layout positions of the destination slots"
    )


def test_blocked_roundtrip_padded_pages():
    """584-byte records in 19008-byte padded pages: the kernel must step blocks by the
    real stride, not the tight block_size * row (every state past block 0 was wrong)."""
    torch.manual_seed(1)
    bs, row, scale = 32, 584, 8
    vals = row - scale
    stride = ((bs * row + 575) // 576) * 576  # 19008
    nb, nl, n_tok = 4, 2, 48

    def pages():
        caches = [
            torch.zeros(nb, stride, dtype=torch.uint8, device=torch_device_type)
            for _ in range(nl)
        ]
        ptrs = torch.tensor(
            [c.data_ptr() for c in caches], dtype=torch.uint64, device=torch_device_type
        )
        return caches, ptrs

    def write(cache, slot, row_bytes):
        b, off = divmod(slot, bs)
        flat = cache[b]
        flat[off * vals : (off + 1) * vals] = row_bytes[:vals]
        flat[bs * vals + off * scale : bs * vals + (off + 1) * scale] = row_bytes[vals:]

    def read(cache, slot):
        b, off = divmod(slot, bs)
        flat = cache[b]
        return torch.cat(
            [
                flat[off * vals : (off + 1) * vals],
                flat[bs * vals + off * scale : bs * vals + (off + 1) * scale],
            ]
        )

    rows = torch.randint(
        0, 255, (nl, n_tok, row), dtype=torch.uint8, device=torch_device_type
    )
    src, src_ptrs = pages()
    src_slots = torch.arange(
        40, 40 + n_tok, dtype=torch.int64, device=torch_device_type
    )
    for layer in range(nl):
        for i, slot in enumerate(src_slots.tolist()):
            write(src[layer], slot, rows[layer, i])

    chunk = torch.zeros(1, nl, n_tok, row, dtype=torch.uint8, device=torch_device_type)
    cuda_ops.multi_layer_kv_transfer(
        chunk,
        src_ptrs,
        src_slots,
        torch.device(torch_device_type),
        nb * bs,
        lmcache_native.TransferDirection.D2H,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        bs,
        0,
        0,
        stride,
    )
    torch_dev.synchronize()
    assert torch.equal(chunk[0], rows)

    dst, dst_ptrs = pages()
    dst_slots = torch.arange(
        17, 17 + n_tok, dtype=torch.int64, device=torch_device_type
    )
    cuda_ops.multi_layer_kv_transfer(
        chunk,
        dst_ptrs,
        dst_slots,
        torch.device(torch_device_type),
        nb * bs,
        lmcache_native.TransferDirection.H2D,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        bs,
        0,
        0,
        stride,
    )
    torch_dev.synchronize()
    got = torch.stack(
        [
            torch.stack([read(dst[layer], s) for s in dst_slots.tolist()])
            for layer in range(nl)
        ]
    )
    assert torch.equal(got, rows)


def test_batched_block_transfer_584_records_live_geometry():
    """The batched store/retrieve kernel on DeepSeek-V4.1's live geometry: three
    ratio-2 layer pages (584 B x 32 states) interleaved in one 116928-byte
    engine-group block. The per-token scale is 8 bytes (one 8-byte transfer
    word), not the indexer's 4-byte fp32 — a 4-byte word drifts every record."""
    torch.manual_seed(2)
    bs, row, scale = 32, 584, 8
    vals = row - scale
    page = bs * row  # 18688
    block_bytes, nl, nb = 116928, 3, 8
    dev = torch.device(torch_device_type)

    def make():
        big = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=dev)
        layers = [
            big[:, k * page : (k + 1) * page].view(nb, bs, row) for k in range(nl)
        ]
        ptrs = torch.tensor(
            [t.data_ptr() for t in layers], dtype=torch.uint64, device=dev
        )
        return big, ptrs

    def write(big, layer, slot, row_bytes):
        b, off = divmod(slot, bs)
        base = layer * page
        big[b, base + off * vals : base + (off + 1) * vals] = row_bytes[:vals]
        sb = base + bs * vals
        big[b, sb + off * scale : sb + (off + 1) * scale] = row_bytes[vals:]

    def read(big, layer, slot):
        b, off = divmod(slot, bs)
        base = layer * page
        sb = base + bs * vals
        return torch.cat(
            [
                big[b, base + off * vals : base + (off + 1) * vals],
                big[b, sb + off * scale : sb + (off + 1) * scale],
            ]
        )

    shape_desc = lmcache_native.PageBufferShapeDesc()
    shape_desc.nl, shape_desc.nb, shape_desc.bs = nl, nb, bs
    shape_desc.nh, shape_desc.hs, shape_desc.element_size, shape_desc.kv_size = (
        1,
        row,
        1,
        1,
    )
    shape_desc.block_stride_elems = block_bytes
    blocks = [5, 1, 6, 2]  # one 256-token chunk = 128 states = 4 blocks
    slots = [b * bs + s for b in blocks for s in range(bs)]
    rows = torch.randint(0, 255, (nl, len(slots), row), dtype=torch.uint8, device=dev)
    src, src_ptrs = make()
    for layer in range(nl):
        for i, s in enumerate(slots):
            write(src, layer, s, rows[layer, i])
    chunk = torch.zeros(1, nl, len(slots), row, dtype=torch.uint8, device=dev)
    cuda_ops.multi_layer_block_kv_transfer(
        src_ptrs,
        [chunk.data_ptr()],
        torch.tensor(blocks, dtype=torch.int64, device=dev),
        dev,
        lmcache_native.TransferDirection.D2H,
        shape_desc,
        len(slots),
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        0,
    )
    torch_dev.synchronize()
    assert torch.equal(chunk[0], rows)
    dst, dst_ptrs = make()
    dblocks = [3, 7, 0, 4]
    cuda_ops.multi_layer_block_kv_transfer(
        dst_ptrs,
        [chunk.data_ptr()],
        torch.tensor(dblocks, dtype=torch.int64, device=dev),
        dev,
        lmcache_native.TransferDirection.H2D,
        shape_desc,
        len(slots),
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        0,
    )
    torch_dev.synchronize()
    dslots = [b * bs + s for b in dblocks for s in range(bs)]
    got = torch.stack(
        [torch.stack([read(dst, layer, s) for s in dslots]) for layer in range(nl)]
    )
    assert torch.equal(got, rows)
