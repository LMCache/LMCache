# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the direct copy-engine transfer (``cudaMemcpyBatchAsync``).

``execute_direct_copy_transfer`` must produce exactly the bytes the staged
kernel path produces: for each eligible layout and both directions, the same
pinned host objects and the same random paged buffers are pushed through the
block transfer kernel (via a GPU staging copy) and through the direct path, and
the results are compared bit for bit. The plan exercises several chunks, a
skipped block prefix, a two-kernel-group object (non-zero byte offset in the
object), a padded MLA block stride, and a small pin-chunk alignment so entries
are split at the allocator's virtual boundaries.
"""

# Standard
from dataclasses import dataclass
import random

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type

pytest.importorskip(
    "lmcache.cuda_ops",
    reason="Requires CUDA extension lmcache.cuda_ops",
)

# First Party
import lmcache.cuda_ops as cuda_ops  # noqa: E402
import lmcache.lmcache_native as lmcache_native  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="Requires CUDA backend",
    ),
    pytest.mark.skipif(
        not hasattr(cuda_ops, "execute_direct_copy_transfer")
        or not cuda_ops.batch_memcpy_supported(),
        reason="cudaMemcpyBatchAsync unavailable (needs CUDA >= 12.8 build/driver)",
    ),
]

Fmt = lmcache_native.EngineKVFormat
H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H

_NB = 64  # paged blocks per layer
_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class _Geometry:
    """One kernel group's paged layout."""

    fmt: "Fmt"
    nl: int
    kv_size: int
    nh: int
    hs: int
    bs: int
    block_stride_elems: int = 0  # padded dim-0 stride (MLA only), 0 = tight

    @property
    def row_elems(self) -> int:
        return self.nh * self.hs

    @property
    def tight_block_elems(self) -> int:
        return self.bs * self.row_elems


def _paged_tensors(g: _Geometry, device: torch.device) -> list[torch.Tensor]:
    """Random paged buffers in the physical layout of ``g.fmt``."""
    nb, bs, nh, hs, nl = _NB, g.bs, g.nh, g.hs, g.nl

    def rnd(shape: list[int]) -> torch.Tensor:
        return torch.rand(shape, dtype=_DTYPE, device=device)

    if g.fmt == Fmt.NL_X_TWO_NB_BS_NH_HS:
        return [rnd([2, nb, bs, nh, hs]) for _ in range(nl)]
    if g.fmt == Fmt.NL_X_NB_TWO_BS_NH_HS:
        return [rnd([nb, 2, bs, nh, hs]) for _ in range(nl)]
    if g.fmt == Fmt.NB_NL_TWO_BS_NH_HS:
        return [rnd([nb, nl, 2, bs, nh, hs])]
    if g.fmt == Fmt.NL_X_NB_BS_HS:
        if g.block_stride_elems:
            # Pool-shared rows: each block row is wider than bs * hs, the
            # group views the leading bs * hs elements (dim-0 stride padded,
            # tokens contiguous inside the block).
            return [
                rnd([nb, g.block_stride_elems])[:, : bs * hs].view(nb, bs, hs)
                for _ in range(nl)
            ]
        return [rnd([nb, bs, hs]) for _ in range(nl)]
    if g.fmt == Fmt.NL_X_NBBS_ONE_HS:
        return [rnd([nb * bs, 1, hs]) for _ in range(nl)]
    if g.fmt in (Fmt.TWO_X_NL_X_NBBS_NH_HS,):
        return [rnd([nb * bs, nh, hs]) for _ in range(2 * nl)]
    if g.fmt in (Fmt.TWO_X_NL_X_NB_BS_NH_HS,):
        return [rnd([nb, bs, nh, hs]) for _ in range(2 * nl)]
    if g.fmt in (Fmt.NL_X_NB_BS_NH_TWO_HS, Fmt.NL_X_NB_BS_NH_CS):
        return [rnd([nb, bs, nh, hs]) for _ in range(nl)]  # hs already fused
    raise ValueError(f"unsupported test format {g.fmt}")


def _shape_desc(g: _Geometry) -> "lmcache_native.PageBufferShapeDesc":
    sd = lmcache_native.PageBufferShapeDesc()
    sd.kv_size = g.kv_size
    sd.nl = g.nl
    sd.nb = _NB
    sd.bs = g.bs
    sd.nh = g.nh
    sd.hs = g.hs
    sd.element_size = _DTYPE.itemsize
    sd.block_stride_elems = g.block_stride_elems
    return sd


def _object_bytes(g: _Geometry, slots: int) -> int:
    return g.kv_size * g.nl * slots * g.row_elems * _DTYPE.itemsize


_GEOMETRIES = {
    "vllm_nhd": _Geometry(Fmt.NL_X_TWO_NB_BS_NH_HS, nl=3, kv_size=2, nh=2, hs=16, bs=8),
    "flashinfer": _Geometry(
        Fmt.NL_X_NB_TWO_BS_NH_HS, nl=3, kv_size=2, nh=2, hs=16, bs=8
    ),
    "cross_layer": _Geometry(
        Fmt.NB_NL_TWO_BS_NH_HS, nl=3, kv_size=2, nh=2, hs=16, bs=8
    ),
    "mla": _Geometry(Fmt.NL_X_NB_BS_HS, nl=3, kv_size=1, nh=1, hs=64, bs=8),
    "mla_padded": _Geometry(
        Fmt.NL_X_NB_BS_HS, nl=3, kv_size=1, nh=1, hs=64, bs=8, block_stride_elems=8 * 96
    ),
    "sglang_mla": _Geometry(Fmt.NL_X_NBBS_ONE_HS, nl=3, kv_size=1, nh=1, hs=64, bs=8),
    "sglang_mha": _Geometry(
        Fmt.TWO_X_NL_X_NBBS_NH_HS, nl=3, kv_size=2, nh=2, hs=16, bs=8
    ),
    "sglang_mha_mp": _Geometry(
        Fmt.TWO_X_NL_X_NB_BS_NH_HS, nl=3, kv_size=2, nh=2, hs=16, bs=8
    ),
    "fused_nhd": _Geometry(
        Fmt.NL_X_NB_BS_NH_TWO_HS, nl=3, kv_size=1, nh=2, hs=32, bs=8
    ),
    "cs_nhd": _Geometry(Fmt.NL_X_NB_BS_NH_CS, nl=3, kv_size=1, nh=2, hs=32, bs=8),
}

_CHUNK_TOKENS = 32  # 4 blocks of 8 tokens per chunk
_NUM_CHUNKS = 3
_SKIP_BLOCKS_FIRST_CHUNK = 1
_PIN_ALIGNMENT = 1 << 12  # 4 KB: forces entry splitting in the test objects


def _kernel_reference(
    groups: list[_Geometry],
    paged: list[list[torch.Tensor]],
    host_objs: list[torch.Tensor],
    block_ids: list[list[int]],
    direction: "lmcache_native.TransferDirection",
    device: torch.device,
) -> None:
    """Run the staged kernel path chunk by chunk (batch size 1)."""
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + _object_bytes(g, _CHUNK_TOKENS))
    blocks_per_chunk = _CHUNK_TOKENS // groups[0].bs
    for chunk_idx, host_obj in enumerate(host_objs):
        # Zeroed so the skipped block prefix reads back as zero, like the
        # untouched pinned object on the direct path.
        staging = torch.zeros_like(host_obj, device=device)
        if direction == H2D:
            staging.copy_(host_obj)
        skip = _SKIP_BLOCKS_FIRST_CHUNK if chunk_idx == 0 else 0
        for gi, g in enumerate(groups):
            ptrs = torch.tensor(
                [t.data_ptr() for t in paged[gi]], dtype=torch.int64, device=device
            )
            ids = torch.tensor(
                block_ids[gi][
                    chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
                ],
                dtype=torch.int64,
                device=device,
            )
            region = staging[offsets[gi] : offsets[gi + 1]]
            cuda_ops.multi_layer_block_kv_transfer(
                ptrs,
                [region.data_ptr()],
                ids,
                device,
                direction,
                _shape_desc(g),
                _CHUNK_TOKENS,
                g.fmt,
                skip,
            )
        if direction == D2H:
            host_obj.copy_(staging)
    torch.cuda.synchronize(device)


def _direct(
    groups: list[_Geometry],
    paged: list[list[torch.Tensor]],
    host_objs: list[torch.Tensor],
    block_ids: list[list[int]],
    direction: "lmcache_native.TransferDirection",
    device: torch.device,
) -> None:
    offset = 0
    specs = []
    for gi, g in enumerate(groups):
        specs.append(
            cuda_ops.DirectCopyGroupSpec(
                [t.data_ptr() for t in paged[gi]],
                _shape_desc(g),
                g.fmt,
                _CHUNK_TOKENS,
                offset,
                block_ids[gi],
            )
        )
        offset += _object_bytes(g, _CHUNK_TOKENS)
    objects = []
    for chunk_idx, host_obj in enumerate(host_objs):
        skip = _SKIP_BLOCKS_FIRST_CHUNK if chunk_idx == 0 else 0
        objects.append(
            cuda_ops.DirectCopyObject(
                host_obj.data_ptr(),
                # Distinct virtual offsets so pin-boundary splitting differs
                # per object and is not aligned with the object start.
                chunk_idx * host_obj.nbytes + 1000,
                host_obj.nbytes,
                chunk_idx,
                [skip] * len(groups),
            )
        )
    cuda_ops.execute_direct_copy_transfer(
        direction, device, _PIN_ALIGNMENT, specs, objects
    )
    torch.cuda.synchronize(device)


def _pinned_objects(groups: list[_Geometry], fill: bool) -> list[torch.Tensor]:
    nbytes = sum(_object_bytes(g, _CHUNK_TOKENS) for g in groups)
    objs = []
    for _ in range(_NUM_CHUNKS):
        t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        if fill:
            t.copy_(torch.randint(0, 256, (nbytes,), dtype=torch.uint8))
        else:
            t.zero_()
        objs.append(t)
    return objs


def _block_ids(groups: list[_Geometry]) -> list[list[int]]:
    blocks_per_chunk = _CHUNK_TOKENS // groups[0].bs
    total = blocks_per_chunk * _NUM_CHUNKS
    return [random.sample(range(_NB), total) for _ in groups]


def _clone_paged(paged: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
    """Clone keeping the physical strides (the padded MLA case)."""
    return [
        [
            torch.empty_strided(
                t.shape, t.stride(), dtype=t.dtype, device=t.device
            ).copy_(t)
            for t in layer_ptrs
        ]
        for layer_ptrs in paged
    ]


def _flat_paged(paged: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    return [t for layer_ptrs in paged for t in layer_ptrs]


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare raw bits: the host objects hold random bytes, some of which
    decode to bf16 NaNs that ``torch.equal`` would report as unequal."""
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(1234)
    torch.manual_seed(1234)


@pytest.mark.parametrize("name", sorted(_GEOMETRIES))
@pytest.mark.parametrize("direction", [H2D, D2H], ids=["retrieve", "store"])
def test_direct_matches_kernel_single_group(name: str, direction) -> None:
    """Each eligible layout scatters/gathers exactly like the kernel path."""
    device = torch.device("cuda:0")
    groups = [_GEOMETRIES[name]]
    paged_kernel = [_paged_tensors(groups[0], device)]
    paged_direct = _clone_paged(paged_kernel)
    block_ids = _block_ids(groups)

    if direction == H2D:
        host_kernel = _pinned_objects(groups, fill=True)
        host_direct = [t.clone().pin_memory() for t in host_kernel]
    else:
        host_kernel = _pinned_objects(groups, fill=False)
        host_direct = _pinned_objects(groups, fill=False)

    _kernel_reference(groups, paged_kernel, host_kernel, block_ids, direction, device)
    _direct(groups, paged_direct, host_direct, block_ids, direction, device)

    if direction == H2D:
        for a, b in zip(
            _flat_paged(paged_kernel), _flat_paged(paged_direct), strict=True
        ):
            assert _bitwise_equal(a, b)
    else:
        for a, b in zip(host_kernel, host_direct, strict=True):
            assert _bitwise_equal(a, b)
        # The skipped prefix block of chunk 0 must stay untouched (zero).
        g = groups[0]
        first_block_bytes = g.tight_block_elems * _DTYPE.itemsize
        assert int(host_direct[0][:first_block_bytes].sum()) == 0


@pytest.mark.parametrize("direction", [H2D, D2H], ids=["retrieve", "store"])
def test_direct_matches_kernel_two_groups_in_one_object(direction) -> None:
    """A second kernel group lives at a non-zero byte offset in the object."""
    device = torch.device("cuda:0")
    groups = [_GEOMETRIES["mla"], _GEOMETRIES["vllm_nhd"]]
    paged_kernel = [_paged_tensors(g, device) for g in groups]
    paged_direct = _clone_paged(paged_kernel)
    block_ids = _block_ids(groups)

    if direction == H2D:
        host_kernel = _pinned_objects(groups, fill=True)
        host_direct = [t.clone().pin_memory() for t in host_kernel]
    else:
        host_kernel = _pinned_objects(groups, fill=False)
        host_direct = _pinned_objects(groups, fill=False)

    _kernel_reference(groups, paged_kernel, host_kernel, block_ids, direction, device)
    _direct(groups, paged_direct, host_direct, block_ids, direction, device)

    if direction == H2D:
        for a, b in zip(
            _flat_paged(paged_kernel), _flat_paged(paged_direct), strict=True
        ):
            assert _bitwise_equal(a, b)
    else:
        for a, b in zip(host_kernel, host_direct, strict=True):
            assert _bitwise_equal(a, b)


def test_direct_rejects_hnd_layout() -> None:
    """HND layouts are refused up front instead of copying garbage."""
    device = torch.device("cuda:0")
    hnd = Fmt.NL_X_TWO_NB_NH_BS_HS
    assert not cuda_ops.direct_copy_format_supported(hnd)
    g = _GEOMETRIES["vllm_nhd"]
    paged = [torch.zeros([2, _NB, g.nh, g.bs, g.hs], dtype=_DTYPE, device=device)]
    host = torch.zeros(_object_bytes(g, _CHUNK_TOKENS), dtype=torch.uint8).pin_memory()
    spec = cuda_ops.DirectCopyGroupSpec(
        [t.data_ptr() for t in paged], _shape_desc(g), hnd, _CHUNK_TOKENS, 0, [0] * 4
    )
    obj = cuda_ops.DirectCopyObject(host.data_ptr(), 0, host.nbytes, 0, [0])
    with pytest.raises(RuntimeError, match="not eligible"):
        cuda_ops.execute_direct_copy_transfer(
            H2D, device, _PIN_ALIGNMENT, [spec], [obj]
        )


def test_direct_rejects_out_of_range_block_and_short_object() -> None:
    """Bounds are checked on the host before anything is enqueued."""
    device = torch.device("cuda:0")
    g = _GEOMETRIES["mla"]
    paged = [_paged_tensors(g, device)]
    host = torch.zeros(_object_bytes(g, _CHUNK_TOKENS), dtype=torch.uint8).pin_memory()

    bad_ids = [0, 1, 2, _NB]  # last id is one past the pool
    spec = cuda_ops.DirectCopyGroupSpec(
        [t.data_ptr() for t in paged[0]],
        _shape_desc(g),
        g.fmt,
        _CHUNK_TOKENS,
        0,
        bad_ids,
    )
    obj = cuda_ops.DirectCopyObject(host.data_ptr(), 0, host.nbytes, 0, [0])
    with pytest.raises(RuntimeError, match="block id"):
        cuda_ops.execute_direct_copy_transfer(
            H2D, device, _PIN_ALIGNMENT, [spec], [obj]
        )

    spec_ok = cuda_ops.DirectCopyGroupSpec(
        [t.data_ptr() for t in paged[0]],
        _shape_desc(g),
        g.fmt,
        _CHUNK_TOKENS,
        0,
        [0, 1, 2, 3],
    )
    short = cuda_ops.DirectCopyObject(host.data_ptr(), 0, host.nbytes - 1, 0, [0])
    with pytest.raises(RuntimeError, match="past the object size"):
        cuda_ops.execute_direct_copy_transfer(
            H2D, device, _PIN_ALIGNMENT, [spec_ok], [short]
        )


def test_format_eligibility_table() -> None:
    """Token-major contiguous-block layouts qualify; HND/blocked-scale do not."""
    eligible = {
        Fmt.NB_NL_TWO_BS_NH_HS,
        Fmt.NL_X_TWO_NB_BS_NH_HS,
        Fmt.NL_X_NB_TWO_BS_NH_HS,
        Fmt.NL_X_NB_BS_HS,
        Fmt.TWO_X_NL_X_NBBS_NH_HS,
        Fmt.NL_X_NBBS_ONE_HS,
        Fmt.TWO_X_NL_X_NB_BS_NH_HS,
        Fmt.NL_X_NB_BS_NH_TWO_HS,
        Fmt.NL_X_NB_BS_NH_CS,
    }
    for fmt in Fmt.__members__.values():
        assert cuda_ops.direct_copy_format_supported(fmt) == (fmt in eligible), fmt
