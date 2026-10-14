# SPDX-License-Identifier: Apache-2.0
"""GPU parity tests for the layer-major object stride.

The layer-major staging layout orders an LMCache object by model depth, so a
given kernel group's layers are separated by the WHOLE layer extent -- every
group's slice for that depth -- rather than by that group's own per-layer
size. ``PageBufferShapeDesc.layer_stride_elems`` is what lets several
consecutive layers of one group still be moved by a single ``nl > 1`` launch.

These tests pin that arithmetic against the pre-feature behaviour: one
``nl = 1`` call per layer, each pointing at its own offset, which needs no
stride at all. A batched ``nl = NL`` call carrying ``layer_stride_elems`` must
land byte-for-byte identical. Both directions are covered: the store writes
the layout the retrieve reads, so a store-side stride bug is equally fatal.

The object buffer is deliberately PADDED between layers (stride > per-layer
extent) and the padding is poisoned. A tightly packed fixture would pass even
if ``scalars_per_layer`` ignored the field entirely, and the D2H case asserts
the padding survives -- writing past a layer's own slice would corrupt a
different kernel group.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
import lmcache.lmcache_native as lmcache_native

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip(
        "CUDA is not available, skipping the test",
        allow_module_level=True,
    )

# First Party
import lmcache.cuda_ops as cuda_ops  # noqa: E402

if not hasattr(lmcache_native.PageBufferShapeDesc(), "layer_stride_elems"):
    pytest.skip(
        "lmcache_native build predates layer_stride_elems",
        allow_module_level=True,
    )

pytestmark = pytest.mark.layerwise

_NL, _NH, _HS = 4, 2, 16
_NB, _BS = 64, 4
_CHUNK = 8  # tokens per LMCache chunk
_KV = 2  # split K/V planes
_DTYPE = torch.bfloat16
_ESIZE = 2
_FMT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
_PAGED_SHAPE = (_KV, _NB, _NH, _BS, _HS)

# Tight extent of one layer of one kernel group, in source elements.
_LAYER_ELEMS = _KV * _CHUNK * _NH * _HS
# Stand-in for the other kernel groups' slices at the same model depth.
_PAD_ELEMS = 96
_STRIDE_ELEMS = _LAYER_ELEMS + _PAD_ELEMS
_POISON = -7.5
_N_OBJECTS = 2
_BLOCKS_PER_CHUNK = _CHUNK // _BS


def _desc(nl, layer_stride_elems):
    """Shape desc for ``nl`` layers of one kernel group.

    ``kv_interleaved`` is mandatory: ``layer_stride_elems`` is honoured only
    by the L2TD branch. At ``nl == 1`` the L2TD and 2LTD offset formulas
    coincide, so the reference and batched calls stay comparable.
    """
    d = lmcache_native.PageBufferShapeDesc()
    d.kv_size = _KV
    d.nl = nl
    d.nb = _NB
    d.bs = _BS
    d.nh = _NH
    d.hs = _HS
    d.element_size = _ESIZE
    d.block_stride_elems = 0
    d.kv_interleaved = True
    d.layer_stride_elems = layer_stride_elems
    return d


def _transfer(direction, paged_ptrs, object_ptrs, block_ids, desc):
    cuda_ops.multi_layer_block_kv_transfer(
        paged_ptrs,
        object_ptrs,
        block_ids,
        block_ids.device,
        direction,
        desc,
        _CHUNK,
        _FMT,
        0,  # skip_prefix_n_blocks
    )


def _ptr_tensor(tensors, dev):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.long, device=dev)


def _strided_objects(dev, fill):
    """``_N_OBJECTS`` layer-major objects, poisoned in the inter-layer gaps."""
    bufs = []
    for _ in range(_N_OBJECTS):
        buf = torch.full((_NL * _STRIDE_ELEMS,), _POISON, dtype=_DTYPE, device=dev)
        if fill:
            for layer in range(_NL):
                base = layer * _STRIDE_ELEMS
                buf[base : base + _LAYER_ELEMS] = torch.randn(
                    _LAYER_ELEMS, dtype=_DTYPE, device=dev
                )
        bufs.append(buf)
    return bufs


def _layer_slice(buf, layer):
    base = layer * _STRIDE_ELEMS
    return buf[base : base + _LAYER_ELEMS]


def _pad_slice(buf, layer):
    base = layer * _STRIDE_ELEMS + _LAYER_ELEMS
    return buf[base : base + _PAD_ELEMS]


@pytest.fixture(name="block_ids")
def _block_ids():
    dev = torch.device(torch_device_type)
    # Non-identity, non-contiguous mapping so a dropped block term shows up.
    return torch.tensor([11, 5, 40, 23], dtype=torch.long, device=dev)


def test_h2d_layer_stride_matches_per_layer_calls(block_ids):
    """One strided ``nl = 4`` scatter == four tight ``nl = 1`` scatters."""
    dev = torch.device(torch_device_type)
    torch.manual_seed(0)

    objects = _strided_objects(dev, fill=True)
    obj_ptrs = [b.data_ptr() for b in objects]

    paged_ref = [
        torch.zeros(*_PAGED_SHAPE, dtype=_DTYPE, device=dev) for _ in range(_NL)
    ]
    paged_new = [torch.zeros_like(t) for t in paged_ref]

    # Reference: pre-feature form -- one call per layer, no stride, each
    # object pointer pre-advanced to that layer's slice by the host.
    for layer in range(_NL):
        _transfer(
            lmcache_native.TransferDirection.H2D,
            _ptr_tensor([paged_ref[layer]], dev),
            [p + layer * _STRIDE_ELEMS * _ESIZE for p in obj_ptrs],
            block_ids,
            _desc(nl=1, layer_stride_elems=0),
        )

    # Under test: a single batched launch walking layers by the stride.
    _transfer(
        lmcache_native.TransferDirection.H2D,
        _ptr_tensor(paged_new, dev),
        obj_ptrs,
        block_ids,
        _desc(nl=_NL, layer_stride_elems=_STRIDE_ELEMS),
    )
    torch_dev.synchronize()

    for layer in range(_NL):
        assert torch.equal(paged_ref[layer], paged_new[layer]), (
            f"H2D layer {layer} mismatch: batched nl={_NL} scatter diverged "
            f"from the per-layer reference"
        )


def test_d2h_layer_stride_matches_per_layer_calls(block_ids):
    """Store side: the batched gather must fill exactly its own slices."""
    dev = torch.device(torch_device_type)
    torch.manual_seed(1)

    paged = [torch.randn(*_PAGED_SHAPE, dtype=_DTYPE, device=dev) for _ in range(_NL)]

    # Reference: one tight object per layer, gathered one layer at a time.
    ref = [
        [
            torch.full((_LAYER_ELEMS,), _POISON, dtype=_DTYPE, device=dev)
            for _ in range(_N_OBJECTS)
        ]
        for _ in range(_NL)
    ]
    for layer in range(_NL):
        _transfer(
            lmcache_native.TransferDirection.D2H,
            _ptr_tensor([paged[layer]], dev),
            [b.data_ptr() for b in ref[layer]],
            block_ids,
            _desc(nl=1, layer_stride_elems=0),
        )

    new = _strided_objects(dev, fill=False)
    _transfer(
        lmcache_native.TransferDirection.D2H,
        _ptr_tensor(paged, dev),
        [b.data_ptr() for b in new],
        block_ids,
        _desc(nl=_NL, layer_stride_elems=_STRIDE_ELEMS),
    )
    torch_dev.synchronize()

    for layer in range(_NL):
        for obj in range(_N_OBJECTS):
            assert torch.equal(ref[layer][obj], _layer_slice(new[obj], layer)), (
                f"D2H object {obj} layer {layer} mismatch"
            )

    # The gaps stand in for other kernel groups' bytes: a store that writes
    # past its own per-layer extent silently corrupts a different group.
    for layer in range(_NL):
        for obj in range(_N_OBJECTS):
            pad = _pad_slice(new[obj], layer)
            assert torch.equal(pad, torch.full_like(pad, _POISON)), (
                f"D2H wrote into the inter-layer gap (object {obj}, layer {layer})"
            )
