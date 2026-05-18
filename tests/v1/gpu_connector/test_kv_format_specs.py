# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the strategy-based KV format dispatch
(:mod:`lmcache.v1.gpu_connector.kv_format`).

These tests are CPU-only on purpose: they exercise the Python-side
shape access and registry plumbing, not the CUDA transfer kernels.
The CUDA-only sibling file ``test_utils_shape_desc.py`` exercises the
``PageBufferShapeDesc`` build path and the kernel-facing data pointers.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector import utils as facade
from lmcache.v1.gpu_connector.kv_format import (
    KVFormatSpec,
    all_gpu_kv_formats,
    detect_format,
    get_spec,
    get_spec_class,
)
import lmcache.c_ops as lmc_ops


# ----------------------------------------------------------------------
# Helpers — build minimal kv_caches values per format.
# ----------------------------------------------------------------------
def _vllm_fa_nhd(nl=4, nb=32, bs=16, nh=8, hs=64):
    """NL x [2, NB, BS, NH, HS]."""
    return [torch.empty(2, nb, bs, nh, hs, dtype=torch.bfloat16) for _ in range(nl)]


def _vllm_fi_nhd(nl=4, nb=32, bs=16, nh=8, hs=64):
    """NL x [NB, 2, BS, NH, HS]."""
    return [torch.empty(nb, 2, bs, nh, hs, dtype=torch.bfloat16) for _ in range(nl)]


def _vllm_fa_hnd(nl=4, nb=32, bs=16, nh=8, hs=64):
    """NL x [2, NB, NH, BS, HS]."""
    return [torch.empty(2, nb, nh, bs, hs, dtype=torch.bfloat16) for _ in range(nl)]


def _vllm_fi_hnd(nl=4, nb=32, bs=16, nh=8, hs=64):
    """NL x [NB, 2, NH, BS, HS]."""
    return [torch.empty(nb, 2, nh, bs, hs, dtype=torch.bfloat16) for _ in range(nl)]


def _vllm_mla(nl=3, nb=32, bs=16, hs=512):
    """NL x [NB, BS, HS]."""
    return [torch.empty(nb, bs, hs, dtype=torch.bfloat16) for _ in range(nl)]


def _sglang_mla(nl=2, pbs=512, hs=128):
    """NL x [PBS, 1, HS]."""
    return [torch.empty(pbs, 1, hs, dtype=torch.bfloat16) for _ in range(nl)]


def _sglang_mha(nl=4, pbs=512, nh=8, hs=64):
    """2 x NL x [PBS, NH, HS]."""
    k = [torch.empty(pbs, nh, hs, dtype=torch.bfloat16) for _ in range(nl)]
    v = [torch.empty(pbs, nh, hs, dtype=torch.bfloat16) for _ in range(nl)]
    return [k, v]


def _cross_layer_nhd(nb=32, nl=80, bs=16, nh=8, hs=64):
    """[NB, NL, 2, BS, NH, HS]."""
    return torch.empty(nb, nl, 2, bs, nh, hs, dtype=torch.bfloat16)


def _cross_layer_hnd(nb=32, nl=80, bs=16, nh=8, hs=64):
    """[NB, NL, 2, NH, BS, HS]."""
    return torch.empty(nb, nl, 2, nh, bs, hs, dtype=torch.bfloat16)


# ----------------------------------------------------------------------
# Registry coverage — every enum value has a registered spec class.
# ----------------------------------------------------------------------
def test_every_enum_value_has_a_spec():
    """Adding a new GPUKVFormat without a spec must fail this test."""
    enum_values = {
        getattr(lmc_ops.GPUKVFormat, name)
        for name in dir(lmc_ops.GPUKVFormat)
        if not name.startswith("_") and name.isupper()
    }
    registered = set(all_gpu_kv_formats())
    missing = enum_values - registered
    assert not missing, f"GPUKVFormat values without a spec: {missing}"


@pytest.mark.parametrize("fmt", list(all_gpu_kv_formats()))
def test_spec_class_metadata_filled(fmt):
    cls = get_spec_class(fmt)
    assert cls is not None
    assert isinstance(cls.format_id, str) and cls.format_id
    assert cls.gpu_kv_format == fmt
    assert cls.shape_desc and isinstance(cls.shape_desc, str)
    assert cls.backend_label and isinstance(cls.backend_label, str)
    assert issubclass(cls, KVFormatSpec)


# ----------------------------------------------------------------------
# Spec accessors — basic shape lookups must match construction values.
# ----------------------------------------------------------------------
def test_vllm_flash_attn_nhd_spec():
    kv = _vllm_fa_nhd(nl=4, nb=32, bs=16, nh=8, hs=64)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS)
    assert spec.num_layers() == 4
    assert spec.num_blocks() == 32
    assert spec.block_size() == 16
    assert spec.num_heads() == 8
    assert spec.head_size() == 64
    assert spec.page_buffer_size() == 32 * 16
    assert spec.hidden_dim() == 8 * 64
    assert spec.tokens_per_layer() == 32 * 16
    assert spec.elements_per_layer() == 2 * 32 * 16 * 8 * 64
    assert not spec.is_mla
    assert not spec.is_hnd
    assert not spec.is_cross_layer


def test_vllm_flash_attn_hnd_spec():
    kv = _vllm_fa_hnd(nl=4, nb=32, bs=16, nh=8, hs=64)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS)
    assert spec.num_blocks() == 32
    assert spec.block_size() == 16
    assert spec.num_heads() == 8
    assert spec.head_size() == 64
    assert spec.is_hnd


def test_vllm_flash_infer_nhd_spec():
    kv = _vllm_fi_nhd(nl=2, nb=64, bs=8, nh=4, hs=128)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS)
    assert spec.num_blocks() == 64
    assert spec.block_size() == 8
    assert spec.num_heads() == 4
    assert spec.head_size() == 128


def test_vllm_flash_infer_hnd_spec():
    kv = _vllm_fi_hnd(nl=2, nb=64, bs=8, nh=4, hs=128)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS)
    assert spec.num_blocks() == 64
    assert spec.block_size() == 8
    assert spec.num_heads() == 4
    assert spec.head_size() == 128
    assert spec.is_hnd


def test_vllm_mla_spec():
    kv = _vllm_mla(nl=3, nb=32, bs=16, hs=512)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.NL_X_NB_BS_HS)
    assert spec.is_mla
    assert spec.is_block_axis_dim0
    assert spec.num_layers() == 3
    assert spec.num_blocks() == 32
    assert spec.block_size() == 16
    assert spec.num_heads() == 1  # absorbed into hidden dim
    assert spec.head_size() == 512
    assert spec.hidden_dim() == 512
    assert spec.elements_per_layer() == 32 * 16 * 512


def test_sglang_mla_spec_no_separate_block_dims():
    kv = _sglang_mla(nl=2, pbs=512, hs=128)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS)
    assert spec.is_mla
    assert not spec.has_separate_block_dims
    assert spec.num_layers() == 2
    assert spec.num_heads() == 1
    assert spec.head_size() == 128
    assert spec.page_buffer_size() == 512
    # Fused-PBS formats refuse num_blocks / block_size separately.
    with pytest.raises(ValueError):
        spec.num_blocks()
    with pytest.raises(ValueError):
        spec.block_size()


def test_sglang_mha_spec():
    kv = _sglang_mha(nl=4, pbs=512, nh=8, hs=64)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS)
    assert not spec.has_separate_block_dims
    assert spec.num_layers() == 4
    assert spec.num_heads() == 8
    assert spec.head_size() == 64
    assert spec.page_buffer_size() == 512
    # MHA stores K and V separately, so elements per layer doubles.
    assert spec.elements_per_layer() == 512 * 8 * 64 * 2


def test_cross_layer_nhd_spec():
    kv = _cross_layer_nhd(nb=32, nl=80, bs=16, nh=8, hs=64)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS)
    assert spec.is_cross_layer
    assert spec.num_layers() == 80
    assert spec.num_blocks() == 32
    assert spec.block_size() == 16
    assert spec.num_heads() == 8
    assert spec.head_size() == 64


def test_cross_layer_hnd_spec():
    kv = _cross_layer_hnd(nb=32, nl=80, bs=16, nh=8, hs=64)
    spec = get_spec(kv, lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS)
    assert spec.is_cross_layer
    assert spec.is_hnd
    assert spec.num_layers() == 80
    assert spec.num_blocks() == 32
    assert spec.block_size() == 16
    assert spec.num_heads() == 8
    assert spec.head_size() == 64


# ----------------------------------------------------------------------
# Detection — round-trip every supported (engine, hint) combo.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kv,engine,hints,expected_format",
    [
        (
            _vllm_fa_nhd(),
            EngineType.VLLM,
            {"kv_layout": "NHD"},
            lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        ),
        (
            _vllm_fi_nhd(),
            EngineType.VLLM,
            {"kv_layout": "NHD"},
            lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        ),
        (
            _vllm_fa_hnd(),
            EngineType.VLLM,
            {"kv_layout": "HND"},
            lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        ),
        (
            _vllm_fi_hnd(),
            EngineType.VLLM,
            {"kv_layout": "HND"},
            lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
        ),
        (
            _vllm_mla(),
            EngineType.VLLM,
            {"kv_layout": "NHD"},
            lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
        ),
        (
            _cross_layer_nhd(),
            EngineType.VLLM,
            {"kv_layout": "NHD"},
            lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
        ),
        (
            _sglang_mla(),
            EngineType.SGLANG,
            {},
            lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
        ),
        (
            _sglang_mha(),
            EngineType.SGLANG,
            {},
            lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS,
        ),
        (
            _cross_layer_hnd(),
            EngineType.TRTLLM,
            {},
            lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
        ),
    ],
)
def test_detect_format_round_trip(kv, engine, hints, expected_format, monkeypatch):
    # vLLM detector force-overrides kv_layout to HND when running on CPU
    # (a legitimate prod safeguard). Pin the device flag to "cuda" here
    # so NHD parametrised cases still take the NHD detection path.
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type",
        "cuda",
    )
    fmt, normalized = detect_format(kv, engine, hints)
    assert fmt == expected_format
    # detect_format does not change a contiguous canonical layout.
    assert normalized is kv or isinstance(normalized, type(kv))


def test_trtllm_4d_pool_normalize_to_6d():
    """TRT-LLM hands us a 4-D pool tensor; normalize must reshape it."""
    nb, nl, kv_dim, nh, bs, hd = 32, 4, 2, 8, 16, 64
    raw = torch.empty(nb, nl, kv_dim, nh * bs * hd, dtype=torch.bfloat16)
    fmt, normalized = detect_format(
        raw,
        EngineType.TRTLLM,
        {"num_kv_heads": nh, "tokens_per_block": bs, "head_dim": hd},
    )
    assert fmt == lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS
    assert isinstance(normalized, torch.Tensor)
    assert tuple(normalized.shape) == (nb, nl, kv_dim, nh, bs, hd)


# ----------------------------------------------------------------------
# Facade backward compatibility — top-level helpers must keep returning
# the same values as before.
# ----------------------------------------------------------------------
def test_facade_metadata_helpers_match_spec():
    fmt = lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    cls = get_spec_class(fmt)
    assert cls is not None
    assert facade.is_mla(fmt) == cls.is_mla
    assert facade.is_cross_layer_format(fmt) == cls.is_cross_layer
    assert facade.is_hnd(fmt) == cls.is_hnd
    assert facade.get_gpu_kv_shape_description(fmt) == cls.shape_desc
    assert facade.get_attention_backend(fmt) == cls.backend_label


def test_facade_per_layer_getters_match_spec():
    kv = _vllm_fa_nhd(nl=4, nb=32, bs=16, nh=8, hs=64)
    fmt = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    spec = get_spec(kv, fmt)

    assert facade.get_num_layers(kv, fmt) == spec.num_layers()
    assert facade.get_num_blocks(kv, fmt) == spec.num_blocks()
    assert facade.get_block_size(kv, fmt) == spec.block_size()
    assert facade.get_num_heads(kv, fmt) == spec.num_heads()
    assert facade.get_head_size(kv, fmt) == spec.head_size()
    assert facade.get_page_buffer_size(kv, fmt) == spec.page_buffer_size()
    assert facade.get_hidden_dim_size(kv, fmt) == spec.hidden_dim()
    assert facade.get_tokens_per_layer(kv, fmt) == spec.tokens_per_layer()
    assert facade.get_elements_per_layer(kv, fmt) == spec.elements_per_layer()
    assert facade.get_dtype(kv, fmt) == spec.dtype()
    assert facade.get_group_data_ptrs(kv, fmt, [0, 2]) == spec.data_ptrs([0, 2])


def test_facade_concrete_shape_matches_spec():
    kv = _vllm_fa_nhd(nl=4, nb=32, bs=16, nh=8, hs=64)
    fmt = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    spec = get_spec(kv, fmt)
    assert facade.get_concrete_gpu_kv_shape(kv, fmt) == spec.concrete_shape_str()


# ----------------------------------------------------------------------
# Extension demo — register a fake spec from a test, ensuring third
# parties can extend the registry without modifying core code.
# ----------------------------------------------------------------------
def test_third_party_can_register_a_spec():
    """Demonstrate that the registry is extensible from outside.

    Subclassing :class:`KVFormatSpec` (or any family base) auto-registers
    via ``__init_subclass__``. We tear down with ``unregister_spec`` so
    other tests are unaffected.
    """
    # First Party
    from lmcache.v1.gpu_connector.kv_format import unregister_spec

    fake_gpu_fmt = object()  # standalone sentinel for the enum slot.

    class _FakeSpec(KVFormatSpec):
        abstract = False
        engine = "fake"
        gpu_kv_format = fake_gpu_fmt  # type: ignore[assignment]
        shape_desc = "FAKE"
        backend_label = "fake-backend"

        def num_layers(self):
            return 0

        def page_buffer_size(self):
            return 0

        def num_heads(self, layer_idx=0):
            return 0

        def head_size(self, layer_idx=0):
            return 0

        def data_ptrs(self, layer_indices):
            return []

        def layout_probe_tensor(self, layer_idx=0):
            return torch.empty(0)

    try:
        # format_id should default to class name minus trailing 'Spec'.
        assert _FakeSpec.format_id == "_Fake"
        assert get_spec_class(fake_gpu_fmt) is _FakeSpec  # type: ignore[arg-type]
    finally:
        unregister_spec(_FakeSpec.format_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
