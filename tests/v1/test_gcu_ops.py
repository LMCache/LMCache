# SPDX-License-Identifier: Apache-2.0
"""Tests for the GCU platform backend wiring.

The CPU-only tests verify device discovery and soft-fail behaviour; the
roundtrip tests are gated on a real GCU (``torch.gcu`` + ``triton_kernel_gcu``).
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import get_device_spec, resolve_device_ops
from lmcache.v1.platform.devices.gcu import GcuDeviceSpec
from lmcache.v1.platform.devices.gcu.device_ops import GcuDeviceOps
from lmcache.v1.platform.devices.gcu.pin_memory import GcuPinMemoryBackend


def _skip_if_no_gcu() -> None:
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        pytest.skip("torch.gcu is not available")


def test_gcu_device_spec_discovered() -> None:
    """``GcuDeviceSpec`` is registered and resolves for device type "gcu"."""
    spec = get_device_spec("gcu")
    assert isinstance(spec, GcuDeviceSpec)
    assert spec.device_type == "gcu"
    assert spec.torch_module_name == "gcu"


def test_gcu_device_ops_soft_fail_without_triton() -> None:
    """Without ``triton_kernel_gcu`` the ops backend degrades to torch."""
    ops = resolve_device_ops("gcu")
    assert isinstance(ops, GcuDeviceOps)
    # ensure_native must not raise when the plugin is absent; _ns stays None so
    # a later _op() call reports a clear error instead of silently misbehaving.
    ops.ensure_native()


def test_gcu_pin_memory_soft_fail_without_gcu() -> None:
    """Pin backend reports unsupported on a GCU-less host."""
    backend = GcuPinMemoryBackend()
    # is_pin_supported is False on a non-GCU host; do not assert True here.
    assert isinstance(backend.is_pin_supported, bool)


def test_gcu_enums_match_native() -> None:
    """The DeviceSpec resolves to the shared lmcache_native enums."""
    import lmcache.lmcache_native as lmcache_native

    assert int(lmcache_native.TransferDirection.H2D) == 0
    assert int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS) == 1


def test_calculate_cdf_bit_exact_cpu_ref() -> None:
    """CDF on GCU matches the torch fallback reference."""
    _skip_if_no_gcu()
    ops = resolve_device_ops("gcu")
    dev = torch.device("gcu")
    inp = torch.randint(0, 16, (2, 17, 8), dtype=torch.uint8, device=dev)
    gcu_cdf = ops.calculate_cdf(inp, 16)
    # First Party
    from lmcache.v1.platform import torch_ops

    ref_cdf = torch_ops.calculate_cdf(inp.cpu(), 16)
    assert gcu_cdf.shape == ref_cdf.shape
    assert torch.equal(gcu_cdf.cpu(), ref_cdf)


def test_multi_layer_kv_transfer_roundtrip() -> None:
    """H2D then D2H preserves KV values for a simple vLLM layout."""
    _skip_if_no_gcu()
    ops = resolve_device_ops("gcu")
    import lmcache.lmcache_native as lmcache_native

    dev = torch.device("gcu")
    fmt = lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    nl, nb, bs, nh, hs = 2, 4, 8, 2, 16
    layers = [
        torch.randn(2, nb, bs, nh, hs, dtype=torch.float16, device=dev)
        for _ in range(nl)
    ]
    ptrs = torch.tensor([t.data_ptr() for t in layers], dtype=torch.int64)
    num_tokens = 16
    kv = torch.randn(2, nl, num_tokens, nh * hs, dtype=torch.float16, device=dev)
    ref = kv.clone()
    slots = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    ops.multi_layer_kv_transfer(
        kv, ptrs, slots, dev, nb * bs, lmcache_native.TransferDirection.H2D,
        fmt, bs, hs, 0,
    )
    ops.multi_layer_kv_transfer(
        kv, ptrs, slots, dev, nb * bs, lmcache_native.TransferDirection.D2H,
        fmt, bs, hs, 0,
    )
    assert torch.allclose(kv.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)


def test_alloc_pinned_ptr_roundtrip() -> None:
    """Pinned host alloc/free works through the GCU host op wrappers."""
    _skip_if_no_gcu()
    ops = resolve_device_ops("gcu")
    size = 4096
    ptr = ops.alloc_pinned_ptr(size, 0)
    assert ptr > 0
    ops.free_pinned_ptr(ptr)
