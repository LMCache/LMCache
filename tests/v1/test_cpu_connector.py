# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure-PyTorch CPU KV connector (``VLLMCPUConnector``).

Covers, without any GPU / c_ops kernels:
1. round-trip store/load correctness on CPU,
2. both HND and NHD physical layouts,
3. negative sentinel slots (``-1``) being skipped on load and zeroed on store.
"""

# Standard
from collections.abc import Callable
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.cpu_connector import VLLMCPUConnector
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat

NUM_LAYERS = 2
NB = 4  # num blocks
BS = 8  # block size
NH = 2  # num kv heads
HS = 4  # head size
HIDDEN = NH * HS


def _make_kvcaches(
    layout: str,
    fill: Callable[[tuple[int, ...], int], torch.Tensor],
    mode: str = "5d",
) -> list[torch.Tensor]:
    """Per-layer paged caches in the requested physical layout.

    mode="5d": split K/V on a leading dim  -> [2, NB, NH, BS, HS] (HND) /
               [2, NB, BS, NH, HS] (NHD).
    mode="4d": vLLM CPU fused K/V on the last dim -> [NB, NH, BS, 2*HS] (HND) /
               [NB, BS, NH, 2*HS] (NHD).
    """
    caches = []
    for layer in range(NUM_LAYERS):
        if mode == "5d":
            shape = (2, NB, NH, BS, HS) if layout == "HND" else (2, NB, BS, NH, HS)
        else:  # 4d fused
            shape = (NB, NH, BS, 2 * HS) if layout == "HND" else (NB, BS, NH, 2 * HS)
        caches.append(fill(shape, layer))
    return caches


def _make_memory_obj(num_tokens: int) -> SimpleNamespace:
    """Duck-typed KV_2LTD MemoryObj: [2, num_layers, T, hidden]."""
    tensor = torch.zeros(2, NUM_LAYERS, num_tokens, HIDDEN)
    return SimpleNamespace(
        tensor=tensor, metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD)
    )


def test_is_gpu_connector_interface_subclass():
    assert issubclass(VLLMCPUConnector, GPUConnectorInterface)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("mode", ["5d", "4d"])
def test_round_trip(layout, mode):
    torch.manual_seed(0)
    conn = VLLMCPUConnector(
        hidden_dim_size=HIDDEN,
        num_layers=NUM_LAYERS,
        kv_layout=layout,
        head_size=HS,
    )
    src = _make_kvcaches(layout, lambda shape, layer: torch.randn(*shape), mode)
    dst = _make_kvcaches(layout, lambda shape, layer: torch.zeros(*shape), mode)

    tokens = NB * BS
    slots = torch.arange(tokens, dtype=torch.int64)
    mo = _make_memory_obj(tokens)

    conn.from_gpu(mo, 0, tokens, kvcaches=src, slot_mapping=slots)
    conn.to_gpu(mo, 0, tokens, kvcaches=dst, slot_mapping=slots)

    # Slots cover every position, so a store-then-load round trip must
    # reconstruct each layer's paged cache exactly.
    for layer in range(NUM_LAYERS):
        assert torch.allclose(dst[layer], src[layer])


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("mode", ["5d", "4d"])
def test_sentinel_slots_skipped(layout, mode):
    torch.manual_seed(1)
    conn = VLLMCPUConnector(
        hidden_dim_size=HIDDEN,
        num_layers=NUM_LAYERS,
        kv_layout=layout,
        head_size=HS,
    )
    # Strictly-positive fill so an uncovered (zeroed) position always differs
    # from src -- makes the exact mismatch count below deterministic.
    src = _make_kvcaches(layout, lambda shape, layer: torch.rand(*shape) + 1.0, mode)
    dst = _make_kvcaches(layout, lambda shape, layer: torch.zeros(*shape), mode)

    # Cover every position, then turn token index 2 into a PAD_SLOT sentinel
    # (-1). Its slot value (2) then maps to no token, so that one position must
    # stay unwritten while every other position round-trips exactly.
    tokens = NB * BS
    slots = torch.arange(tokens, dtype=torch.int64)
    sentinel_token = 2
    slots[sentinel_token] = -1
    mo = _make_memory_obj(tokens)

    conn.from_gpu(mo, 0, tokens, kvcaches=src, slot_mapping=slots)
    # The stored row for the sentinel token must be zeroed (not garbage).
    assert torch.count_nonzero(mo.tensor[0, :, sentinel_token, :]) == 0
    assert torch.count_nonzero(mo.tensor[1, :, sentinel_token, :]) == 0

    conn.to_gpu(mo, 0, tokens, kvcaches=dst, slot_mapping=slots)
    # dst must equal src everywhere except the single uncovered slot, which
    # holds one token's worth of K and V scalars (2 * NH * HS).
    for layer in range(NUM_LAYERS):
        assert int((dst[layer] != src[layer]).sum()) == 2 * NH * HS


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("mode", ["5d", "4d"])
def test_prefix_skip_not_written(layout, mode):
    # to_gpu must skip the first ``vllm_cached_tokens`` tokens (already present
    # in vLLM's own paged cache), leaving those positions unwritten.
    torch.manual_seed(2)
    conn = VLLMCPUConnector(
        hidden_dim_size=HIDDEN,
        num_layers=NUM_LAYERS,
        kv_layout=layout,
        head_size=HS,
    )
    # Strictly-positive fill so a skipped (zeroed) position always differs from
    # src -- makes the exact mismatch count below deterministic.
    src = _make_kvcaches(layout, lambda shape, layer: torch.rand(*shape) + 1.0, mode)
    dst = _make_kvcaches(layout, lambda shape, layer: torch.zeros(*shape), mode)

    tokens = NB * BS
    slots = torch.arange(tokens, dtype=torch.int64)
    skip = 5
    mo = _make_memory_obj(tokens)

    conn.from_gpu(mo, 0, tokens, kvcaches=src, slot_mapping=slots)
    conn.to_gpu(
        mo, 0, tokens, kvcaches=dst, slot_mapping=slots, vllm_cached_tokens=skip
    )
    # The first ``skip`` slot positions stay zero; everything else round-trips.
    # Each skipped token is one K + one V row (2 * NH * HS scalars) per layer.
    for layer in range(NUM_LAYERS):
        assert int((dst[layer] != src[layer]).sum()) == skip * 2 * NH * HS


def test_fused_layout_infers_head_size_without_head_size_arg():
    # head_size omitted: the 4-D fused path must infer HS from the cache's last
    # dim rather than depend on head_size (regression for the NH>1 fallback bug).
    conn = VLLMCPUConnector(
        hidden_dim_size=HIDDEN, num_layers=NUM_LAYERS, kv_layout="HND"
    )
    src = _make_kvcaches("HND", lambda shape, layer: torch.randn(*shape), "4d")
    dst = _make_kvcaches("HND", lambda shape, layer: torch.zeros(*shape), "4d")
    tokens = NB * BS
    slots = torch.arange(tokens, dtype=torch.int64)
    mo = _make_memory_obj(tokens)
    conn.from_gpu(mo, 0, tokens, kvcaches=src, slot_mapping=slots)
    conn.to_gpu(mo, 0, tokens, kvcaches=dst, slot_mapping=slots)
    for layer in range(NUM_LAYERS):
        assert torch.allclose(dst[layer], src[layer])


def test_mla_not_supported():
    with pytest.raises(NotImplementedError):
        VLLMCPUConnector(hidden_dim_size=HIDDEN, num_layers=NUM_LAYERS, use_mla=True)


def test_invalid_kv_layout_raises():
    with pytest.raises(ValueError):
        VLLMCPUConnector(
            hidden_dim_size=HIDDEN, num_layers=NUM_LAYERS, kv_layout="BOGUS"
        )


def test_connector_package_imports_on_cpu():
    # Regression: lmcache.c_ops is a device shim that resolves to the pure-torch
    # CpuDeviceOps baseline on CPU, so importing the connector package (and thus
    # reaching the CPU dispatch branch) must not fail on a CPU-only box.
    # First Party
    import lmcache.v1.gpu_connector as pkg

    assert hasattr(pkg, "CreateGPUConnector")


def test_create_gpu_connector_returns_cpu_connector_on_cpu():
    # Regression for the review claim that the CPU connector path is unreachable:
    # on a CPU build the vLLM factory must dispatch to VLLMCPUConnector.
    # First Party
    from lmcache import torch_device_type
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector import CreateGPUConnector

    if torch_device_type != "cpu":
        pytest.skip("CPU dispatch is only exercised on a CPU build")

    config = SimpleNamespace(
        enable_pd=False,
        enable_blending=False,
        use_gpu_connector_v3=False,
        use_layerwise=False,
    )
    metadata = SimpleNamespace(
        local_worker_id=0,
        use_mla=False,
        kv_shape=(NUM_LAYERS, 2, BS, NH, HS),
    )
    conn = CreateGPUConnector(config, metadata, EngineType.VLLM)
    assert isinstance(conn, VLLMCPUConnector)


def test_cpu_ignores_nhd_layout_hint(monkeypatch):
    # vLLM's CPU attention backend misreports its layout (it is HND); the
    # KV-format detector forces HND on CPU, so from_metadata must ignore an
    # (incorrect) NHD hint rather than interpret axes wrongly.
    monkeypatch.delenv("LMCACHE_CPU_KV_LAYOUT", raising=False)
    metadata = SimpleNamespace(use_mla=False, kv_shape=(NUM_LAYERS, 2, BS, NH, HS))
    conn = VLLMCPUConnector.from_metadata(metadata, layout_hints={"kv_layout": "NHD"})
    assert conn.kv_layout == "HND"


def test_cpu_rejects_use_layerwise():
    # The CPU connector is not layerwise; the factory must fail fast instead of
    # returning a connector that breaks the layerwise retrieve path at runtime.
    # First Party
    from lmcache import torch_device_type
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector import CreateGPUConnector

    if torch_device_type != "cpu":
        pytest.skip("CPU dispatch is only exercised on a CPU build")

    config = SimpleNamespace(
        enable_pd=False,
        enable_blending=False,
        use_gpu_connector_v3=False,
        use_layerwise=True,
    )
    metadata = SimpleNamespace(
        local_worker_id=0,
        use_mla=False,
        kv_shape=(NUM_LAYERS, 2, BS, NH, HS),
    )
    with pytest.raises(ValueError):
        CreateGPUConnector(config, metadata, EngineType.VLLM)
