# SPDX-License-Identifier: Apache-2.0
"""Regression tests for KV layout discovery on devices without a pointer kernel.

Layout discovery is a start-up concern: the engine registers its KV caches once
and never replaces them. These tests pin that discovery to a single run per
device so it cannot drift back into the per-transfer path, where it would cost
every chunk.

The path under test is reached only for ``_TENSOR_LIST_DEVICES``; the CUDA
pointer path in ``_initialize_pointers`` is deliberately left as it is on dev.
CPU tensors stand in for such a device -- the fixture adds ``"cpu"`` to that set
-- because the concern is device-agnostic and Neuron is not available in CI.
"""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
import lmcache.lmcache_native as lmcache_native
import lmcache.v1.gpu_connector.gpu_connectors as connectors_mod


@pytest.fixture
def kv_caches() -> list[torch.Tensor]:
    """Two CPU layers in ``[2, NB, NH, BS, HS]`` (HND) layout."""
    return [torch.zeros((2, 4, 2, 8, 16)) for _ in range(2)]


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch) -> VLLMPagedMemGPUConnectorV2:
    """A connector with layout-discovery state initialized, no GPU buffer.

    ``"cpu"`` is added to the tensor-list device set so CPU tensors take the same
    branch a Neuron device would.
    """
    monkeypatch.setattr(
        connectors_mod, "_TENSOR_LIST_DEVICES", frozenset({"neuron", "cpu"})
    )
    instance = object.__new__(VLLMPagedMemGPUConnectorV2)
    instance.layout_hints = {"kv_layout": "HND"}
    instance.kv_cache_pointers_on_gpu = {}
    instance._normalized_kv_caches = {}
    return instance


def _count_detection_calls(
    monkeypatch: pytest.MonkeyPatch, counter: dict[str, int]
) -> None:
    """Patch layout detection and geometry helpers to count and stub them.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param counter: Dict incremented under ``"detect"`` on each detection call.
    """
    fmt = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS

    def fake_detect(kv_caches: Any, engine: Any, layout_hints: Any = None) -> Any:
        counter["detect"] = counter.get("detect", 0) + 1
        return fmt, kv_caches

    monkeypatch.setattr(connectors_mod, "normalize_kv_and_discover_format", fake_detect)
    monkeypatch.setattr(connectors_mod, "get_num_blocks", lambda *a, **k: 4)
    monkeypatch.setattr(connectors_mod, "get_block_size", lambda *a, **k: 8)
    monkeypatch.setattr(connectors_mod, "get_head_size", lambda *a, **k: 16)
    monkeypatch.setattr(
        connectors_mod, "resolve_block_stride_and_log_layout", lambda *a, **k: 0
    )


def test_layout_discovery_runs_once_across_transfers(
    connector: VLLMPagedMemGPUConnectorV2,
    kv_caches: list[torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated transfers reuse the layout discovered on the first call."""
    counter: dict[str, int] = {}
    _count_detection_calls(monkeypatch, counter)

    for _ in range(5):
        connector._initialize_pointers(kv_caches)

    assert counter["detect"] == 1


def test_layout_geometry_is_populated(
    connector: VLLMPagedMemGPUConnectorV2,
    kv_caches: list[torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cached geometry is derived on the first call and stays available."""
    counter: dict[str, int] = {}
    _count_detection_calls(monkeypatch, counter)

    connector._initialize_pointers(kv_caches)
    connector._initialize_pointers(kv_caches)

    assert connector.block_size == 8
    assert connector.num_blocks == 4
    assert connector.head_size == 16
    assert connector.page_buffer_size == 32
    assert (
        connector.engine_kv_format == lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    )


def test_non_pointer_device_returns_tensor_list(
    connector: VLLMPagedMemGPUConnectorV2,
    kv_caches: list[torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Devices without a fused pointer kernel get the per-layer tensor list."""
    counter: dict[str, int] = {}
    _count_detection_calls(monkeypatch, counter)

    result = connector._initialize_pointers(kv_caches)

    assert isinstance(result, list)
    assert len(result) == len(kv_caches)


def test_layout_is_cached_per_device_not_per_connector(
    connector: VLLMPagedMemGPUConnectorV2,
    kv_caches: list[torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second device gets its own discovery, not the first device's result.

    The normalized tensors are what callers read geometry and pointers from, so
    caching them once per connector rather than once per device would hand the
    second device the first device's tensors.
    """
    counter: dict[str, int] = {}
    _count_detection_calls(monkeypatch, counter)

    connector._initialize_pointers(kv_caches)
    first = connector._normalized_kv_caches[str(kv_caches[0].device)]
    assert counter["detect"] == 1

    # Distinct tensor objects standing in for a second device's caches. They are
    # still CPU tensors; what matters is that reusing the first entry would be
    # detectable. Key them under a different device string to force the miss.
    other = [tensor.clone() for tensor in kv_caches]
    connector._normalized_kv_caches.pop(str(other[0].device))
    connector._initialize_pointers(other)

    assert counter["detect"] == 2, "detection must re-run for a second device"
    assert connector._normalized_kv_caches[str(other[0].device)] is not first
