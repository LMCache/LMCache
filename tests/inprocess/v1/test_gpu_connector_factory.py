# SPDX-License-Identifier: Apache-2.0
"""Accelerator dispatch and feature guards for the in-process connector factory."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.gpu_connector import CreateGPUConnector
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.inprocess.v1.gpu_connector as gpu_connector_module

pytestmark = pytest.mark.musa


def _make_metadata() -> LMCacheMetadata:
    """Minimal metadata accepted by ``CreateGPUConnector``."""
    return LMCacheMetadata(
        model_name="musa_support_test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, 16, 8, 64),
    )


def _make_config(**overrides: Any) -> LMCacheEngineConfig:
    """Default config plus the requested overrides."""
    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class _FakeTorchDev:
    """Stand-in for ``torch.musa`` / ``torch.xpu`` / ``torch.cuda``."""

    def __init__(self, device_count: int = 1) -> None:
        self._device_count = device_count

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self._device_count

    def current_device(self) -> int:
        return 0

    def set_device(self, _idx: int) -> None:
        return


def _patch_device(monkeypatch: pytest.MonkeyPatch, device_type: str) -> None:
    """Pretend the current accelerator is ``device_type`` in the factory."""
    monkeypatch.setattr(gpu_connector_module, "torch_device_type", device_type)
    monkeypatch.setattr(gpu_connector_module, "torch_dev", _FakeTorchDev())


def test_create_gpu_connector_blending_rejected_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enable_blending`` must fail fast on MUSA."""
    _patch_device(monkeypatch, "musa")
    config = _make_config(enable_blending=True, use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="enable_blending"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


@pytest.mark.parametrize("device_type", ["musa", "hpu"])
def test_create_gpu_connector_blending_rejected_on_unsupported_devices(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    """The blending guard rejects devices without a blending connector."""
    _patch_device(monkeypatch, device_type)
    config = _make_config(enable_blending=True, use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="enable_blending"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


def test_create_gpu_connector_v3_rejected_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_gpu_connector_v3`` must fail fast on MUSA."""
    _patch_device(monkeypatch, "musa")
    config = _make_config(use_gpu_connector_v3=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="use_gpu_connector_v3"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


def test_create_gpu_connector_layerwise_rejected_on_hpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HPU ships only ``VLLMPagedMemHPUConnectorV2`` (no layerwise variant).

    Previously, ``use_layerwise=True`` on HPU silently fell through into
    ``VLLMPagedMemLayerwiseGPUConnector`` — the CUDA layerwise connector —
    which then crashed on HPU tensors when constructing
    ``torch.cuda.Stream()``. The guard must reject this combination with a
    clear error before any device-specific construction.
    """
    _patch_device(monkeypatch, "hpu")
    config = _make_config(use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError) as exc:
        CreateGPUConnector(config, metadata, EngineType.VLLM)
    message = str(exc.value)
    assert "use_layerwise" in message
    assert "hpu" in message.lower()


def test_create_gpu_connector_musa_dispatches_to_musa_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``torch_device_type == 'musa'`` selects a MUSA connector class.

    We do not need real MUSA hardware: the factory's ``torch.device`` call
    is patched (PyTorch builds without ``torch_musa`` reject the 'musa'
    string), and ``from_metadata`` is replaced on the MUSA connectors so
    the assertion checks only the dispatch.
    """
    _patch_device(monkeypatch, "musa")
    # First Party
    from lmcache.inprocess.v1.gpu_connector import musa_connectors as musa_mod

    monkeypatch.setattr(
        gpu_connector_module.torch, "device", lambda *_a, **_kw: "musa:0"
    )

    sentinel_v2 = object()
    sentinel_layer = object()
    monkeypatch.setattr(
        musa_mod.VLLMPagedMemMUSAConnectorV2,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: sentinel_v2),
    )
    monkeypatch.setattr(
        musa_mod.VLLMPagedMemLayerwiseMUSAConnector,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: sentinel_layer),
    )

    metadata = _make_metadata()
    assert (
        CreateGPUConnector(_make_config(use_layerwise=False), metadata, EngineType.VLLM)
        is sentinel_v2
    )
    assert (
        CreateGPUConnector(_make_config(use_layerwise=True), metadata, EngineType.VLLM)
        is sentinel_layer
    )
