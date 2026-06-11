# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable tests for MUSA detection and vLLM connector dispatch."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import CreateGPUConnector
from lmcache.v1.metadata import LMCacheMetadata
import lmcache as lmc
import lmcache.v1.gpu_connector as gpu_connector_module


def _make_metadata() -> LMCacheMetadata:
    """Create minimal metadata accepted by ``CreateGPUConnector``.

    Returns:
        Metadata for a small synthetic KV cache.
    """
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
    """Create a default engine config with field overrides.

    Args:
        **overrides: Config attributes to set after construction.

    Returns:
        An LMCache engine config.
    """
    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class _FakeTorchDev:
    """Stand-in for ``torch.musa`` / ``torch.xpu`` / ``torch.cuda``."""

    def __init__(self, device_count: int = 1) -> None:
        """Initialize the fake accelerator module.

        Args:
            device_count: Number of devices to report.
        """
        self._device_count = device_count

    def is_available(self) -> bool:
        """Return whether the fake device is available.

        Returns:
            Always ``True``.
        """
        return True

    def device_count(self) -> int:
        """Return the fake device count.

        Returns:
            The configured device count.
        """
        return self._device_count

    def current_device(self) -> int:
        """Return the current fake device index.

        Returns:
            Always ``0``.
        """
        return 0

    def set_device(self, _idx: int) -> None:
        """Set the fake current device.

        Args:
            _idx: Ignored device index.
        """
        return


def _patch_device(monkeypatch: pytest.MonkeyPatch, device_type: str) -> None:
    """Pretend the connector factory is running on ``device_type``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        device_type: Accelerator name to expose to the factory.
    """
    monkeypatch.setattr(gpu_connector_module, "torch_device_type", device_type)
    monkeypatch.setattr(gpu_connector_module, "torch_dev", _FakeTorchDev())


class _StubTorch:
    """Minimal stand-in for ``torch`` exposing what ``_detect_device`` reads."""

    def __init__(
        self,
        *,
        has_musa: bool = False,
        has_xpu: bool = False,
        has_hpu: bool = False,
        musa_available: bool = False,
        xpu_available: bool = False,
        hpu_available: bool = False,
    ) -> None:
        """Initialize the torch stub.

        Args:
            has_musa: Whether to expose ``torch.musa``.
            has_xpu: Whether to expose ``torch.xpu``.
            has_hpu: Whether to expose ``torch.hpu``.
            musa_available: Return value for ``torch.musa.is_available``.
            xpu_available: Return value for ``torch.xpu.is_available``.
            hpu_available: Return value for ``torch.hpu.is_available``.
        """
        self.cuda = SimpleNamespace(is_available=lambda: True)
        if has_musa:
            self.musa = SimpleNamespace(is_available=lambda: musa_available)
        if has_xpu:
            self.xpu = SimpleNamespace(is_available=lambda: xpu_available)
        if has_hpu:
            self.hpu = SimpleNamespace(is_available=lambda: hpu_available)


def _detect_with_stub(stub: _StubTorch) -> tuple[Any, str]:
    """Run ``_detect_device`` with ``torch`` swapped for a stub.

    Args:
        stub: Stub torch module.

    Returns:
        The detected torch device module and device type.
    """
    with patch.dict("sys.modules", {"torch": stub}):
        return lmc._detect_device()


def test_detect_device_prefers_musa_when_available() -> None:
    """``_detect_device`` returns MUSA when ``torch.musa`` is available."""
    stub = _StubTorch(
        has_musa=True,
        has_xpu=True,
        has_hpu=True,
        musa_available=True,
        xpu_available=True,
        hpu_available=True,
    )
    dev, name = _detect_with_stub(stub)
    assert name == "musa"
    assert dev is stub.musa


def test_detect_device_falls_back_past_unavailable_musa() -> None:
    """``_detect_device`` falls through MUSA when it is unavailable."""
    stub = _StubTorch(
        has_musa=True,
        has_xpu=True,
        musa_available=False,
        xpu_available=True,
    )
    _, name = _detect_with_stub(stub)
    assert name == "xpu"


def test_detect_device_cuda_fallback_when_no_alt_accelerator() -> None:
    """``_detect_device`` returns CUDA when no alternate accelerator exists."""
    stub = _StubTorch()
    dev, name = _detect_with_stub(stub)
    assert name == "cuda"
    assert dev is stub.cuda


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"enable_blending": True}, "enable_blending"),
        ({"use_gpu_connector_v3": True}, "use_gpu_connector_v3"),
        ({"use_layerwise": True}, "use_layerwise"),
    ],
)
def test_create_gpu_connector_rejects_unsupported_musa_vllm_features(
    monkeypatch: pytest.MonkeyPatch,
    config_kwargs: dict[str, bool],
    message: str,
) -> None:
    """MUSA vLLM dispatch fails fast for unsupported feature flags.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        config_kwargs: Config flag override under test.
        message: Expected error message fragment.
    """
    _patch_device(monkeypatch, "musa")
    with pytest.raises(ValueError, match=message) as exc_info:
        CreateGPUConnector(
            _make_config(**config_kwargs), _make_metadata(), EngineType.VLLM
        )
    assert "this PR" not in str(exc_info.value)


def test_create_gpu_connector_rejects_sglang_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUSA dispatch reports SGLang as unsupported without PR-only wording."""
    _patch_device(monkeypatch, "musa")
    with pytest.raises(ValueError, match="SGLang on MUSA") as exc_info:
        CreateGPUConnector(_make_config(), _make_metadata(), EngineType.SGLANG)
    assert "this PR" not in str(exc_info.value)


def test_create_gpu_connector_musa_dispatches_to_vllm_musa_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUSA + vLLM + non-layerwise selects ``VLLMPagedMemMUSAConnectorV2``."""
    _patch_device(monkeypatch, "musa")

    # First Party
    from lmcache.v1.gpu_connector import musa_connectors as musa_mod

    monkeypatch.setattr(
        gpu_connector_module.torch, "device", lambda *_a, **_kw: "musa:0"
    )

    sentinel_v2 = object()
    monkeypatch.setattr(
        musa_mod.VLLMPagedMemMUSAConnectorV2,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: sentinel_v2),
    )

    connector = CreateGPUConnector(
        _make_config(use_layerwise=False), _make_metadata(), EngineType.VLLM
    )
    assert connector is sentinel_v2
