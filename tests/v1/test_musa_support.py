# SPDX-License-Identifier: Apache-2.0
"""
MUSA support unit tests that do not require MUSA hardware.

These tests cover the design contract documented in
``docs/source/developer_guide/musa_support_design.rst``:

- Device detection via the registry-driven
  :func:`lmcache.v1.platform._detect_device`.

The tests provide a stub accelerator registry so they run on any platform.
"""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import importlib

# Third Party
import pytest

# First Party
import lmcache as lmc

pytestmark = pytest.mark.musa

# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------


class _StubTorch:
    """Minimal stand-in for ``torch`` exposing only what ``_detect_device`` reads."""

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
        self.cuda = SimpleNamespace(is_available=lambda: True)
        if has_musa:
            self.musa = SimpleNamespace(is_available=lambda: musa_available)
        if has_xpu:
            self.xpu = SimpleNamespace(is_available=lambda: xpu_available)
        if has_hpu:
            self.hpu = SimpleNamespace(is_available=lambda: hpu_available)


def _detect_with_stub(stub: _StubTorch) -> tuple[Any, str]:
    """Run ``_detect_device`` with ``torch`` swapped for the stub."""
    # First Party
    from lmcache.v1.platform._device_detect import _detect_device

    with patch.dict("sys.modules", {"torch": stub}):
        dev, name, _ = _detect_device()
        return dev, name


def test_detect_device_prefers_musa_when_available() -> None:
    """``_detect_device`` returns MUSA whenever ``torch.musa.is_available()``."""
    # Standard
    import os

    os.environ["DEVICE_TYPE"] = "musa"
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
    del os.environ["DEVICE_TYPE"]


def test_detect_device_falls_back_past_unavailable_musa() -> None:
    """Falls through MUSA when ``torch.musa.is_available()`` is False."""
    # Standard
    import os

    os.environ["DEVICE_TYPE"] = "musa"
    stub = _StubTorch(
        has_musa=True,
        musa_available=False,
    )
    _, name = _detect_with_stub(stub)
    assert name == "cuda"
    del os.environ["DEVICE_TYPE"]


def test_detect_device_cuda_fallback_when_no_alt_accelerator() -> None:
    """Default fallback is CUDA so existing CUDA tests/paths keep working."""
    stub = _StubTorch()
    dev, name = _detect_with_stub(stub)
    assert name == "cuda"
    assert dev is stub.cuda


# ---------------------------------------------------------------------------
# Module import sanity
# ---------------------------------------------------------------------------


def test_lmcache_exports_torch_dev_and_torch_device_type() -> None:
    """The contract used by every accelerator-aware module is the
    ``torch_dev`` / ``torch_device_type`` pair exported from ``lmcache``.

    Re-import to defeat any per-test monkeypatching above.
    """
    importlib.reload(lmc)
    assert hasattr(lmc, "torch_dev")
    assert isinstance(lmc.torch_device_type, str)
    assert lmc.torch_device_type in {"cuda", "musa", "xpu", "hpu", "npu", "cpu"}
