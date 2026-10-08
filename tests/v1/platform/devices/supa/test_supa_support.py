# SPDX-License-Identifier: Apache-2.0
"""Biren SUPA support unit tests.

These tests cover the device-backend contract documented in
``docs/design/v1/platform/devices/supa/README.md``:

- Registry discovery of :class:`~lmcache.v1.platform.devices.supa.SupaDeviceSpec`.
- ``torch_br``-driven availability probing, including the case where the
  runtime is absent.
- The engine-driven-only capability surface (no IPC handle transfer, no event
  IPC backend, no pin-memory backend).

``torch`` is replaced with a stub in ``sys.modules`` so the suite runs on any
platform.  The hardware-verification test is gated behind the ``supa`` pytest
marker and skipped when ``torch_br`` is not installed.
"""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform._device_detect import get_device_spec
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.devices.supa import SupaDeviceSpec
from lmcache.v1.platform.devices.supa.device_ops import SupaDeviceOps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTorch:
    """Minimal ``torch`` stand-in exposing only what detection reads."""

    def __init__(self, supa: object = None) -> None:
        if supa is not None:
            self.supa = supa


def _is_available_with(supa: object) -> bool:
    """Run ``SupaDeviceSpec.is_available`` against a stubbed ``torch``.

    ``torch_br`` is forced to be unimportable so the probe never depends on
    whether the runtime happens to be installed in the test environment.
    """
    with patch.dict("sys.modules", {"torch": _StubTorch(supa), "torch_br": None}):
        return SupaDeviceSpec().is_available()


# ---------------------------------------------------------------------------
# Registry discovery
# ---------------------------------------------------------------------------


def test_spec_is_discovered_by_the_registry() -> None:
    """Defining the subclass is enough -- no manual registration needed."""
    spec = get_device_spec("supa")
    assert isinstance(spec, SupaDeviceSpec)


def test_device_identifiers() -> None:
    """``device_type`` and ``torch_module_name`` both resolve to ``supa``."""
    spec = SupaDeviceSpec()
    assert spec.device_type == "supa"
    assert spec.torch_module_name == "supa"
    assert spec.backend_name == "supa"


def test_resolve_device_ops_returns_the_supa_ops_singleton() -> None:
    """``resolve_device_ops`` binds the SUPA ops instead of raising."""
    ops = resolve_device_ops("supa")
    assert isinstance(ops, SupaDeviceOps)
    assert ops.device_type == "supa"
    # Cached singleton: repeated lookups share native bindings and state.
    assert resolve_device_ops("supa") is ops


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------


def test_available_when_torch_supa_reports_a_usable_device() -> None:
    """A present, available ``torch.supa`` makes the spec available."""
    assert _is_available_with(SimpleNamespace(is_available=lambda: True)) is True


def test_unavailable_when_torch_supa_reports_no_device() -> None:
    """``torch.supa.is_available() is False`` makes the spec unavailable."""
    assert _is_available_with(SimpleNamespace(is_available=lambda: False)) is False


def test_unavailable_without_the_torch_br_runtime() -> None:
    """A torch build without ``torch_br`` is not available."""
    assert _is_available_with(None) is False


# ---------------------------------------------------------------------------
# Engine-driven-only capability surface
# ---------------------------------------------------------------------------


def test_handle_transfer_is_unavailable() -> None:
    """SUPA opts out of the base class' permissive default."""
    assert DeviceSpec().is_handle_transfer_available() is True
    assert SupaDeviceSpec().is_handle_transfer_available() is False


def test_no_ipc_wrapper_and_no_event_backend() -> None:
    """Neither LMCache-driven building block is advertised yet."""
    spec = SupaDeviceSpec()
    assert spec.ipc_wrapper_cls is None
    assert spec.event_ipc_backend is None


def test_create_cache_context_is_not_implemented() -> None:
    """The cache context is LMCache-driven only, so it stays unimplemented."""
    spec = SupaDeviceSpec()
    with pytest.raises(NotImplementedError):
        spec.create_cache_context()


def test_pin_memory_falls_back_to_the_default_backend() -> None:
    """No SUPA pin-memory backend is registered, so the default applies."""
    spec: Any = SupaDeviceSpec()
    assert spec.pin_memory_backend is None


# ---------------------------------------------------------------------------
# Hardware verification (skipped on non-Biren hosts)
# ---------------------------------------------------------------------------


@pytest.mark.supa
def test_supa_device_type_matches_torch_br() -> None:
    """Fail loudly if our assumed device_type no longer matches ``torch_br``.

    This test runs only when ``torch_br`` is installed and the SUPA runtime is
    usable (e.g. on Biren hardware); otherwise it is skipped.
    """
    try:
        # Third Party
        import torch_br  # noqa: F401 — side-effect: registers torch.supa
    except ImportError:
        pytest.skip("torch_br not installed — cannot verify the SUPA device string")

    # Third Party
    import torch

    assert hasattr(torch, "supa"), (
        "torch_br is installed but torch.supa does not exist. SUPA may have "
        "changed how it registers the device; update SupaDeviceSpec."
    )
    if not torch.supa.is_available():
        pytest.skip("torch.supa is present but no SUPA device is available")
    assert SupaDeviceSpec().is_available() is True
    tensor = torch.empty(1, device="supa:0")
    assert tensor.device.type == "supa", (
        f"SupaDeviceSpec assumes device_type='supa' but torch_br reports "
        f"'{tensor.device.type}'. Update SupaDeviceSpec."
    )
