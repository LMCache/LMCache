# SPDX-License-Identifier: Apache-2.0
"""RBLN support unit tests that do not require RBLN hardware.

These tests cover the device-backend contract documented in
``docs/design/v1/platform/rbln/README.md``:

- Registry discovery of :class:`~lmcache.v1.platform.rbln.RblnDeviceSpec`.
- Availability probing, including the case where ``torch.rbln.is_available()``
  raises because every NPU is already claimed.
- The engine-driven-only capability surface (no IPC handle transfer, no event
  IPC backend, no cache context).

``torch`` is replaced with a stub in ``sys.modules`` rather than importing
``torch_rbln``, so the suite runs on any platform.
"""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform._device_detect import _detect_device, get_device_spec
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.rbln import RblnDeviceSpec
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raise_runtime_error() -> bool:
    """Stand in for a ``torch.rbln.is_available()`` that cannot claim an NPU."""
    raise RuntimeError(
        "rbln_register_device_id failed for rbln:0 on physical NPU(s) [0] (rc=1)"
    )


class _StubTorch:
    """Minimal ``torch`` stand-in exposing only what detection reads."""

    def __init__(self, rbln: object = None) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: False)
        if rbln is not None:
            self.rbln = rbln


def _is_available_with(rbln: object) -> bool:
    """Run ``RblnDeviceSpec.is_available`` against a stubbed ``torch``."""
    with patch.dict("sys.modules", {"torch": _StubTorch(rbln)}):
        return RblnDeviceSpec().is_available()


# ---------------------------------------------------------------------------
# Registry discovery
# ---------------------------------------------------------------------------


def test_spec_is_discovered_by_the_registry() -> None:
    """Defining the subclass is enough -- no manual registration needed."""
    spec = get_device_spec("rbln")
    assert isinstance(spec, RblnDeviceSpec)


def test_device_identifiers() -> None:
    """``device_type`` and ``torch_module_name`` both resolve to ``rbln``."""
    spec = RblnDeviceSpec()
    assert spec.device_type == "rbln"
    assert spec.torch_module_name == "rbln"


def test_resolve_device_ops_returns_the_rbln_ops_singleton() -> None:
    """``resolve_device_ops`` binds the RBLN ops instead of raising."""
    ops = resolve_device_ops("rbln")
    assert isinstance(ops, RblnDeviceOps)
    assert ops.device_type == "rbln"
    # Cached singleton: repeated lookups share native bindings and state.
    assert resolve_device_ops("rbln") is ops


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------


def test_is_available_when_torch_reports_a_usable_device() -> None:
    """A present, available ``torch.rbln`` makes the spec available."""
    assert _is_available_with(SimpleNamespace(is_available=lambda: True)) is True


def test_unavailable_when_torch_reports_no_device() -> None:
    """``torch.rbln.is_available() is False`` makes the spec unavailable."""
    assert _is_available_with(SimpleNamespace(is_available=lambda: False)) is False


def test_unavailable_without_the_torch_rbln_backend() -> None:
    """A torch build without the RBLN backend registered is not available."""
    assert _is_available_with(None) is False


def test_unavailable_instead_of_raising_when_every_npu_is_claimed() -> None:
    """A raising ``torch.rbln.is_available()`` degrades to "unavailable".

    Detection runs on every LMCache start. On a host where another process
    already holds the NPUs, ``torch.rbln.is_available()`` raises; letting that
    escape would abort import for every co-tenant process.
    """
    assert _is_available_with(SimpleNamespace(is_available=_raise_runtime_error)) is (
        False
    )


def test_detect_device_selects_rbln_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DEVICE_TYPE=rbln`` routes detection to the RBLN torch module."""
    monkeypatch.setenv("DEVICE_TYPE", "rbln")
    stub = _StubTorch(SimpleNamespace(is_available=lambda: True))
    with patch.dict("sys.modules", {"torch": stub}):
        torch_dev, device_type, backend_name = _detect_device()
    assert device_type == "rbln"
    assert backend_name == "rbln"
    assert torch_dev is stub.rbln


# ---------------------------------------------------------------------------
# Engine-driven-only capability surface
# ---------------------------------------------------------------------------


def test_handle_transfer_is_unavailable() -> None:
    """RBLN opts out of the base class' permissive default.

    ``torch.rbln`` exposes no ``Event`` type, so ``mp_transfer_mode=
    lmcache_driven`` must fail at its documented validation point.
    """
    assert DeviceSpec().is_handle_transfer_available() is True
    assert RblnDeviceSpec().is_handle_transfer_available() is False


def test_no_ipc_wrapper_and_no_event_backend() -> None:
    """Neither LMCache-driven building block is advertised."""
    spec = RblnDeviceSpec()
    assert spec.ipc_wrapper_cls is None
    assert spec.event_ipc_backend is None


def test_create_cache_context_is_not_implemented() -> None:
    """The cache context is LMCache-driven only, so it stays unimplemented."""
    spec = RblnDeviceSpec()
    with pytest.raises(NotImplementedError):
        spec.create_cache_context()


def test_pin_memory_falls_back_to_the_default_backend() -> None:
    """No RBLN pin-memory backend is registered, so the default applies."""
    spec: Any = RblnDeviceSpec()
    assert spec.pin_memory_backend is None
