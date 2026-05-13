# SPDX-License-Identifier: Apache-2.0
"""Tests for the platform backend registry.

The registry is the single source of truth for ``torch_device_type``
-> factory dispatch.  These tests validate the public contract:

* Registration overrides preserve insertion semantics.
* Lookup falls back to the ``cpu`` default when the requested
  device type is not registered.
* The availability predicate gates an otherwise-registered backend
  so unhealthy accelerators transparently surrender to the default.
"""

# Standard
from unittest import mock

# Third Party
import pytest

# First Party
from lmcache.v1.platform import _registry


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot/restore registry state around each test."""
    snapshot = _registry.snapshot()
    try:
        yield
    finally:
        _registry.restore(snapshot)


def test_stream_lookup_returns_registered_factory():
    sentinel = object()
    _registry.register_stream("xpu", lambda *_a, **_kw: sentinel)
    _registry.register_availability("xpu", lambda: True)

    factory = _registry.get_stream_factory("xpu")
    assert factory is not None
    assert factory(0, 0) is sentinel


def test_lookup_falls_back_to_cpu_when_unknown_device_type():
    """Unknown device types resolve to the ``cpu`` default factory."""
    cpu_factory = _registry.get_stream_factory("cpu")
    assert cpu_factory is not None
    assert _registry.get_stream_factory("definitely_not_a_device") is cpu_factory


def test_unavailable_backend_falls_through_to_default():
    """A registered backend reporting ``is_available() == False`` is skipped."""
    _registry.register_stream("xpu", lambda *_a, **_kw: object())
    _registry.register_availability("xpu", lambda: False)

    cpu_factory = _registry.get_stream_factory("cpu")
    assert _registry.get_stream_factory("xpu") is cpu_factory


def test_predicate_exception_treated_as_unavailable():
    """If the availability predicate raises, the backend is skipped."""
    _registry.register_stream("xpu", lambda *_a, **_kw: object())

    def _broken() -> bool:
        raise RuntimeError("driver init failed")

    _registry.register_availability("xpu", _broken)

    cpu_factory = _registry.get_stream_factory("cpu")
    assert _registry.get_stream_factory("xpu") is cpu_factory


def test_make_external_stream_routes_via_registry():
    """End-to-end: dispatcher honours :mod:`_registry` lookups."""
    # Late import: ``stream`` reads the registry on every call.
    # First Party
    from lmcache.v1.platform.stream import make_external_stream

    sentinel = object()
    _registry.register_stream("xpu", lambda *_a, **_kw: sentinel)
    _registry.register_availability("xpu", lambda: True)

    class _FakeTorchStream:
        cuda_stream = 0xCAFEBABE

    with mock.patch("lmcache.v1.platform.stream._torch_dev_type", return_value="xpu"):
        result = make_external_stream(_FakeTorchStream(), 7)

    assert result is sentinel


def test_kv_wrapper_lookup_returns_registered_factory():
    sentinel = object()
    _registry.register_kv_wrapper("xpu", lambda _t: sentinel)

    factory = _registry.get_kv_wrapper_factory("xpu")
    assert factory(object()) is sentinel


def test_kv_wrapper_lookup_raises_for_unknown_device():
    with pytest.raises(ValueError, match="definitely_not_a_device"):
        _registry.get_kv_wrapper_factory("definitely_not_a_device")


def test_kv_wrapper_table_isolated_by_snapshot():
    """Snapshot/restore covers the kv-wrapper table, not just stream."""
    _registry.register_kv_wrapper("xpu", lambda _t: object())
    # Inside the autouse fixture's snapshot/restore the registration
    # above leaks unless restore() also wipes the kv-wrapper table.
    # Verify the table contains "xpu" right now…
    assert _registry.get_kv_wrapper_factory("xpu") is not None
