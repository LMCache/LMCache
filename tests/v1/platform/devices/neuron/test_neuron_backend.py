# SPDX-License-Identifier: Apache-2.0
"""Tests for the AWS Trainium (Neuron) platform backend.

These tests verify the NeuronDeviceSpec and NeuronDeviceOps classes
conform to the DeviceSpec/DeviceOps contract.  Most tests run on any
host (the Neuron SDK is not required); the hardware-verification test
is skipped when torch_neuronx is not installed.
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
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.devices.neuron import NeuronDeviceSpec
import lmcache.v1.platform as platform_pkg


class _StubTorch:
    """Minimal ``torch`` stand-in exposing only what detection reads."""

    def __init__(self, neuron: object = None) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: False)
        if neuron is not None:
            self.neuron = neuron


def _raise_runtime_error() -> bool:
    """Stand in for a ``torch.neuron.is_available()`` that cannot claim a core."""
    raise RuntimeError("NRT init failed: all NeuronCores are in use")


def _is_available_with(neuron: object, *, torch_neuronx: object) -> bool:
    """Run ``is_available`` with stubbed modules; ``None`` means not installed."""
    with patch.dict(
        "sys.modules", {"torch": _StubTorch(neuron), "torch_neuronx": torch_neuronx}
    ):
        return NeuronDeviceSpec().is_available()


@pytest.fixture
def neuron_spec() -> NeuronDeviceSpec:
    return NeuronDeviceSpec()


@pytest.fixture
def isolated_registry() -> Any:
    """Snapshot the device-spec table so tests can install fakes safely."""
    saved = dict(platform_pkg._DEVICE_REGISTRY)
    try:
        yield saved
    finally:
        platform_pkg._DEVICE_REGISTRY.clear()
        platform_pkg._DEVICE_REGISTRY.update(saved)


# -- Spec contract ---------------------------------------------------------


def test_neuron_spec_properties(neuron_spec: NeuronDeviceSpec) -> None:
    """NeuronDeviceSpec exposes the correct device identity."""
    assert neuron_spec.device_type == "neuron"
    assert neuron_spec.torch_module_name == "neuron"
    assert neuron_spec.ops_cls is DeviceOps


# -- Registry integration --------------------------------------------------


def test_neuron_registry_integration(
    isolated_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NeuronDeviceSpec resolves through the standard registry path."""
    monkeypatch.setattr(
        platform_pkg,
        "_DEVICE_REGISTRY",
        {**isolated_registry, "neuron": NeuronDeviceSpec()},
    )
    assert type(resolve_device_ops("neuron")) is DeviceOps


# -- Availability guards ---------------------------------------------------


def test_neuron_not_available_without_sdk(
    neuron_spec: NeuronDeviceSpec,
) -> None:
    """is_available() returns False when torch_neuronx is not installed."""
    assert neuron_spec.is_available() is False


def test_neuron_no_handle_transfer(
    neuron_spec: NeuronDeviceSpec,
) -> None:
    """Trainium does not support IPC handle transfer."""
    assert neuron_spec.is_handle_transfer_available() is False


def test_neuron_no_ipc_wrapper(
    neuron_spec: NeuronDeviceSpec,
) -> None:
    """No IPC wrapper — engine-driven path only."""
    assert neuron_spec.ipc_wrapper_cls is None


def test_neuron_no_event_ipc(
    neuron_spec: NeuronDeviceSpec,
) -> None:
    """No event IPC backend — engine-driven path only."""
    assert neuron_spec.event_ipc_backend is None


# -- Hardware verification (skipped on non-Neuron hosts) -------------------


@pytest.mark.neuron
def test_neuron_device_type_matches_torch_neuronx() -> None:
    """Fail loudly if our assumed device_type no longer matches
    what torch_neuronx actually registers.

    This test runs only when torch_neuronx is installed (e.g. on
    Trainium hardware).  On non-Neuron hosts it is skipped.
    """
    pytest.importorskip(
        "torch_neuronx",
        reason="torch_neuronx not installed — cannot verify device string",
    )
    # Third Party
    import torch

    assert hasattr(torch, "neuron"), (
        "torch_neuronx is installed but torch.neuron does not exist. "
        "The Neuron SDK may have changed how it registers the device. "
        "Update NeuronDeviceSpec.device_type and torch_module_name."
    )
    assert torch.neuron.is_available(), (
        "torch.neuron exists but is_available() is False. "
        "Neuron driver may not be loaded (lsmod | grep neuron)."
    )
    t = torch.tensor([1.0], device="neuron:0")
    actual = t.device.type
    assert actual == "neuron", (
        f"NeuronDeviceSpec assumes device_type='neuron' but "
        f"torch_neuronx reports '{actual}'. Update NeuronDeviceSpec."
    )


# -- Init flow: discovery -> availability -> detection ---------------------


def test_spec_is_discovered_by_the_registry() -> None:
    """Defining the subclass is enough -- no manual registration needed."""
    assert isinstance(get_device_spec("neuron"), NeuronDeviceSpec)


def test_available_when_torch_neuron_registered_without_torch_neuronx() -> None:
    """vllm-neuron registers ``torch.neuron`` without ``torch_neuronx``."""
    neuron = SimpleNamespace(is_available=lambda: True)
    assert _is_available_with(neuron, torch_neuronx=None) is True


def test_available_with_torch_neuronx_installed() -> None:
    """A host with ``torch_neuronx`` and a usable device is available."""
    neuron = SimpleNamespace(is_available=lambda: True)
    assert _is_available_with(neuron, torch_neuronx=SimpleNamespace()) is True


def test_unavailable_without_the_torch_neuron_backend() -> None:
    """No ``torch.neuron`` and no ``torch_neuronx`` means not available."""
    assert _is_available_with(None, torch_neuronx=None) is False


def test_unavailable_when_torch_neuron_reports_no_device() -> None:
    """``torch.neuron.is_available() is False`` makes the spec unavailable."""
    neuron = SimpleNamespace(is_available=lambda: False)
    assert _is_available_with(neuron, torch_neuronx=None) is False


def test_unavailable_instead_of_raising_when_cores_are_claimed() -> None:
    """A raising ``torch.neuron.is_available()`` degrades to "unavailable"."""
    neuron = SimpleNamespace(is_available=_raise_runtime_error)
    assert _is_available_with(neuron, torch_neuronx=None) is False


def test_detect_device_selects_registered_torch_neuron_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DEVICE_TYPE=neuron`` routes detection to the registered ``torch.neuron``."""
    monkeypatch.setenv("DEVICE_TYPE", "neuron")
    stub = _StubTorch(SimpleNamespace(is_available=lambda: True))
    with patch.dict("sys.modules", {"torch": stub, "torch_neuronx": None}):
        torch_dev, device_type, backend_name = _detect_device()
    assert device_type == "neuron"
    assert backend_name == "neuron"
    assert torch_dev is stub.neuron
