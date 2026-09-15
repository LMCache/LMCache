# SPDX-License-Identifier: Apache-2.0
"""Tests for the AWS Trainium (Neuron) platform backend.

These tests verify the NeuronDeviceSpec and NeuronDeviceOps classes
conform to the DeviceSpec/DeviceOps contract.  Most tests run on any
host (the Neuron SDK is not required); the hardware-verification test
is skipped when torch_neuronx is not installed.
"""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.neuron import NeuronDeviceSpec
import lmcache.v1.platform as platform_pkg


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


# -- Visible-device parsing ------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", 1),
        ("   ", 1),
        ("5", 1),
        ("0,1,2", 3),
        ("0-7", 8),
        ("0-3,8-11", 8),
        ("0-3,8", 5),
        ("0-0", 1),
        ("3-1", 1),
        ("garbage", 1),
    ],
)
def test_parse_visible_devices(value: str, expected: int) -> None:
    """NEURON_VISIBLE_DEVICES ranges are counted, not just comma-separated ids.

    vllm-neuron sets this in range form (e.g. "0-7" for two replicas of four
    cores), so counting commas alone under-reports a whole-node allocation.
    """
    # First Party
    from lmcache.v1.platform.neuron import parse_visible_devices

    assert parse_visible_devices(value) == expected


# -- torch module adapter --------------------------------------------------


def test_adapt_torch_module_supplies_missing_methods(
    neuron_spec: NeuronDeviceSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter answers the calls torch.neuron does not define."""
    monkeypatch.setenv("NEURON_VISIBLE_DEVICES", "0-7")

    class BareModule:
        """Stands in for a torch.neuron without the CUDA-shaped surface."""

        def is_available(self) -> bool:
            return True

    adapted = neuron_spec.adapt_torch_module(BareModule())

    assert adapted.device_count() == 8
    assert adapted.set_device(0) is None
    assert adapted.current_stream().synchronize() is None
    assert adapted.Stream is not None


def test_adapt_torch_module_does_not_mutate_the_module(
    neuron_spec: NeuronDeviceSpec,
) -> None:
    """Adapting leaves the vendor's module untouched."""

    class BareModule:
        """Stands in for a torch.neuron without the CUDA-shaped surface."""

    module = BareModule()
    neuron_spec.adapt_torch_module(module)

    assert not hasattr(module, "set_device")
    assert not hasattr(module, "device_count")
    assert not hasattr(module, "Stream")


def test_adapt_torch_module_forwards_real_attributes(
    neuron_spec: NeuronDeviceSpec,
) -> None:
    """Attributes the real module defines are passed straight through."""

    class BareModule:
        """Stands in for a torch.neuron exposing its own attribute."""

        sentinel = "from-the-real-module"

    adapted = neuron_spec.adapt_torch_module(BareModule())

    assert adapted.sentinel == "from-the-real-module"
