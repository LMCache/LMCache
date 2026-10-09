# SPDX-License-Identifier: Apache-2.0
"""AWS Trainium (Neuron) platform helpers.

Targets the TorchNeuron native PyTorch backend (Neuron SDK >= 2.27),
which registers as ``torch.device("neuron")`` via PyTorch's
``rename_privateuse1_backend`` mechanism.  The older ``torch-neuronx``
XLA-based backend (``device_type="xla"``) is a fundamentally different
runtime with lazy-evaluation semantics and is not supported by this
spec.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.base.device_spec import DeviceSpec


class NeuronDeviceSpec(DeviceSpec):
    """Neuron device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "neuron"

    @property
    def torch_module_name(self) -> str:
        return "neuron"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        return DeviceOps

    def is_available(self) -> bool:
        """Check Neuron availability.

        Imports ``torch_neuronx`` to trigger
        ``rename_privateuse1_backend("neuron")``, which registers
        ``torch.neuron``.  Without this import the device module does
        not exist on ``torch`` and detection silently fails.
        """
        try:
            # Third Party
            import torch

            try:
                # Third Party
                import torch_neuronx  # noqa: F401 — side-effect: registers torch.neuron
            except ImportError:
                return False
            return hasattr(torch, "neuron") and torch.neuron.is_available()  # type: ignore[attr-defined]
        except Exception:
            return False

    def is_handle_transfer_available(self) -> bool:
        return False
