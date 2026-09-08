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

# Standard
from typing import Any
import os

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.base.device_spec import DeviceSpec


def parse_visible_devices(value: str) -> int:
    """Count the NeuronCores named by a ``NEURON_VISIBLE_DEVICES`` value.

    The variable accepts comma-separated ids, inclusive ranges, or a mix of
    both -- vllm-neuron uses range form (``"0-7"`` for two replicas of four
    cores). Counting commas alone reports ``1`` for a range spanning a whole
    node.

    Args:
        value: Raw ``NEURON_VISIBLE_DEVICES`` value, e.g. ``"0-3,8"``.

    Returns:
        Number of cores named, or ``1`` when the value is empty or malformed.
    """
    if not value.strip():
        return 1
    total = 0
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                start, stop = int(low), int(high)
            except ValueError:
                return 1
            if stop < start:
                return 1
            total += stop - start + 1
        else:
            try:
                int(part)
            except ValueError:
                return 1
            total += 1
    return total or 1


class _NoOpStream:
    """Synchronous stand-in for a CUDA stream on a device without them."""

    def synchronize(self) -> None:
        """Return immediately; work on this device is already complete."""

    def wait_stream(self, other: object) -> None:
        """Return immediately; there is no stream ordering to enforce.

        Args:
            other: Accepted for signature compatibility and ignored.
        """


class _NeuronTorchModule:
    """Adapter over ``torch.neuron`` supplying the calls LMCache makes.

    ``torch.neuron`` does not expose ``set_device``, ``device_count``,
    ``Stream`` or ``current_stream``. Rather than adding them to the vendor's
    module -- which is process-global and shared with every other library --
    this proxies every real attribute through and answers only for the missing
    ones. Core selection on Neuron is external, via ``NEURON_VISIBLE_DEVICES``,
    so ``set_device`` is a no-op by design.
    """

    def __init__(self, module: Any) -> None:
        """Wrap a torch device module.

        Args:
            module: The real ``torch.neuron`` module.
        """
        self._module = module

    def __getattr__(self, name: str) -> Any:
        """Forward any attribute LMCache asks for to the wrapped module.

        Args:
            name: Attribute name.

        Returns:
            The wrapped module's attribute.

        Raises:
            AttributeError: If the wrapped module does not define it.
        """
        return getattr(self._module, name)

    def __repr__(self) -> str:
        """Return a representation naming the wrapped module."""
        return f"_NeuronTorchModule({self._module!r})"

    def set_device(self, device: object = None) -> None:
        """Accept and ignore a device selection.

        Neuron core visibility is set by ``NEURON_VISIBLE_DEVICES`` before the
        process starts; there is no in-process equivalent to switch to.

        Args:
            device: Accepted for signature compatibility and ignored.
        """

    def device_count(self) -> int:
        """Return the number of NeuronCores visible to this process.

        Returns:
            The count named by ``NEURON_VISIBLE_DEVICES``, or ``1``.
        """
        return parse_visible_devices(os.environ.get("NEURON_VISIBLE_DEVICES", ""))

    @property
    def Stream(self) -> type[_NoOpStream]:  # noqa: N802 -- mirrors torch.cuda.Stream
        """Return the no-op stream type used in place of CUDA streams."""
        return _NoOpStream

    def current_stream(self) -> _NoOpStream:
        """Return a no-op stream standing in for the current device stream."""
        return _NoOpStream()


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

        Supports both the legacy ``torch_neuronx`` XLA shim (which calls
        ``rename_privateuse1_backend("neuron")``) and the native PyTorch
        Neuron backend (SDK >= 2.27) where ``torch.neuron`` is registered
        by ``vllm_neuron`` or the runtime itself.
        """
        try:
            # Third Party
            import torch

            if hasattr(torch, "neuron") and torch.neuron.is_available():  # type: ignore[attr-defined]
                return True
            try:
                # Third Party
                import torch_neuronx  # noqa: F401
            except ImportError:
                return False
            return hasattr(torch, "neuron") and torch.neuron.is_available()  # type: ignore[attr-defined]
        except Exception:
            return False

    def adapt_torch_module(self, torch_module: Any) -> Any:
        """Wrap ``torch.neuron`` so the calls LMCache makes all resolve.

        Args:
            torch_module: The real ``torch.neuron`` module.

        Returns:
            A :class:`_NeuronTorchModule` proxying it.
        """
        return _NeuronTorchModule(torch_module)

    def is_handle_transfer_available(self) -> bool:
        return False
