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
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_spec import DeviceSpec

logger = init_logger(__name__)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_ops import DeviceOps


def _neuron_device_count() -> int:
    """Number of visible NeuronCores.

    The Neuron runtime's own environment variables are authoritative when set;
    otherwise fall back to the number of Neuron devices exposed by the driver.
    """
    # Standard
    import glob
    import os

    visible = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if visible:
        # Comma-separated list of core ids and/or "lo-hi" ranges.
        count = 0
        for part in visible.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, _, hi = part.partition("-")
                count += int(hi) - int(lo) + 1
            else:
                count += 1
        if count:
            return count

    num_cores = os.environ.get("NEURON_RT_NUM_CORES")
    if num_cores:
        return int(num_cores)

    # A count of zero would make callers treat a single node as multi-node.
    return len(glob.glob("/dev/neuron*")) or 1


def _neuron_synchronize() -> None:
    """Flush pending device work.

    ``torch_xla.sync()`` is issued when torch-xla is loaded, so the lazy XLA
    backend (where pending work *is* queued until a step boundary) is flushed
    instead of silently skipped.  Otherwise this does nothing: the native
    TorchNeuron backend exposes no stream or event abstraction, so there is no
    asynchronous queue to drain and device work has already completed by the
    time control returns.

    Failures are swallowed: an unavailable step barrier must not take down the
    transfer path.
    """
    # Standard
    import sys

    # Importing torch_xla here would drag in its own device setup, so only step
    # it when something else has already loaded it.
    torch_xla = sys.modules.get("torch_xla")
    if torch_xla is None:
        return
    try:
        xla_sync = getattr(torch_xla, "sync", None)
        if xla_sync is not None:
            xla_sync()
            return
        # Third Party
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    except Exception:  # pragma: no cover - defensive
        logger.debug("torch_xla step barrier failed; continuing", exc_info=True)


def _neuron_set_device(device: object) -> None:
    """Select a NeuronCore.

    The runtime binds cores at process start via ``NEURON_RT_VISIBLE_CORES``,
    and ``torch.neuron.current_device()`` is hard-coded to 0, so there is no
    per-process device to switch to.  Accepted and ignored.
    """
    return None


# Methods the Neuron SDK does not put on ``torch.neuron``, which LMCache's
# generic ``torch_dev`` call sites expect.  libtorch_neuronx_lite registers the
# device module as a bare ``types.ModuleType`` and attaches only
# ``is_available`` and ``current_device``, so filling in the rest here keeps the
# Neuron-specific knowledge in the Neuron backend instead of spreading
# ``hasattr`` branches across shared code.
_DEVICE_MODULE_SHIMS = {
    "synchronize": _neuron_synchronize,
    "device_count": _neuron_device_count,
    "set_device": _neuron_set_device,
}


def _install_device_module_shims(torch_neuron: Any) -> None:
    """Attach the missing device-module methods, without overwriting the SDK's.

    Idempotent, and never shadows a real implementation: if a future Neuron SDK
    ships one of these, that one wins.
    """
    for name, impl in _DEVICE_MODULE_SHIMS.items():
        if not hasattr(torch_neuron, name):
            setattr(torch_neuron, name, impl)


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
        # First Party
        from lmcache.v1.platform.devices.neuron.device_ops import NeuronDeviceOps

        return NeuronDeviceOps

    def is_available(self) -> bool:
        """Check Neuron availability.

        Uses ``torch.neuron`` if already registered (e.g. by vllm-neuron),
        else imports ``torch_neuronx`` to register it.  Also fills in the
        device-module methods the SDK omits, so that the generic
        ``torch_dev.<method>()`` call sites throughout LMCache work on Neuron
        without each one needing a Neuron-specific branch.
        """
        try:
            # Third Party
            import torch

            if not hasattr(torch, "neuron"):
                try:
                    # Third Party
                    import torch_neuronx  # noqa: F401 — side-effect: registers torch.neuron
                except ImportError:
                    return False
            if not hasattr(torch, "neuron"):
                return False
            _install_device_module_shims(torch.neuron)  # type: ignore[attr-defined]
            return torch.neuron.is_available()  # type: ignore[attr-defined]
        except Exception:
            return False

    def is_handle_transfer_available(self) -> bool:
        return False
