# SPDX-License-Identifier: Apache-2.0
"""Neuron ops backend: torch baseline with optional native overlay.

:class:`NeuronDeviceOps` attempts to bind a native ops module in
:meth:`ensure_native`.  When no native extension is available (the
expected case for initial bring-up), the instance stays on the torch
baseline.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class NeuronDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "neuron"

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True
        try:
            # Third Party
            import neuron_device.ops as native  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "neuron native ops not found; NeuronDeviceOps stays on "
                "the torch baseline for all ops."
            )
            return
        self.bind_native(native)
