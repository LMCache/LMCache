# SPDX-License-Identifier: Apache-2.0
"""Cambricon MLU device discovery and stream primitives."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import importlib

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class MluDeviceSpec(DeviceSpec):
    """MLU backend for tensor-based, engine-driven KV transfers.

    Importing this module does not import torch_mlu or initialize a device.
    IPC and arbitrary host-memory registration retain the unsupported base
    capabilities; pinned tensors can still be allocated through torch_mlu.
    """

    @property
    def device_type(self) -> str:
        """Return torch_mlu's device type."""
        return "mlu"

    @property
    def torch_module_name(self) -> str:
        """Return the torch module registered by torch_mlu."""
        return "mlu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        """Load MLU operations after device discovery has completed."""
        # First Party
        from lmcache.v1.platform.devices.mlu.device_ops import MluDeviceOps

        return MluDeviceOps

    def is_available(self) -> bool:
        """Probe torch_mlu, returning False if its runtime is unavailable."""
        try:
            if not hasattr(torch, "mlu"):
                importlib.import_module("torch_mlu")
            return torch.mlu.is_available()
        except (ImportError, OSError, RuntimeError):
            logger.debug("MLU runtime is unavailable", exc_info=True)
            return False

    def get_stream_handle(self, stream: object) -> int:
        """Return the native CNRT queue handle of a torch_mlu stream.

        Args:
            stream: A torch_mlu Stream owned by the caller.

        Returns:
            Integer CNRT queue handle. The caller retains stream ownership.

        Raises:
            TypeError: If the stream does not expose an integer mlu_stream.
        """
        handle = getattr(stream, "mlu_stream", None)
        if not isinstance(handle, int):
            raise TypeError("MLU stream must expose an integer mlu_stream")
        return handle
