# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU device-event IPC backend."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend

logger = init_logger(__name__)


def _torch_npu_module() -> Any:
    """Return ``torch.npu``, failing closed when torch_npu is absent."""
    # Third Party
    import torch

    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError(
            "torch_npu is not installed; NPU interprocess events are unavailable."
        )
    return npu


def _handle_summary(handle: bytes) -> str:
    """Return a stable diagnostic summary of an event handle."""
    return f"len={len(handle)} head={handle[:16].hex()}"


class NpuEventIPCBackend(DefaultEventIPCBackend):
    """Event IPC backend over torch_npu interprocess events.

    torch_npu implements the CUDA-style event ABI (``interprocess=True``,
    ``ipc_handle``, ``from_ipc_handle``), so the default adapter applies
    directly; ``check_event_support`` fails closed on builds lacking it.
    """

    def __init__(self, event_module: Any | None = None) -> None:
        """Create an NPU event IPC backend.

        Args:
            event_module: Optional torch.npu-like module. Defaults to the
                installed torch_npu; injectable for tests.
        """
        super().__init__(
            event_module=(
                event_module if event_module is not None else _torch_npu_module()
            ),
            device_type="npu",
        )

    def export_event(self, event: object, device: object) -> bytes:
        """Serialize an event for another process.

        Unlike CUDA, where an event carries its own device, CANN derives an
        interprocess handle from the calling thread's current device, so the
        serialization runs with ``device`` pinned current regardless of the
        caller's ambient device (e.g. a multi-device server handler thread).
        Source-event liveness pinning is inherited from the default backend.

        Args:
            event: Backend-native event to export.
            device: Device that owns the event.

        Returns:
            Serialized event handle.
        """
        with self._event_module.device(device):  # type: ignore[attr-defined]
            return super().export_event(event, device)

    def import_event(self, handle: bytes, device: object) -> object:
        """Import a serialized event handle, diagnosing failures.

        Args:
            handle: Serialized event handle from another process.
            device: Device on which to import.

        Returns:
            The imported backend-native event.

        Raises:
            RuntimeError: When the underlying import fails, enriched with
                the handle summary and target device for triage.
        """
        try:
            return super().import_event(handle, device)
        except Exception as exc:
            logger.warning(
                "NPU event import failed: device=%s %s error=%s",
                device,
                _handle_summary(handle),
                exc,
            )
            raise
