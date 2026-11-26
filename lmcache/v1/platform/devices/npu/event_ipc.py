# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU device-event IPC backend."""

# Future
from __future__ import annotations

# Standard
import os
from typing import Any

# First Party
from lmcache.v1.common.logging import init_logger
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

    CANN device event resources (notify slots backing each ``aclrtCreateEvent``)
    are far scarcer than CUDA's, so the exported-event liveness window shrinks
    from the historical 2048 to a small ring; aged-out events are settled via
    ``synchronize()`` before their last reference is dropped, so
    ``aclrtDestroyEvent`` reclaims them deterministically. Without this, a
    long-serving MP worker accumulates live driver events until event
    creation fails with error 207001 ("driver error: out of memory").
    """

    _DEFAULT_RING_SIZE = 128
    _RING_SIZE_ENV = "LMCACHE_NPU_EVENT_RING_SIZE"

    def __init__(
        self,
        event_module: Any | None = None,
        exported_event_ring_size: int | None = None,
    ) -> None:
        """Create an NPU event IPC backend.

        Args:
            event_module: Optional torch.npu-like module. Defaults to the
                installed torch_npu; injectable for tests.
            exported_event_ring_size: Liveness-window size for exported
                events. Defaults to the ``LMCACHE_NPU_EVENT_RING_SIZE``
                environment variable or 128.
        """
        if exported_event_ring_size is None:
            raw = os.environ.get(self._RING_SIZE_ENV)
            exported_event_ring_size = (
                int(raw) if raw is not None else self._DEFAULT_RING_SIZE
            )
        super().__init__(
            event_module=(
                event_module if event_module is not None else _torch_npu_module()
            ),
            device_type="npu",
            exported_event_ring_size=exported_event_ring_size,
        )

    def _on_event_evicted(self, event: Any) -> None:
        """Settle an event before its last reference is dropped.

        Args:
            event: The evicted backend-native event.
        """
        try:
            event.synchronize()
        except Exception:
            logger.warning(
                "NPU exported-event reclaim: synchronize failed; "
                "dropping reference anyway.",
                exc_info=True,
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
        """Import a serialized event handle under ``device``'s context.

        CANN derives the imported event from the calling thread's current
        device context (``aclIpcOpenEventHandle``), not from the ``device``
        argument alone, so a handler thread serving several devices must pin
        the owning device before opening the handle — the same guard
        :meth:`export_event` applies on the way out.

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
            with self._event_module.device(device):  # type: ignore[attr-defined]
                return super().import_event(handle, device)
        except Exception as exc:
            logger.warning(
                "NPU event import failed: device=%s %s error=%s",
                device,
                _handle_summary(handle),
                exc,
            )
            raise

    def record_event(self, event: object, stream: object) -> None:
        """Record ``event`` on ``stream`` under the stream's device context.

        torch_npu creates the underlying driver event lazily at record time
        (``AclrtCreateEventWithFlag``), bound to the calling thread's current
        device context. Recording under a stale context allocates the event
        in the wrong device's event pool — it is then never reclaimed by the
        owning device (error 207001) and the record itself can be rejected
        (EE1010, "stream does not belong to the current context").

        Args:
            event: Backend-native event to record.
            stream: Stream that should record the event.
        """
        with self._event_module.device(stream.device):  # type: ignore[attr-defined]
            super().record_event(event, stream)

    def wait_event(self, event: object, stream: object) -> None:
        """Make ``stream`` wait for ``event`` under the stream's device context.

        The stream-side wait is queued against the stream's device context,
        so the same pinning as :meth:`record_event` applies.

        Args:
            event: Backend-native event to wait on.
            stream: Stream that should wait for the event.
        """
        with self._event_module.device(stream.device):  # type: ignore[attr-defined]
            super().wait_event(event, stream)
