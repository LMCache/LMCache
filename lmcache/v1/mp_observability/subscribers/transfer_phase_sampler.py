# SPDX-License-Identifier: Apache-2.0

"""Pops finished gather/DMA phase timings onto the event bus.

Subscribes to ``MP_STORE_END`` / ``MP_RETRIEVE_END``. Both are published on
the transfer stream, so by the time one is dispatched every section of that
transfer has completed on the GPU: popping then yields the transfer's full
sample set, published as one ``MP_TRANSFER_PHASE_SAMPLES`` event for the
metrics and tracing subscribers.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)

try:
    # Third Party
    import torch  # noqa: F401 — must be imported before native extensions

    # First Party
    from lmcache import device_ops as _device_ops

    _HAS_TRANSFER_PHASE_TIMING = hasattr(_device_ops, "pop_completed_phase_timings")
except ImportError:
    _HAS_TRANSFER_PHASE_TIMING = False


class TransferPhaseSampler(EventSubscriber):
    """Bridges the native phase-timing recorder to the event bus.

    Args:
        bus: Bus that receives the ``MP_TRANSFER_PHASE_SAMPLES`` events; see
            the ``EventType`` docstring for the metadata layout.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_END: self._on_transfer_end,
            EventType.MP_RETRIEVE_END: self._on_transfer_end,
        }

    def _on_transfer_end(self, event: Event) -> None:
        """Pop every finished sample and publish them in one event.

        No-op when the native op is unavailable or nothing has finished.
        """
        if not _HAS_TRANSFER_PHASE_TIMING:
            return
        samples = _device_ops.pop_completed_phase_timings()
        if not samples:
            return
        self._bus.publish(
            Event(
                event_type=EventType.MP_TRANSFER_PHASE_SAMPLES,
                metadata={"samples": samples},
            )
        )
