# SPDX-License-Identifier: Apache-2.0

"""MP transfer metrics subscriber — OTel counters for GPU store/retrieve.

Counts the LMCache-driven GPU transfers that the MP server puts on, and
takes off, the device stream:

- ``MP_STORE_SUBMITTED`` / ``MP_RETRIEVE_SUBMITTED`` are published
  CPU-synchronously by ``LMCacheDrivenTransferModule``, right before the
  copy is enqueued on the device stream.
- ``MP_STORE_END`` / ``MP_RETRIEVE_END`` are published *from* the device
  stream, once the copy has actually run.

Each pair carries the same ``device`` attribute and nothing else, so
``submitted - finished`` is a per-device count of transfers currently in
flight on the GPU.
"""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class MPTransferCountersSubscriber(EventSubscriber):
    """Maintains OTel counters for submitted and finished GPU transfers.

    Metrics (all counters, attr: ``device``):

    - ``lmcache_mp.num_submitted_stores`` — GPU stores enqueued
      (``MP_STORE_SUBMITTED``)
    - ``lmcache_mp.num_finished_stores`` — GPU stores completed
      (``MP_STORE_END``)
    - ``lmcache_mp.num_submitted_retrieves`` — GPU retrieves enqueued
      (``MP_RETRIEVE_SUBMITTED``)
    - ``lmcache_mp.num_finished_retrieves`` — GPU retrieves completed
      (``MP_RETRIEVE_END``)

    "Finished" counts a transfer *leaving* the device stream, not its
    success: a store that committed nothing (``stored_count == 0``) and a
    retrieve that missed both increment. Use ``lmcache_mp.num_chunks_loaded``
    and the L0↔L1 throughput histograms for success and volume.

    ``device`` is the only attribute on all four counters. The SUBMITTED
    events carry no ``engine_id`` / ``model_name`` (see EVENTS.md), so
    labeling the END side more richly would force a PromQL aggregation to
    line the two label sets up again before subtracting them.
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.transfer")
        self._submitted_stores = meter.create_counter(
            "lmcache_mp.num_submitted_stores",
            description=(
                "Total GPU→CPU store transfers enqueued on the device stream."
            ),
        )
        self._finished_stores = meter.create_counter(
            "lmcache_mp.num_finished_stores",
            description=(
                "Total GPU→CPU store transfers completed on the device stream "
                "(regardless of how many chunks were committed)."
            ),
        )
        self._submitted_retrieves = meter.create_counter(
            "lmcache_mp.num_submitted_retrieves",
            description=(
                "Total CPU→GPU retrieve transfers enqueued on the device stream."
            ),
        )
        self._finished_retrieves = meter.create_counter(
            "lmcache_mp.num_finished_retrieves",
            description=(
                "Total CPU→GPU retrieve transfers completed on the device "
                "stream (regardless of how many chunks were retrieved)."
            ),
        )

    # -- EventSubscriber interface -----------------------------------------

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_SUBMITTED: self._on_store_submitted,
            EventType.MP_STORE_END: self._on_store_finished,
            EventType.MP_RETRIEVE_SUBMITTED: self._on_retrieve_submitted,
            EventType.MP_RETRIEVE_END: self._on_retrieve_finished,
        }

    # -- Event handlers ----------------------------------------------------

    def _on_store_submitted(self, event: Event) -> None:
        self._submitted_stores.add(1, attributes=self._device_attrs(event))

    def _on_store_finished(self, event: Event) -> None:
        self._finished_stores.add(1, attributes=self._device_attrs(event))

    def _on_retrieve_submitted(self, event: Event) -> None:
        self._submitted_retrieves.add(1, attributes=self._device_attrs(event))

    def _on_retrieve_finished(self, event: Event) -> None:
        self._finished_retrieves.add(1, attributes=self._device_attrs(event))

    # -- Attributes --------------------------------------------------------

    @staticmethod
    def _device_attrs(event: Event) -> dict[str, str]:
        """Build ``{"device": ...}`` from *event*, or ``{}`` if it is absent.

        A missing ``device`` still counts — dropping the increment would
        silently understate the totals — but lands on the dimensionless
        data point rather than inventing a device label.

        Args:
            event: The MP store/retrieve event being counted.

        Returns:
            The OTel attribute dict for the counter increment.
        """
        device = event.metadata.get("device")
        if device is None:
            return {}
        return {"device": str(device)}
