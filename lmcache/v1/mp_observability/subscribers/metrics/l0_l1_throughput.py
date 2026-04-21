# SPDX-License-Identifier: Apache-2.0

"""L0↔L1 throughput metrics subscriber.

Emits two OTel histograms in GB/s, labeled by ``engine_id`` and ``gpu_id``:
  - ``lmcache_mp.l0_l1_store_throughput_gbs``  — GPU→CPU (L0→L1) store
  - ``lmcache_mp.l0_l1_load_throughput_gbs``   — CPU→GPU (L1→L0) load

Implementation:
  - Correlates ``MP_STORE_START`` → ``MP_STORE_END`` and
    ``MP_RETRIEVE_START`` → ``MP_RETRIEVE_END`` pairs by ``session_id``
    (= vLLM ``request_id``).
  - START/END events fire on the GPU cupy stream (``publish_on_stream``),
    so their timestamps reflect true GPU-stream time for the D2H/H2D
    copies — not Python/lock overhead.
  - Sampling decision made at START time via ``random.random() <
    sample_rate``.  Unsampled sessions leave zero state.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any
import random

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class L0L1ThroughputSubscriber(EventSubscriber):
    """Records L0↔L1 throughput by correlating MP_*_START→MP_*_END pairs.

    Parameters:
        sample_rate: Fraction of requests to track (0, 1.0].  Default 0.01
            (1%), matching other lifecycle subscribers.
    """

    def __init__(self, sample_rate: float = 0.01) -> None:
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self._sample_rate = sample_rate

        # session_id -> t_start. Populated only for sampled sessions.
        self._pending_store: dict[str, float] = {}
        self._pending_load: dict[str, float] = {}

        meter = metrics.get_meter("lmcache.l0_l1")
        self._store_hist = meter.create_histogram(
            "lmcache_mp.l0_l1_store_throughput_gbs",
            description=(
                "Histogram of L0→L1 (GPU→CPU) store throughput in GB/s, "
                "measured per request as total_bytes / (end_ts - start_ts) "
                "on the GPU cupy stream."
            ),
            unit="GB/s",
        )
        self._load_hist = meter.create_histogram(
            "lmcache_mp.l0_l1_load_throughput_gbs",
            description=(
                "Histogram of L1→L0 (CPU→GPU) load throughput in GB/s, "
                "measured per request as total_bytes / (end_ts - start_ts) "
                "on the GPU cupy stream."
            ),
            unit="GB/s",
        )

    # -- EventSubscriber interface -----------------------------------------

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_store_start,
            EventType.MP_STORE_END: self._on_store_end,
            EventType.MP_RETRIEVE_START: self._on_retrieve_start,
            EventType.MP_RETRIEVE_END: self._on_retrieve_end,
        }

    # -- Store path (L0→L1, GPU→CPU) ---------------------------------------

    def _on_store_start(self, event: Event) -> None:
        if random.random() >= self._sample_rate:
            return
        if event.session_id:
            self._pending_store[event.session_id] = event.timestamp

    def _on_store_end(self, event: Event) -> None:
        self._record(
            event=event,
            pending=self._pending_store,
            hist=self._store_hist,
        )

    # -- Retrieve path (L1→L0, CPU→GPU) ------------------------------------

    def _on_retrieve_start(self, event: Event) -> None:
        if random.random() >= self._sample_rate:
            return
        if event.session_id:
            self._pending_load[event.session_id] = event.timestamp

    def _on_retrieve_end(self, event: Event) -> None:
        self._record(
            event=event,
            pending=self._pending_load,
            hist=self._load_hist,
        )

    # -- Core computation --------------------------------------------------

    @staticmethod
    def _record(
        event: Event,
        pending: dict[str, float],
        hist: Any,
    ) -> None:
        t_start = pending.pop(event.session_id, None)
        if t_start is None:
            return  # session wasn't sampled

        total_bytes = event.metadata.get("total_bytes", 0)
        if total_bytes <= 0:
            return

        dt = event.timestamp - t_start
        if dt <= 0:
            return

        engine_id = event.metadata.get("engine_id")
        gpu_id = event.metadata.get("gpu_id")
        attrs: dict[str, Any] = {}
        if engine_id is not None:
            attrs["engine_id"] = str(engine_id)
        if gpu_id is not None:
            attrs["gpu_id"] = str(gpu_id)

        hist.record(total_bytes / dt / 1e9, attributes=attrs)
