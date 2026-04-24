# SPDX-License-Identifier: Apache-2.0

"""Worker-side engine metrics subscriber.

Emits counters tied to what the MP server delivers back to each vLLM
worker.  Today the only metric is ``lmcache_mp.num_chunks_loaded``;
future worker-scoped counters (e.g., bytes loaded, blocks touched) would
land here too.
"""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class EngineMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters tied to the MP server's retrieve path
    (worker-side; ``worker_id`` = vLLM worker instance id).

    Metrics:
    - ``lmcache_mp.num_chunks_loaded`` — chunks loaded from LMCache into
      the engine via ``retrieve()`` (attr: ``worker_id``).

    ``worker_id`` on this metric names the vLLM **worker** instance id and
    is distinct from any scheduler-scoped ``scheduler_id`` attribute used
    elsewhere — the two IDs come from different vLLM processes and should
    not be cross-joined on dashboards.
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.engine")
        self._num_chunks_loaded = meter.create_counter(
            "lmcache_mp.num_chunks_loaded",
            description=(
                "Total number of LMCache chunks loaded into the engine "
                "(summed across all retrieve operations)."
            ),
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_RETRIEVE_END: self._on_retrieve_end,
        }

    def _on_retrieve_end(self, event: Event) -> None:
        retrieved_count = int(event.metadata.get("retrieved_count", 0))
        if retrieved_count <= 0:
            return
        # MP_RETRIEVE_END carries the worker's instance_id under the
        # ``engine_id`` key; re-emit as ``worker_id`` so the attribute
        # name disambiguates from any scheduler-side id used elsewhere.
        engine_id = event.metadata.get("engine_id")
        attrs: dict[str, str] = {}
        if engine_id is not None:
            attrs["worker_id"] = str(engine_id)
        self._num_chunks_loaded.add(retrieved_count, attributes=attrs)
