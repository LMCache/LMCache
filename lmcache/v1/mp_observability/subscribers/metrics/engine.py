# SPDX-License-Identifier: Apache-2.0

"""Worker-side engine metrics subscriber.

Emits counters tied to what the MP server delivers back to each vLLM
worker: the keys loaded per retrieve and, per object group, which tier
served them.  Future worker-scoped counters (e.g., bytes loaded, blocks
touched) would land here too.
"""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


def _per_group_counts(value: object) -> list[int]:
    """Decode a ``*_delivered_keys_by_group`` field into per-group counts.

    The emitter writes the counts as a comma-separated string (index = object
    group id) because events recorded on a device stream keep only int and str
    metadata values; a list of ints is accepted too.  Anything else, or a
    malformed string, reads as absent so the callback never raises.
    """
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        try:
            return [int(part) for part in parts if part]
        except ValueError:
            return []
    if isinstance(value, (list, tuple)):
        try:
            return [int(count) for count in value]
        except (TypeError, ValueError):
            return []
    return []


class EngineMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters tied to the MP server's retrieve path
    (worker-side; ``worker_id`` = vLLM worker instance id).

    Metrics:
    - ``lmcache_mp.num_chunks_loaded`` — object keys loaded from LMCache
      into the engine via ``retrieve()`` (attrs: ``worker_id``,
      ``model_name``, ``cache_salt``).
    - ``lmcache_mp.retrieve_delivered_l1_keys`` /
      ``lmcache_mp.retrieve_delivered_l2_keys`` /
      ``lmcache_mp.retrieve_delivered_unattributed_keys`` — the same keys
      split by the tier that served them, per object group (attrs as
      above plus ``object_group_id``).  The three sum to
      ``num_chunks_loaded``; see ``MP_RETRIEVE_END`` in EVENTS.md for the
      attribution rule.

    ``worker_id`` on these metrics names the vLLM **worker** instance id and
    is distinct from any scheduler-scoped ``scheduler_id`` attribute used
    elsewhere — the two IDs come from different vLLM processes and should
    not be cross-joined on dashboards.
    """

    #: ``(counter name, event metadata field)`` per delivered-key tier.
    _DELIVERED_TIERS: tuple[tuple[str, str], ...] = (
        ("lmcache_mp.retrieve_delivered_l1_keys", "l1_delivered_keys_by_group"),
        ("lmcache_mp.retrieve_delivered_l2_keys", "l2_delivered_keys_by_group"),
        (
            "lmcache_mp.retrieve_delivered_unattributed_keys",
            "unattributed_delivered_keys_by_group",
        ),
    )

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache_mp.engine")
        self._num_chunks_loaded = meter.create_counter(
            "lmcache_mp.num_chunks_loaded",
            description=(
                "Total number of LMCache chunks loaded into the engine "
                "(summed across all retrieve operations)."
            ),
        )
        descriptions = {
            "lmcache_mp.retrieve_delivered_l1_keys": (
                "Keys delivered to the engine by retrieve() that L1 already "
                "held when the request's lookup ran, per object group."
            ),
            "lmcache_mp.retrieve_delivered_l2_keys": (
                "Keys delivered to the engine by retrieve() that the request's "
                "prefetch loaded from L2, per object group."
            ),
            "lmcache_mp.retrieve_delivered_unattributed_keys": (
                "Keys delivered to the engine by retrieve() with no lookup "
                "record to attribute them (never guessed), per object group."
            ),
        }
        self._delivered = {
            field: meter.create_counter(
                name, unit="keys", description=descriptions[name]
            )
            for name, field in self._DELIVERED_TIERS
        }

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
        attrs: dict[str, str] = {}
        engine_id = event.metadata.get("engine_id")
        if engine_id is not None:
            attrs["worker_id"] = str(engine_id)
        model_name = event.metadata.get("model_name")
        if model_name is not None:
            attrs["model_name"] = str(model_name)
        cache_salt = event.metadata.get("cache_salt")
        if cache_salt is not None:
            attrs["cache_salt"] = str(cache_salt)
        self._num_chunks_loaded.add(retrieved_count, attributes=attrs)

        # Per-tier split, one series per object group.  Emitters that
        # predate the attribution fields only feed ``num_chunks_loaded``.
        for field, counter in self._delivered.items():
            for object_group_id, count in enumerate(
                _per_group_counts(event.metadata.get(field))
            ):
                if count > 0:
                    counter.add(
                        count,
                        attributes={**attrs, "object_group_id": str(object_group_id)},
                    )
