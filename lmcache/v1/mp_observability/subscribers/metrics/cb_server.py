# SPDX-License-Identifier: Apache-2.0

"""Blend metrics subscriber — OTel counters for cache blending events."""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict

# Third Party
from opentelemetry import metrics
from opentelemetry.metrics import Histogram

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)

# Cap on outstanding START timestamps awaiting their END. A lookup abandoned
# mid-poll never sends its END, so unmatched STARTs are normal; evict the
# oldest rather than grow without bound.
_MAX_PENDING_PHASES = 8192

#: Pairing key for a phase's START/END: ``(phase, session_id, worker_id)``.
#: ``worker_id`` is ``None`` for the once-per-request lookup legs; at TP>1
#: every rank publishes its own retrieve/scatter pair under the shared
#: request_id, so the worker must be part of the key or rank B's START
#: overwrites rank A's and the recorded interval spans two workers.
_PhaseKey = tuple[str, str, int | None]


class BlendMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for cache blending (CB) operations.

    Metrics:
    - ``lmcache_blend.lookup_requests``              — total CB lookup calls
    - ``lmcache_blend.lookup_requested_tokens``      — chunk-aligned tokens
      submitted for CB lookup (denominator of the blend token-level hit
      rate).  Sub-chunk trailing tokens are excluded because they cannot
      hit by design.
    - ``lmcache_blend.lookup_hit_tokens``            — tokens served by
      blend during the lookup (numerator of the blend token-level hit
      rate).  Equal to ``storage_hits * chunk_size``.
    - ``lmcache_blend.lookup_fingerprint_hits``      — fingerprint table hits
    - ``lmcache_blend.lookup_storage_hits``          — chunks confirmed in storage
    - ``lmcache_blend.lookup_stale_chunks``          — fingerprint hits evicted as stale
    - ``lmcache_blend.lookup_no_gpu_context_errors`` — lookup failures: no GPU context
    - ``lmcache_blend.retrieve_requests``            — total CB retrieve calls
    - ``lmcache_blend.retrieve_chunks``              — chunks requested for retrieval
    - ``lmcache_blend.retrieve_failures``            — retrieves with success=False
    - ``lmcache_blend.store_pre_computed_requests``  — total CB store_pre_computed calls
    - ``lmcache_blend.store_pre_computed_chunks``    — chunks via store_pre_computed
    - ``lmcache_blend.store_pre_computed_failures``  — store_pre_computed failures
    - ``lmcache_blend.store_final_requests``         — total CB store_final calls
    - ``lmcache_blend.store_final_chunks``           — chunks stored via store_final
    - ``lmcache_blend.store_final_failures``         — store_final failures
    - ``lmcache_blend.fingerprints_registered``      — chunks in fingerprint table
    - ``lmcache_blend.chunks_evicted``               — evicted from fingerprint table

    V3 phase metrics — the sub-legs of the unified lookup and the retrieve
    scatter. Each ``*_duration`` pairs the phase's START/END events by
    ``(session, worker_id)`` -- at TP>1 each rank's retrieve/scatter is its own
    sample -- and measures the interval the same-named trace span covers (see
    ``docs/design/v1/mp_observability/blend_v3_observability.md``):

    - ``lmcache_blend.lookup_duration``              — cb.lookup, incl. poll waits
    - ``lmcache_blend.fingerprint_match_duration``   — local fingerprint match
    - ``lmcache_blend.prefix_lookup_duration``       — prefix leg (L1+L2)
    - ``lmcache_blend.coordinator_match_duration``   — fleet-directory match leg
    - ``lmcache_blend.sparse_prefetch_duration``     — non-prefix L2->L1 prefetch
    - ``lmcache_blend.retrieve_duration``            — cb.retrieve (GPU-timed)
    - ``lmcache_blend.scatter_duration``             — L1->paged scatter (GPU-timed)
    - ``lmcache_blend.fingerprint_matches``          — chunks matched per lookup
    - ``lmcache_blend.coordinator_matches``          — segments from the coordinator
    - ``lmcache_blend.coordinator_match_timeouts``   — match legs past their deadline
    - ``lmcache_blend.sparse_prefetch_l2_keys``      — keys needing an L2 read
    - ``lmcache_blend.sparse_prefetch_found_keys``   — those keys that landed in L1
    - ``lmcache_blend.scatter_tokens``               — tokens written into paged KV
    - ``lmcache_blend.scatter_prefix_chunks``        — chunks written as-is
    - ``lmcache_blend.scatter_shifted_chunks``       — chunks re-RoPE'd on the way in
    - ``lmcache_blend.scatter_dropped_chunks``       — matches past the allocated slots
    - ``lmcache_blend.retrieve_noops``               — retrieves that scattered
      nothing, labeled by ``reason``
    """

    # Phase name per START / END event. Both directions must map to the same
    # name, which is the histogram suffix and the trace-span basename.
    _PHASE_BY_START: dict[EventType, str] = {
        EventType.CB_LOOKUP_START: "lookup",
        EventType.CB_FINGERPRINT_MATCH_START: "fingerprint_match",
        EventType.CB_PREFIX_LOOKUP_START: "prefix_lookup",
        EventType.CB_COORDINATOR_MATCH_START: "coordinator_match",
        EventType.CB_SPARSE_PREFETCH_START: "sparse_prefetch",
        EventType.CB_RETRIEVE_START: "retrieve",
        EventType.CB_SCATTER_START: "scatter",
    }

    _PHASE_BY_END: dict[EventType, str] = {
        EventType.CB_LOOKUP_END: "lookup",
        EventType.CB_FINGERPRINT_MATCH_END: "fingerprint_match",
        EventType.CB_PREFIX_LOOKUP_END: "prefix_lookup",
        EventType.CB_COORDINATOR_MATCH_END: "coordinator_match",
        EventType.CB_SPARSE_PREFETCH_END: "sparse_prefetch",
        EventType.CB_RETRIEVE_END: "retrieve",
        EventType.CB_SCATTER_END: "scatter",
    }

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.blend")

        # _PhaseKey -> START timestamp, bounded LRU (see _MAX_PENDING_PHASES).
        self._phase_starts: OrderedDict[_PhaseKey, float] = OrderedDict()
        # Set once the LRU has evicted a START; the warning fires once per
        # subscriber so a leak (or a too-small cap) is visible without flooding.
        self._phase_eviction_logged = False
        self._phase_hists: dict[str, Histogram] = {
            phase: meter.create_histogram(
                f"lmcache_blend.{phase}_duration",
                description=(
                    f"Wall-clock duration of the CB {phase} phase, measured "
                    "between its START and END events for one request."
                ),
                unit="ms",
            )
            for phase in dict.fromkeys(self._PHASE_BY_START.values())
        }

        self._lookup_requests = meter.create_counter(
            "lmcache_blend.lookup_requests",
            description="Total CB lookup requests",
        )
        self._lookup_requested_tokens = meter.create_counter(
            "lmcache_blend.lookup_requested_tokens",
            description=(
                "Total tokens submitted for CB lookup (denominator of the "
                "blend token-level hit rate). Only chunk-aligned tokens "
                "are counted."
            ),
            unit="tokens",
        )
        self._lookup_hit_tokens = meter.create_counter(
            "lmcache_blend.lookup_hit_tokens",
            description=(
                "Total tokens served by blend during lookup (numerator of "
                "the blend token-level hit rate). Equal to "
                "storage_hits * chunk_size."
            ),
            unit="tokens",
        )
        self._lookup_prefix_hit_tokens = meter.create_counter(
            "lmcache_blend.lookup_prefix_hit_tokens",
            description="Tokens served by blend from the prefix (L1+L2).",
            unit="tokens",
        )
        self._lookup_non_prefix_hit_tokens = meter.create_counter(
            "lmcache_blend.lookup_non_prefix_hit_tokens",
            description="Tokens served by blend from non-prefix (shifted) chunks.",
            unit="tokens",
        )
        self._lookup_segmented_prefix_hit_tokens = meter.create_counter(
            "lmcache_blend.lookup_segmented_prefix_hit_tokens",
            description=(
                "Tokens served by blend from the segmented-prefix tail "
                "(post-gap same-position chunks reused via the prefix leg)."
            ),
            unit="tokens",
        )
        self._lookup_fingerprint_hits = meter.create_counter(
            "lmcache_blend.lookup_fingerprint_hits",
            description="Chunks matched by local fingerprint table",
        )
        self._lookup_storage_hits = meter.create_counter(
            "lmcache_blend.lookup_storage_hits",
            description="Chunks confirmed present in storage after prefetch",
        )
        self._lookup_stale_chunks = meter.create_counter(
            "lmcache_blend.lookup_stale_chunks",
            description="Fingerprint hits evicted as stale (not in storage)",
        )
        self._lookup_no_gpu_ctx_errors = meter.create_counter(
            "lmcache_blend.lookup_no_gpu_context_errors",
            description="Lookup failures due to missing GPU context",
        )
        self._retrieve_requests = meter.create_counter(
            "lmcache_blend.retrieve_requests",
            description="Total CB retrieve requests",
        )
        self._retrieve_chunks = meter.create_counter(
            "lmcache_blend.retrieve_chunks",
            description="Total chunks requested for CB retrieval",
        )
        self._retrieve_failures = meter.create_counter(
            "lmcache_blend.retrieve_failures",
            description="CB retrieve operations that returned success=False",
        )
        self._store_pre_computed_requests = meter.create_counter(
            "lmcache_blend.store_pre_computed_requests",
            description="Total CB store_pre_computed requests",
        )
        self._store_pre_computed_chunks = meter.create_counter(
            "lmcache_blend.store_pre_computed_chunks",
            description="Chunks stored via CB store_pre_computed",
        )
        self._store_pre_computed_failures = meter.create_counter(
            "lmcache_blend.store_pre_computed_failures",
            description="CB store_pre_computed failures",
        )
        self._store_final_requests = meter.create_counter(
            "lmcache_blend.store_final_requests",
            description="Total CB store_final requests",
        )
        self._store_final_chunks = meter.create_counter(
            "lmcache_blend.store_final_chunks",
            description="Chunks stored via CB store_final",
        )
        self._store_final_failures = meter.create_counter(
            "lmcache_blend.store_final_failures",
            description="CB store_final failures",
        )
        self._fingerprints_registered = meter.create_counter(
            "lmcache_blend.fingerprints_registered",
            description="Chunks indexed into the fingerprint table",
        )
        self._chunks_evicted = meter.create_counter(
            "lmcache_blend.chunks_evicted",
            description="Stale chunks evicted from the fingerprint table",
        )
        self._fingerprint_matches = meter.create_counter(
            "lmcache_blend.fingerprint_matches",
            description="Chunks returned by the local fingerprint matcher",
        )
        self._coordinator_matches = meter.create_counter(
            "lmcache_blend.coordinator_matches",
            description="Segments returned by the coordinator fleet directory",
        )
        self._coordinator_match_timeouts = meter.create_counter(
            "lmcache_blend.coordinator_match_timeouts",
            description=(
                "Coordinator match legs abandoned at their deadline (the "
                "lookup proceeds local-only, so reuse silently shrinks)"
            ),
        )
        self._sparse_prefetch_l2_keys = meter.create_counter(
            "lmcache_blend.sparse_prefetch_l2_keys",
            description="Non-prefix keys the sparse prefetch had to read from L2",
        )
        self._sparse_prefetch_found_keys = meter.create_counter(
            "lmcache_blend.sparse_prefetch_found_keys",
            description="Sparse-prefetch keys that became L1-resident",
        )
        self._scatter_tokens = meter.create_counter(
            "lmcache_blend.scatter_tokens",
            description="Tokens written from L1 into the request's paged KV",
            unit="tokens",
        )
        self._scatter_prefix_chunks = meter.create_counter(
            "lmcache_blend.scatter_prefix_chunks",
            description="Scattered chunks kept at their stored position (no re-RoPE)",
        )
        self._scatter_shifted_chunks = meter.create_counter(
            "lmcache_blend.scatter_shifted_chunks",
            description="Scattered chunks re-RoPE'd to a new position",
        )
        self._scatter_dropped_chunks = meter.create_counter(
            "lmcache_blend.scatter_dropped_chunks",
            description="Matched chunks dropped for exceeding the allocated slots",
        )
        self._retrieve_noops = meter.create_counter(
            "lmcache_blend.retrieve_noops",
            description=(
                "CB retrieves that returned success without scattering "
                "anything, labeled by reason. Each one degrades its request "
                "to a full recompute."
            ),
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the mapping of event types to handler callbacks."""
        return {
            EventType.CB_LOOKUP_START: self._on_lookup_start,
            EventType.CB_LOOKUP_END: self._on_lookup_end,
            EventType.CB_RETRIEVE_START: self._on_retrieve_start,
            EventType.CB_RETRIEVE_END: self._on_retrieve_end,
            EventType.CB_RETRIEVE_NOOP: self._on_retrieve_noop,
            EventType.CB_STORE_PRE_COMPUTED_START: self._on_store_pre_start,
            EventType.CB_STORE_PRE_COMPUTED_END: self._on_store_pre_end,
            EventType.CB_STORE_FINAL_START: self._on_store_final_start,
            EventType.CB_STORE_FINAL_END: self._on_store_final_end,
            EventType.CB_FINGERPRINTS_REGISTERED: self._on_fingerprints_registered,
            EventType.CB_CHUNKS_EVICTED: self._on_chunks_evicted,
            EventType.CB_FINGERPRINT_MATCH_START: self._on_phase_start,
            EventType.CB_FINGERPRINT_MATCH_END: self._on_fingerprint_match_end,
            EventType.CB_PREFIX_LOOKUP_START: self._on_phase_start,
            EventType.CB_PREFIX_LOOKUP_END: self._on_phase_end,
            EventType.CB_COORDINATOR_MATCH_START: self._on_phase_start,
            EventType.CB_COORDINATOR_MATCH_END: self._on_coordinator_match_end,
            EventType.CB_SPARSE_PREFETCH_START: self._on_sparse_prefetch_start,
            EventType.CB_SPARSE_PREFETCH_END: self._on_sparse_prefetch_end,
            EventType.CB_SCATTER_START: self._on_scatter_start,
            EventType.CB_SCATTER_END: self._on_phase_end,
        }

    # ------------------------------------------------------------------
    # Phase duration pairing
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_key(phase: str, event: Event) -> _PhaseKey:
        """Build the START/END pairing key for ``event`` (see ``_PhaseKey``).

        Anything that is not an ``int`` counts as "no worker": V2 events and
        the lookup legs omit the field, and ``publish_on_stream``'s native
        recorder carries a ``None`` ``worker_id`` as the string ``"None"``.
        """
        worker_id = event.metadata.get("worker_id")
        return (
            phase,
            event.session_id,
            worker_id if isinstance(worker_id, int) else None,
        )

    def _on_phase_start(self, event: Event) -> None:
        """Stash a phase START timestamp for the matching END to consume."""
        phase = self._PHASE_BY_START[event.event_type]
        key = self._phase_key(phase, event)
        self._phase_starts[key] = event.timestamp
        self._phase_starts.move_to_end(key)
        while len(self._phase_starts) > _MAX_PENDING_PHASES:
            evicted, _ = self._phase_starts.popitem(last=False)
            if not self._phase_eviction_logged:
                self._phase_eviction_logged = True
                logger.warning(
                    "CB phase-duration pairing evicted an unmatched START %r: "
                    "more than %d STARTs are awaiting their END. Abandoned "
                    "lookups cause this legitimately; a steady stream means a "
                    "phase is not publishing its END. Logged once per subscriber.",
                    evicted,
                    _MAX_PENDING_PHASES,
                )

    def _on_phase_end(self, event: Event) -> None:
        """Record the phase duration in ms, if its START was seen."""
        phase = self._PHASE_BY_END[event.event_type]
        start_ts = self._phase_starts.pop(self._phase_key(phase, event), None)
        if start_ts is None:
            return  # no matching START (evicted, or the bus started mid-request)
        dt_ms = (event.timestamp - start_ts) * 1000.0
        if dt_ms < 0:
            # GPU-callback timestamps can invert against a CPU-timestamped pair.
            return
        self._phase_hists[phase].record(dt_ms)

    def _on_lookup_start(self, event: Event) -> None:
        self._lookup_requests.add(1)
        self._on_phase_start(event)

    def _on_lookup_end(self, event: Event) -> None:
        self._on_phase_end(event)
        self._lookup_requested_tokens.add(event.metadata["requested_tokens"])
        self._lookup_hit_tokens.add(event.metadata["hit_tokens"])
        self._lookup_prefix_hit_tokens.add(event.metadata.get("prefix_hit_tokens", 0))
        self._lookup_non_prefix_hit_tokens.add(
            event.metadata.get("non_prefix_hit_tokens", 0)
        )
        self._lookup_segmented_prefix_hit_tokens.add(
            event.metadata.get("segmented_prefix_hit_tokens", 0)
        )
        self._lookup_fingerprint_hits.add(event.metadata["fingerprint_hits"])
        self._lookup_storage_hits.add(event.metadata["storage_hits"])
        self._lookup_stale_chunks.add(event.metadata["stale_chunks"])
        if event.metadata.get("no_gpu_context"):
            self._lookup_no_gpu_ctx_errors.add(1)

    def _on_retrieve_start(self, event: Event) -> None:
        self._retrieve_requests.add(1)
        self._retrieve_chunks.add(event.metadata["num_chunks"])
        self._on_phase_start(event)

    def _on_retrieve_end(self, event: Event) -> None:
        self._on_phase_end(event)
        if not event.metadata.get("success", True):
            self._retrieve_failures.add(1)

    def _on_retrieve_noop(self, event: Event) -> None:
        reason = str(event.metadata.get("reason", "unknown"))
        self._retrieve_noops.add(1, attributes={"reason": reason})

    def _on_fingerprint_match_end(self, event: Event) -> None:
        self._on_phase_end(event)
        self._fingerprint_matches.add(event.metadata.get("matches", 0))

    def _on_coordinator_match_end(self, event: Event) -> None:
        self._on_phase_end(event)
        self._coordinator_matches.add(event.metadata.get("matches", 0))
        if event.metadata.get("timed_out"):
            self._coordinator_match_timeouts.add(1)

    def _on_sparse_prefetch_start(self, event: Event) -> None:
        self._on_phase_start(event)
        self._sparse_prefetch_l2_keys.add(event.metadata.get("l2_keys", 0))

    def _on_sparse_prefetch_end(self, event: Event) -> None:
        self._on_phase_end(event)
        self._sparse_prefetch_found_keys.add(event.metadata.get("found_keys", 0))

    def _on_scatter_start(self, event: Event) -> None:
        self._on_phase_start(event)
        self._scatter_tokens.add(event.metadata.get("scattered_tokens", 0))
        self._scatter_prefix_chunks.add(event.metadata.get("n_prefix", 0))
        self._scatter_shifted_chunks.add(event.metadata.get("n_shifted", 0))
        self._scatter_dropped_chunks.add(event.metadata.get("dropped", 0))

    def _on_store_pre_start(self, event: Event) -> None:
        self._store_pre_computed_requests.add(1)

    def _on_store_pre_end(self, event: Event) -> None:
        self._store_pre_computed_chunks.add(event.metadata["stored_chunks"])
        if not event.metadata.get("success", True):
            self._store_pre_computed_failures.add(1)

    def _on_store_final_start(self, event: Event) -> None:
        self._store_final_requests.add(1)

    def _on_store_final_end(self, event: Event) -> None:
        self._store_final_chunks.add(event.metadata["stored_chunks"])
        if not event.metadata.get("success", True):
            self._store_final_failures.add(1)

    def _on_fingerprints_registered(self, event: Event) -> None:
        self._fingerprints_registered.add(event.metadata["num_chunks"])

    def _on_chunks_evicted(self, event: Event) -> None:
        self._chunks_evicted.add(event.metadata["num_chunks"])
