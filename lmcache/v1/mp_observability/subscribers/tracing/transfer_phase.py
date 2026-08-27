# SPDX-License-Identifier: Apache-2.0

"""Gather/DMA phase breakdown of each transfer as child spans.

Consumes ``MP_TRANSFER_PHASE_SAMPLES`` and emits, per transfer, two children
under the request's ``mp.store`` / ``mp.retrieve`` span -- ``transfer.kernel``
and ``transfer.staging`` -- stacked back to back, each as long as that
phase's total busy time, with the phase totals as attributes.

Samples arrive after the parent span has ended (``TransferPhaseSampler``
pops them on ``MP_*_END``); a transfer is emitted at the first samples event
after its END. Design: ``docs/design/observability/request-event-span.md``
(Example 3).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import Any, Sequence
import time

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import SpanRegistry
from lmcache.v1.platform.ops_types import TransferPhase

logger = init_logger(__name__)

try:
    # Third Party
    from opentelemetry import trace

    _tracer = trace.get_tracer("lmcache_mp.transfer")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

# Parent logical span name (as registered by MPServerTracingSubscriber).
_PARENT_BY_DIRECTION: dict[int, str] = {
    int(TransferDirection.D2H): "store",
    int(TransferDirection.H2D): "retrieve",
}
_PARENT_BY_EVENT: dict[EventType, str] = {
    EventType.MP_STORE_START: "store",
    EventType.MP_STORE_END: "store",
    EventType.MP_RETRIEVE_START: "retrieve",
    EventType.MP_RETRIEVE_END: "retrieve",
}
_SPAN_NAME_BY_PHASE: dict[int, str] = {
    int(TransferPhase.KERNEL): "transfer.kernel",
    int(TransferPhase.STAGING): "transfer.staging",
}


@dataclass
class _PhaseTotals:
    """Running totals of one phase of one transfer."""

    first_start_s: float  # earliest section start (real time)
    last_end_s: float  # latest section end (real time)
    busy_s: float = 0.0  # sum of section durations (same stream: no overlap)
    nbytes: int = 0
    num_steps: int = 0


@dataclass
class _TransferTotals:
    """Running totals of one transfer, per phase."""

    session_id: str
    parent_ctx: Any
    device_index: int = -1  # set by the first sample
    phases: dict[int, _PhaseTotals] = field(default_factory=dict)
    ended: bool = False  # MP_*_END seen


class TransferPhaseTracingSubscriber(EventSubscriber):
    """Attaches a per-phase breakdown of each transfer to its request's span.

    Must be registered after ``MPServerTracingSubscriber`` on the same
    registry: it reads the ``store`` / ``retrieve`` span that subscriber
    opens on the same ``MP_*_START`` event.

    Args:
        registry: Shared span registry the MP server subscriber writes to.
        parent_ttl_s: How long a captured parent context (and its totals)
            is kept waiting for samples before it is dropped; normally it
            is released as soon as the transfer's spans are emitted.
    """

    def __init__(self, registry: SpanRegistry, parent_ttl_s: float = 120.0) -> None:
        self._registry = registry
        self._parent_ttl_s = parent_ttl_s
        # (session_id, parent logical name) -> (otel context, captured_at)
        self._parents: dict[tuple[str, str], tuple[Any, float]] = {}
        # (session_id, parent logical name) -> running totals
        self._transfers: dict[tuple[str, str], _TransferTotals] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_transfer_start,
            EventType.MP_RETRIEVE_START: self._on_transfer_start,
            EventType.MP_STORE_END: self._on_transfer_end,
            EventType.MP_RETRIEVE_END: self._on_transfer_end,
            EventType.MP_TRANSFER_PHASE_SAMPLES: self._on_samples,
        }

    def shutdown(self) -> None:
        for transfer in self._transfers.values():
            self._emit(transfer)
        self._transfers.clear()
        self._parents.clear()

    # -- Parent capture -------------------------------------------------------

    def _on_transfer_start(self, event: Event) -> None:
        """Capture the parent context; samples arrive after the span ends."""
        if not _HAS_OTEL or not event.session_id:
            return
        parent = _PARENT_BY_EVENT[event.event_type]
        ctx = self._registry.get_context(event.session_id, parent)
        if ctx is None:
            return
        self._parents[(event.session_id, parent)] = (ctx, time.monotonic())

    def _on_transfer_end(self, event: Event) -> None:
        """Mark the transfer complete (END is stream-published: all
        sections done)."""
        self._expire_parents()
        key = (event.session_id, _PARENT_BY_EVENT[event.event_type])
        transfer = self._transfers.get(key)
        if transfer is None:
            entry = self._parents.get(key)
            if entry is None:
                return
            transfer = _TransferTotals(session_id=key[0], parent_ctx=entry[0])
            self._transfers[key] = transfer
        transfer.ended = True

    # -- Sample accumulation --------------------------------------------------

    def _on_samples(self, event: Event) -> None:
        if not _HAS_OTEL:
            return
        self._expire_parents()
        for sample in event.metadata.get("samples", ()):
            self._accumulate(sample)
        # END is stream-published, so a samples event queued after it holds
        # everything the transfer had left: emit ended transfers now.
        ended = [k for k, t in self._transfers.items() if t.ended]
        for key in ended:
            self._emit(self._transfers.pop(key))
            self._parents.pop(key, None)

    def _accumulate(self, sample: Sequence[Any]) -> None:
        """Fold one sample into its transfer's per-phase totals.

        Skips silently: samples without wall-clock bounds (anchor failed),
        samples without a captured parent, malformed samples.
        """
        if len(sample) != 8:
            return
        phase, direction, device_index, elapsed_ms, nbytes, session_id, t0, t1 = sample
        if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
            return
        if not isinstance(elapsed_ms, (int, float)) or not isinstance(nbytes, int):
            return
        if t0 <= 0 or t1 < t0:
            return
        try:
            parent = _PARENT_BY_DIRECTION.get(direction)
            known_phase = phase in _SPAN_NAME_BY_PHASE
        except TypeError:  # unhashable field
            return
        if parent is None or not known_phase:
            return
        sid = str(session_id)
        entry = self._parents.get((sid, parent))
        if entry is None:
            return
        parent_ctx, _ = entry
        transfer = self._transfers.get((sid, parent))
        if transfer is None:
            transfer = _TransferTotals(session_id=sid, parent_ctx=parent_ctx)
            self._transfers[(sid, parent)] = transfer
        totals = transfer.phases.get(int(phase))
        if totals is None:
            totals = _PhaseTotals(first_start_s=t0, last_end_s=t1)
            transfer.phases[int(phase)] = totals
        totals.first_start_s = min(totals.first_start_s, t0)
        totals.last_end_s = max(totals.last_end_s, t1)
        totals.busy_s += elapsed_ms / 1e3
        totals.nbytes += nbytes
        totals.num_steps += 1
        transfer.device_index = int(device_index)

    def _emit(self, transfer: _TransferTotals) -> None:
        """Emit the stacked phase spans of one transfer."""
        if not transfer.phases:
            return
        # Stacked in execution order from the transfer's real start.
        ordered = sorted(transfer.phases.items(), key=lambda kv: kv[1].first_start_s)
        cursor_s = ordered[0][1].first_start_s
        for phase, totals in ordered:
            span = _tracer.start_span(
                _SPAN_NAME_BY_PHASE[phase],
                context=transfer.parent_ctx,
                start_time=int(cursor_s * 1e9),
            )
            span.set_attribute("session_id", transfer.session_id)
            span.set_attribute("device_index", transfer.device_index)
            span.set_attribute("num_steps", totals.num_steps)
            span.set_attribute("nbytes", totals.nbytes)
            span.set_attribute("busy_seconds", totals.busy_s)
            span.set_attribute("first_start_s", totals.first_start_s)
            span.set_attribute("last_end_s", totals.last_end_s)
            if totals.busy_s > 0:
                span.set_attribute(
                    "throughput_GB_per_second",
                    totals.nbytes / totals.busy_s / 1e9,
                )
            cursor_s += totals.busy_s
            span.end(end_time=int(cursor_s * 1e9))

    def _expire_parents(self) -> None:
        """Drop parent contexts (and their totals) older than the TTL."""
        cutoff = time.monotonic() - self._parent_ttl_s
        stale = [k for k, (_, at) in self._parents.items() if at < cutoff]
        for k in stale:
            del self._parents[k]
            self._transfers.pop(k, None)
