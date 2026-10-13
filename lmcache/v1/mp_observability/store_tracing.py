# SPDX-License-Identifier: Apache-2.0
"""Bounded contributor links for shared L2 store scheduling batches."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar
import os
import secrets

try:
    # Third Party
    from opentelemetry import context, trace
    from opentelemetry.context import Context
    from opentelemetry.trace import Link, NonRecordingSpan, SpanContext, TraceFlags
except ImportError:
    _OTEL_AVAILABLE = False
else:
    _OTEL_AVAILABLE = True

# First Party
from lmcache.v1.mp_observability.propagation import extract_trace_context

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class StoreTraceContributors:
    """Bound links while retaining sampling decisions from every contributor.

    Args:
        max_links: Maximum distinct span contexts retained for links.

    Missing or invalid contexts mark the attribution as incomplete. Sampling
    summaries remain accurate even when valid contexts exceed the link bound.
    Instances are owned by a caller; shared instances require external locking.
    """

    max_links: int = 128
    contexts: dict[tuple[int, int], SpanContext] = field(
        default_factory=dict, repr=False
    )
    has_valid: bool = False
    has_sampled: bool = False
    has_unknown: bool = False

    def add(self, carrier: Mapping[str, str] | None) -> None:
        """Include one contributor's sampling decision and, if space permits, link.

        Args:
            carrier: W3C headers captured at reservation, or None if unavailable.
        """
        if not _OTEL_AVAILABLE:
            return
        span = trace.get_current_span(extract_trace_context(carrier)).get_span_context()
        if not span.is_valid:
            self.has_unknown = True
            return
        self.has_valid = True
        self.has_sampled |= span.trace_flags.sampled
        if len(self.contexts) < self.max_links:
            self.contexts[(span.trace_id, span.span_id)] = span

    def merge(self, other: StoreTraceContributors) -> None:
        """Merge all sampling flags and as many distinct links as fit.

        Args:
            other: Contributor summary to merge without modifying it.
        """
        self.has_valid |= other.has_valid
        self.has_sampled |= other.has_sampled
        self.has_unknown |= other.has_unknown
        for identity, span in other.contexts.items():
            if len(self.contexts) >= self.max_links:
                break
            self.contexts[identity] = span


def mark_store_trace_error() -> None:
    """Mark the active store scheduling span as failed without exception data.

    Does nothing when propagation is disabled or the OTel API is unavailable.
    """
    if _OTEL_AVAILABLE and os.environ.get("LMCACHE_MP_TRACE_CONTEXT") == "1":
        trace.get_current_span().set_status(trace.StatusCode.ERROR)


def run_with_store_trace(
    contributors: StoreTraceContributors,
    handler: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run scheduling in an independent context linked to observed writers.

    An entirely known, unsampled batch gets an independent non-recording
    context. Other batches use the existing root sampler. No provider or
    exporter is created. Ambient baggage is cleared for every enabled batch.

    Args:
        contributors: Bounded links plus complete sampling summary.
        handler: Synchronous scheduling/submission function.
        args: Positional arguments forwarded to the handler.
        kwargs: Keyword arguments forwarded to the handler.

    Returns:
        The handler's original result.

    Raises:
        BaseException: Any handler exception, unchanged and without exporting
            its text, traceback, or attributes to the scheduling span.
    """
    if not _OTEL_AVAILABLE or os.environ.get("LMCACHE_MP_TRACE_CONTEXT") != "1":
        return handler(*args, **kwargs)
    token = context.attach(Context())
    try:
        if (
            contributors.has_valid
            and not contributors.has_sampled
            and not contributors.has_unknown
        ):
            parent = NonRecordingSpan(
                SpanContext(
                    secrets.randbits(128) or 1,
                    secrets.randbits(64) or 1,
                    False,
                    TraceFlags(0),
                )
            )
            with trace.use_span(parent):
                return handler(*args, **kwargs)
        with trace.get_tracer("lmcache_mp.server").start_as_current_span(
            "mp.l2.store.schedule",
            links=[Link(span) for span in contributors.contexts.values()],
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                return handler(*args, **kwargs)
            except BaseException:
                span.set_status(trace.StatusCode.ERROR)
                raise
    finally:
        context.detach(token)
