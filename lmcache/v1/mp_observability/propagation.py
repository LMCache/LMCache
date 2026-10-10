# SPDX-License-Identifier: Apache-2.0
"""Opt-in W3C context propagation without configuring another tracer provider.

Set ``LMCACHE_MP_TRACE_CONTEXT=1`` in both the caller and MP server processes.
Configure the server's existing ``--enable-tracing`` and ``--otlp-endpoint``
options separately. The caller must have an active OTel span when submitting
a ZMQ request containing ``IPCCacheServerKey`` (lookup, store or retrieve).

The key carries optional headers to the worker; an Event snapshots them before
the EventBus thread creates the existing request span. Old key maps decode with
no headers, and old msgspec key decoders ignore the additional map field.
Disable the environment switch to stop injecting and honoring remote parents.
The existing provider's sampler still decides whether a span is recorded.

CPU submission events retain the parent before asynchronous GPU callbacks.
gRPC carries the same headers in per-call metadata, including keyless control
RPCs. L2 prefetch queues retain per-request snapshots; shared store batches
link their writers during scheduling. Native storage queues remain separate.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Mapping
from typing import ParamSpec, TypeVar
import os
import secrets

# Third Party
from opentelemetry import context, trace
from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

P = ParamSpec("P")
T = TypeVar("T")


def capture_trace_context() -> dict[str, str]:
    """Return W3C trace headers when LMCACHE_MP_TRACE_CONTEXT=1, else empty.

    Only traceparent and tracestate are carried; baggage and request payloads
    are never included. No SDK or exporter is installed by this function.

    Returns:
        W3C headers, or an empty dictionary when disabled.
    """
    if os.environ.get("LMCACHE_MP_TRACE_CONTEXT") != "1":
        return {}
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def extract_trace_context(carrier: Mapping[str, str] | None) -> Context:
    """Extract an isolated OTel context, ignoring invalid or oversized headers.

    Returns an empty context when propagation is disabled or headers are
    absent. The OpenTelemetry API is an existing LMCache dependency.

    Args:
        carrier: Optional W3C headers captured in the submitting process.

    Returns:
        A remote parent context, or an empty context for invalid input.
    """
    headers: dict[str, str] = {}
    if os.environ.get("LMCACHE_MP_TRACE_CONTEXT") == "1" and carrier:
        for name, limit in (("traceparent", 512), ("tracestate", 512)):
            value = carrier.get(name)
            if isinstance(value, str) and len(value) <= limit:
                headers[name] = value
    return TraceContextTextMapPropagator().extract(headers, context=Context())


def run_with_trace_context(
    carrier: Mapping[str, str] | None,
    handler: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run a handler under an isolated context and restore it even on failure.

    The caller must invoke this inside the executing worker, not the submitting
    thread. Exceptions, including cancellation, propagate unchanged.

    Args:
        carrier: Optional W3C headers captured in the submitting process.
        handler: Synchronous request handler to execute.
        args: Positional arguments forwarded to the handler.
        kwargs: Keyword arguments forwarded to the handler.

    Returns:
        The handler's original result.

    Raises:
        BaseException: Any exception raised by the handler, unchanged.
    """
    if os.environ.get("LMCACHE_MP_TRACE_CONTEXT") != "1":
        return handler(*args, **kwargs)
    token = context.attach(extract_trace_context(carrier))
    try:
        return handler(*args, **kwargs)
    finally:
        context.detach(token)


def run_with_trace_links(
    carriers: list[dict[str, str]],
    handler: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Schedule a shared store batch under a root span linked to its writers.

    A batch is never assigned to whichever writer happened to finish last.
    Only valid, distinct W3C parents become links. No keys or error payloads
    are exported. The span measures scheduling, not the asynchronous I/O.

    Args:
        carriers: Writer contexts collected for the shared batch.
        handler: Synchronous batch submission handler.
        args: Positional arguments forwarded to the handler.
        kwargs: Keyword arguments forwarded to the handler.

    Returns:
        The handler's original result.

    Raises:
        BaseException: Any exception raised by the handler, unchanged.
    """
    if os.environ.get("LMCACHE_MP_TRACE_CONTEXT") != "1":
        return run_with_trace_context({}, handler, *args, **kwargs)
    parents: dict[tuple[int, int], SpanContext] = {}
    for carrier in carriers:
        parent = trace.get_current_span(
            extract_trace_context(carrier)
        ).get_span_context()
        if parent.is_valid:
            parents[(parent.trace_id, parent.span_id)] = parent
            if len(parents) >= 128:
                break
    # A linked batch must not turn an entirely unsampled workload into a
    # new sampled root merely because the provider's root sampler is AlwaysOn.
    if parents and not any(parent.trace_flags.sampled for parent in parents.values()):
        # Preserve the unsampled decision for downstream ParentBased tracers
        # without choosing a contributing writer as the shared batch's parent.
        batch = NonRecordingSpan(
            SpanContext(
                secrets.randbits(128) or 1,
                secrets.randbits(64) or 1,
                is_remote=False,
                trace_flags=TraceFlags(0),
            )
        )
        token = context.attach(trace.set_span_in_context(batch, Context()))
        try:
            return handler(*args, **kwargs)
        finally:
            context.detach(token)
    if not parents:
        return run_with_trace_context({}, handler, *args, **kwargs)
    tracer = trace.get_tracer("lmcache_mp.server")
    with tracer.start_as_current_span(
        "mp.l2.store.schedule",
        context=Context(),
        links=[trace.Link(parent) for parent in parents.values()],
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            return handler(*args, **kwargs)
        except BaseException:
            # Exception text can contain cache keys or request content.
            span.set_status(trace.StatusCode.ERROR)
            raise
