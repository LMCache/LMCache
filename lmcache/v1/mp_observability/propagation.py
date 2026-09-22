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

This boundary does not cover gRPC, blending, keyless control RPCs, GPU callback
context capture, or the native Mooncake storage adapter's own task queues.
"""

# Standard
from collections.abc import Callable, Mapping
from typing import Any, TypeVar
import os

T = TypeVar("T")


def capture_trace_context() -> dict[str, str]:
    """Return W3C trace headers when LMCACHE_MP_TRACE_CONTEXT=1, else empty.

    Only traceparent and tracestate are carried; baggage and request payloads
    are never included. No SDK or exporter is installed by this function.
    """
    if os.environ.get("LMCACHE_MP_TRACE_CONTEXT") != "1":
        return {}
    try:
        # Third Party
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError:
        return {}
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def extract_trace_context(carrier: Mapping[str, str] | None) -> Any:
    """Extract an isolated OTel context, ignoring invalid or oversized headers.

    Returns an empty context when propagation is disabled or headers are
    absent. Requires the OTel API; callers without OTel should skip this call.
    """
    # Third Party
    from opentelemetry.context import Context
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    headers: dict[str, str] = {}
    if os.environ.get("LMCACHE_MP_TRACE_CONTEXT") == "1" and carrier:
        for name, limit in (("traceparent", 512), ("tracestate", 512)):
            value = carrier.get(name)
            if isinstance(value, str) and len(value) <= limit:
                headers[name] = value
    return TraceContextTextMapPropagator().extract(headers, context=Context())


def run_with_trace_context(
    carrier: Mapping[str, str] | None, handler: Callable[..., T], *args: Any
) -> T:
    """Run a handler under an isolated context and restore it even on failure.

    The caller must invoke this inside the executing worker, not the submitting
    thread. Exceptions, including cancellation, propagate unchanged.
    """
    try:
        # Third Party
        from opentelemetry import context
    except ImportError:
        return handler(*args)
    token = context.attach(extract_trace_context(carrier))
    try:
        return handler(*args)
    finally:
        context.detach(token)
