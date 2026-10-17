# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for bounded store attribution and scheduling context."""

# Standard
from collections.abc import Iterator
from queue import Queue
from unittest.mock import create_autospec
import asyncio
import subprocess
import sys

# Third Party
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
    StoreListener,
)
from lmcache.v1.distributed.storage_controllers.store_policy import DefaultStorePolicy
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.mp_observability import propagation, store_tracing
from lmcache.v1.mp_observability.propagation import capture_trace_context
from lmcache.v1.mp_observability.store_tracing import (
    StoreTraceContributors,
    run_with_store_trace,
)
import lmcache.v1.distributed.storage_controllers.store_controller as store_module

TraceSetup = tuple[TracerProvider, InMemorySpanExporter]


def parent(number: int, sampled: bool = True) -> NonRecordingSpan:
    """Create a deterministic contributor context."""
    return NonRecordingSpan(
        SpanContext(number, number + 1, False, TraceFlags(int(sampled)))
    )


def key(number: int) -> ObjectKey:
    """Create a CPU cache key."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(number), model_name="test", kv_rank=0
    )


def carrier(number: int, sampled: bool = True) -> dict[str, str]:
    """Capture one contributor without retaining ambient context."""
    with trace.use_span(parent(number, sampled)):
        return capture_trace_context()


@pytest.fixture
def tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """Use an isolated SDK without replacing the process's tracer provider."""
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(trace, "get_tracer", lambda *args, **kwargs: tracer)
    yield provider, exporter
    provider.shutdown()


@pytest.mark.parametrize("sampled_at", [1, 129, None])
def test_sampling_summary_survives_link_cap(
    tracing: TraceSetup, sampled_at: int | None
) -> None:
    """A sampled contributor beyond 128 retained links still reaches the sampler."""
    provider, exporter = tracing
    contributors = StoreTraceContributors()
    for number in range(1, 130):
        contributors.add(carrier(number, number == sampled_at))
    token = context.attach(baggage.set_baggage("private", "payload"))
    try:
        with trace.use_span(parent(999)):

            def schedule() -> int:
                assert baggage.get_baggage("private") is None
                current = trace.get_current_span().get_span_context()
                assert current.is_valid and current.trace_id != 999
                with provider.get_tracer("child").start_as_current_span("adapter"):
                    pass
                return 7

            assert run_with_store_trace(contributors, schedule) == 7
            assert trace.get_current_span().get_span_context().trace_id == 999
            assert baggage.get_baggage("private") == "payload"
    finally:
        context.detach(token)
    spans = exporter.get_finished_spans()
    if sampled_at is None:
        assert spans == ()
    else:
        batch = next(span for span in spans if span.name == "mp.l2.store.schedule")
        assert batch.parent is None and len(batch.links) == 128
        assert not batch.attributes
        child = next(span for span in spans if span.name == "adapter")
        assert child.parent.span_id == batch.context.span_id


@pytest.mark.parametrize("unknown_first", [True, False])
def test_unknown_contributor_uses_existing_root_sampler(
    tracing: TraceSetup, unknown_first: bool
) -> None:
    contributors = StoreTraceContributors()
    for item in (
        [None, carrier(1, False)] if unknown_first else [carrier(1, False), None]
    ):
        contributors.add(item)
    run_with_store_trace(contributors, lambda: None)
    assert len(tracing[1].get_finished_spans()) == 1


@pytest.mark.parametrize("error", [ValueError, asyncio.CancelledError])
def test_failure_restores_context_without_exception_payload(
    tracing: TraceSetup, error: type[BaseException]
) -> None:
    contributors = StoreTraceContributors()
    contributors.add(carrier(1))

    def fail() -> None:
        raise error("secret request payload")

    with trace.use_span(parent(999)):
        with pytest.raises(error):
            run_with_store_trace(contributors, fail)
        assert trace.get_current_span().get_span_context().trace_id == 999
    span = tracing[1].get_finished_spans()[0]
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description is None and not span.events and not span.attributes
    assert run_with_store_trace(contributors, lambda: 3) == 3


def test_disabled_preserves_ambient_context(
    tracing: TraceSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LMCACHE_MP_TRACE_CONTEXT")
    with trace.use_span(parent(999)):
        assert (
            run_with_store_trace(
                StoreTraceContributors(),
                lambda: trace.get_current_span().get_span_context().trace_id,
            )
            == 999
        )
    assert tracing[1].get_finished_spans() == ()


@pytest.mark.parametrize("sampled_at", [1, 9, None])
def test_listener_per_key_bound_keeps_all_sampling_flags(
    tracing: TraceSetup, sampled_at: int | None
) -> None:
    listener = StoreListener()
    try:
        for number in range(1, 10):
            with trace.use_span(parent(number, number == sampled_at)):
                listener.on_l1_keys_reserved_write([key(1)])
        listener.on_l1_keys_write_finished([key(1)])
        keys, contributors = listener.pop_pending_batch()
        assert keys == [key(1)] and listener.pending_count() == 0
        run_with_store_trace(contributors, lambda: None)
        spans = tracing[1].get_finished_spans()
        if sampled_at is None:
            assert spans == ()
        else:
            assert len(spans) == 1 and len(spans[0].links) == 8
    finally:
        listener.close()


def test_listener_eviction_and_cleanup_preserve_keys(
    tracing: TraceSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_module, "MAX_TRACKED_WRITER_KEYS", 1)
    listener = StoreListener()
    try:
        with trace.use_span(parent(1)):
            listener.on_l1_keys_reserved_write([key(1)])
        with trace.use_span(parent(2, False)):
            listener.on_l1_keys_reserved_write([key(2)])
        with trace.use_span(parent(999)):
            listener.on_l1_keys_write_finished([key(1), key(2)])
        keys, contributors = listener.pop_pending_batch()
        assert keys == [key(1), key(2)]
        run_with_store_trace(contributors, lambda: None)
        span = tracing[1].get_finished_spans()[0]
        assert {link.context.trace_id for link in span.links} == {2}
        for cleanup in [
            listener.on_l1_keys_deleted_by_manager,
            listener.on_l1_keys_finish_write_and_reserve_read,
        ]:
            with trace.use_span(parent(3)):
                listener.on_l1_keys_reserved_write([key(3)])
            cleanup([key(3)])
            assert listener.pending_count() == 0
            listener.on_l1_keys_write_finished([key(3)])
            keys, contributors = listener.pop_pending_batch()
            assert (
                keys == [key(3)]
                and not contributors.contexts
                and contributors.has_unknown
            )
        listener.on_l1_keys_write_finished([key(4)])
        assert listener.pop_pending_keys() == [key(4)]
        assert listener.pop_pending_keys() == []
    finally:
        listener.close()


@pytest.mark.skipif(sys.platform != "linux", reason="controllers use poll/eventfd")
@pytest.mark.parametrize("adapter_count", [1, 2])
def test_controller_caught_failure_marks_span_and_continues(
    tracing: TraceSetup, monkeypatch: pytest.MonkeyPatch, adapter_count: int
) -> None:
    """Failed submissions release locks; other adapters and later batches continue."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10)
    adapters = [MockL2Adapter(config) for _ in range(adapter_count)]
    adapter = adapters[0]
    data = torch.ones(4, dtype=torch.float32)
    obj = TensorMemoryObj(
        data,
        MemoryObjMetadata(
            shape=torch.Size([4]),
            dtype=torch.float32,
            address=0,
            phy_size=16,
            fmt=MemoryFormat.KV_2LTD,
            ref_count=1,
        ),
        parent_allocator=None,
    )
    l1 = create_autospec(L1Manager, instance=True)
    l1.reserve_read.side_effect = lambda keys: {k: (L1Error.SUCCESS, obj) for k in keys}
    completed: Queue[list[ObjectKey]] = Queue()
    l1.finish_read.side_effect = lambda keys: completed.put(keys)
    original_submit = adapter.submit_store_task

    def submit(keys: list[ObjectKey], objs: list[MemoryObj]) -> int:
        if keys == [key(1)]:
            raise ValueError("secret adapter failure")
        return original_submit(keys, objs)

    monkeypatch.setattr(adapter, "submit_store_task", submit)
    controller = StoreController(
        l1,
        [adapter for adapter in adapters],
        [L2AdapterDescriptor(i, config) for i in range(adapter_count)],
        DefaultStorePolicy(),
    )
    listener = l1.register_listener.call_args.args[0]
    controller.start()
    try:
        for number in [1, 2]:
            with trace.use_span(parent(number)):
                listener.on_l1_keys_reserved_write([key(number)])
            listener.on_l1_keys_write_finished([key(number)])
            for _ in adapters:
                assert completed.get(timeout=5) == [key(number)]
            if number == 2:
                assert all(a.debug_has_key(key(2)) for a in adapters)
    finally:
        controller.stop()
        for a in adapters:
            a.close()
    spans = tracing[1].get_finished_spans()
    assert len(spans) == 2
    assert spans[0].status.status_code == trace.StatusCode.ERROR
    assert spans[1].status.status_code == trace.StatusCode.UNSET
    assert [{link.context.trace_id for link in s.links} for s in spans] == [{1}, {2}]
    assert all(
        s.parent is None
        and not s.events
        and not s.attributes
        and s.status.description is None
        for s in spans
    )
    assert [call.args[0] for call in l1.finish_read.call_args_list] == (
        [[key(1)]] * adapter_count + [[key(2)]] * adapter_count
    )


def test_evicted_key_rereservation_remains_unknown(
    tracing: TraceSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An evicted sampled writer cannot become an unsampled-only batch on reuse."""
    monkeypatch.setattr(store_module, "MAX_TRACKED_WRITER_KEYS", 1)
    listener = StoreListener()
    try:
        for number, objkey, sampled in [
            (1, key(1), True),
            (2, key(2), False),
            (3, key(1), False),
        ]:
            with trace.use_span(parent(number, sampled)):
                listener.on_l1_keys_reserved_write([objkey])
        listener.on_l1_keys_write_finished([key(1)])
        keys, contributors = listener.pop_pending_batch()
        assert keys == [key(1)] and contributors.has_unknown
        run_with_store_trace(contributors, lambda: None)
        assert len(tracing[1].get_finished_spans()) == 1
    finally:
        listener.close()


def test_store_tracing_without_otel_api() -> None:
    """The optional module preserves handler behavior in a stdlib-only process."""
    script = """
import importlib.util
import os
import sys
import types
for name in ("lmcache", "lmcache.v1", "lmcache.v1.mp_observability"):
    sys.modules[name] = types.ModuleType(name)
for name, path in zip(("propagation", "store_tracing"), sys.argv[1:]):
    fullname = "lmcache.v1.mp_observability." + name
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
os.environ["LMCACHE_MP_TRACE_CONTEXT"] = "1"
contributors = module.StoreTraceContributors()
contributors.add(None)
module.mark_store_trace_error()
assert module.run_with_store_trace(contributors, lambda value: value + 1, 4) == 5
try:
    module.run_with_store_trace(contributors, lambda: 1 / 0)
except ZeroDivisionError:
    pass
else:
    raise AssertionError("handler exception was swallowed")
"""
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            script,
            propagation.__file__,
            store_tracing.__file__,
        ],
        check=True,
    )


def test_listener_batch_merge_keeps_late_sampled_writer(tracing: TraceSetup) -> None:
    """The 129th key affects sampling even after the batch's link list fills."""
    listener = StoreListener()
    try:
        keys = [key(number) for number in range(1, 130)]
        for number, objkey in enumerate(keys, 1):
            with trace.use_span(parent(number, number == 129)):
                listener.on_l1_keys_reserved_write([objkey])
        listener.on_l1_keys_write_finished(keys)
        pending, contributors = listener.pop_pending_batch()
        assert pending == keys
        run_with_store_trace(contributors, lambda: None)
        span = tracing[1].get_finished_spans()[0]
        assert len(span.links) == 128 and span.parent is None
    finally:
        listener.close()
