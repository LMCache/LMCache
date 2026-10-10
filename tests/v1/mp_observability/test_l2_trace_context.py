# SPDX-License-Identifier: Apache-2.0
"""CPU tracing tests with real controller queues and the existing mock L2."""

# Standard
from unittest.mock import create_autospec
import asyncio
import select
import sys
import threading

# Third Party
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
    StoreListener,
)
from lmcache.v1.distributed.storage_controllers.store_policy import DefaultStorePolicy
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.propagation import (
    capture_trace_context,
    extract_trace_context,
    run_with_trace_links,
)

linux_controller = pytest.mark.skipif(
    sys.platform != "linux", reason="controllers use poll/eventfd"
)


def parent(number: int, sampled: bool = True) -> NonRecordingSpan:
    return NonRecordingSpan(
        SpanContext(number, number + 1, False, TraceFlags(int(sampled)))
    )


def key(number: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(number), model_name="test_model", kv_rank=0
    )


def memory_obj() -> TensorMemoryObj:
    data = torch.ones(4, dtype=torch.float32)
    metadata = MemoryObjMetadata(
        shape=torch.Size([4]),
        dtype=torch.float32,
        address=0,
        phy_size=16,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(data, metadata, parent_allocator=None)


def layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([4])], dtypes=[torch.float32])


def wait_fd(fd: int) -> bool:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(5000))


class ObservedL2(MockL2Adapter):
    """Observe public submissions; the upstream mock still completes its I/O."""

    def __init__(self, config: MockL2AdapterConfig) -> None:
        super().__init__(config)
        self.seen: list[tuple[str, int, bool]] = []
        self.stored = threading.Event()

    def observe(self, operation: str) -> None:
        ctx = trace.get_current_span().get_span_context()
        self.seen.append((operation, ctx.trace_id, ctx.trace_flags.sampled))

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_descs: dict[int, MemoryLayoutDesc]
    ) -> int:
        self.observe("lookup")
        return super().submit_lookup_and_lock_task(keys, layout_descs)

    def submit_load_task(self, keys: list[ObjectKey], objs: list[MemoryObj]) -> int:
        self.observe("load")
        return super().submit_load_task(keys, objs)

    def submit_store_task(self, keys: list[ObjectKey], objs: list[MemoryObj]) -> int:
        self.observe("store")
        result = super().submit_store_task(keys, objs)
        self.stored.set()
        return result


@linux_controller
@pytest.mark.parametrize("sampled", [True, False])
@pytest.mark.parametrize("fail_first_lookup", [False, True])
def test_prefetch_queue_keeps_each_parent(
    monkeypatch: pytest.MonkeyPatch, sampled: bool, fail_first_lookup: bool
) -> None:
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10)
    adapter = ObservedL2(config)
    keys = [key(1), key(2)]
    warm = adapter.submit_store_task(keys, [memory_obj(), memory_obj()])
    assert wait_fd(adapter.get_store_event_fd())
    assert warm in adapter.pop_completed_store_tasks()
    adapter.seen.clear()
    submit_lookup = adapter.submit_lookup_and_lock_task

    def lookup_with_failure(
        lookup_keys: list[ObjectKey], layout_descs: dict[int, MemoryLayoutDesc]
    ) -> int:
        if fail_first_lookup and lookup_keys == [keys[0]]:
            adapter.observe("lookup")
            raise ValueError("test lookup submission failure")
        return submit_lookup(lookup_keys, layout_descs)

    monkeypatch.setattr(adapter, "submit_lookup_and_lock_task", lookup_with_failure)
    # Exercise controller queues with CPU buffers; no accelerator allocation.
    l1 = create_autospec(L1Manager, instance=True)
    l1.reserve_read.side_effect = lambda keys, read_locks=1: {
        k: (L1Error.KEY_NOT_EXIST, None) for k in keys
    }
    l1.reserve_write.side_effect = lambda keys, **kwargs: {
        k: (L1Error.SUCCESS, memory_obj()) for k in keys
    }
    l1config = L1ManagerConfig(
        L1MemoryManagerConfig(
            size_in_bytes=4096, use_lazy=True, init_size_in_bytes=4096
        )
    )
    l1.config = l1config
    l1.l1_manager_id = 0
    controller = PrefetchController(
        [l1],
        [L1ManagerDescriptor(0, l1config)],
        [adapter],
        [L2AdapterDescriptor(0, config)],
        DefaultPrefetchPolicy(),
        max_in_flight=1,
    )
    controller.start()
    try:
        requests = []
        for number, objkey in enumerate(keys, 10):
            group = GroupedObjectKeys(
                keys=[objkey], object_group_id=0, layout_desc=layout()
            )
            with trace.use_span(parent(number, sampled)):
                requests.append(
                    controller.submit_prefetch_request(
                        PrefetchTaskSpec(key_groups=[group])
                    )
                )
        for index, request in enumerate(requests):
            assert controller.wait_prefetch_result(request, timeout=5)
            result = controller.query_prefetch_result(request)
            assert result is not None
            failed = fail_first_lookup and index == 0
            assert result.hit_cells[0].popcount() == (0 if failed else 1)
            assert result.l1_owners == ({} if failed else {keys[index]: 0})
        expected = [
            ("lookup", 10, sampled),
            ("lookup", 11, sampled),
            ("load", 11, sampled),
        ]
        if not fail_first_lookup:
            expected.append(("load", 10, sampled))
        assert sorted(adapter.seen) == sorted(expected)
    finally:
        controller.stop()
        adapter.close()


@linux_controller
def test_shared_store_links_writers_without_selecting_a_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("lmcache_mp.server")
    monkeypatch.setattr(trace, "get_tracer", lambda *args, **kwargs: tracer)
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10)
    adapter = ObservedL2(config)
    l1 = create_autospec(L1Manager, instance=True)
    l1.reserve_read.side_effect = lambda keys: {
        k: (L1Error.SUCCESS, memory_obj()) for k in keys
    }
    completed = threading.Event()
    l1.finish_read.side_effect = lambda *args, **kwargs: completed.set()
    controller = StoreController(
        l1, [adapter], [L2AdapterDescriptor(0, config)], DefaultStorePolicy()
    )
    listener = l1.register_listener.call_args.args[0]
    assert isinstance(listener, StoreListener)
    keys = [key(1), key(2)]
    for number, objkey in enumerate(keys, 10):
        with trace.use_span(parent(number)):
            listener.on_l1_keys_reserved_write([objkey])
    listener.on_l1_keys_write_finished(keys)
    controller.start()
    try:
        assert completed.wait(5)
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "mp.l2.store.schedule" and span.parent is None
        assert {link.context.trace_id for link in span.links} == {10, 11}
        assert adapter.seen == [("store", span.context.trace_id, True)]
        assert not span.attributes
    finally:
        controller.stop()
        adapter.close()
        provider.shutdown()


@pytest.mark.parametrize("sampled", [True, False])
def test_batch_link_bounds_and_sampling(
    monkeypatch: pytest.MonkeyPatch, sampled: bool
) -> None:
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        trace, "get_tracer", lambda *args, **kwargs: provider.get_tracer("test")
    )
    carriers = []
    for number in range(1, 151):
        with trace.use_span(parent(number, sampled)):
            carriers.append(capture_trace_context())
    try:
        assert run_with_trace_links(carriers + carriers, lambda: 7) == 7
        spans = exporter.get_finished_spans()
        if sampled:
            assert len(spans) == 1 and len(spans[0].links) == 128
        else:
            assert not spans
    finally:
        provider.shutdown()


def test_abandoned_writer_eviction_keeps_pending_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    listener = StoreListener()
    first, last = key(0), key(10001)
    try:
        with trace.use_span(parent(1)):
            listener.on_l1_keys_reserved_write([first])
        with trace.use_span(parent(2)):
            listener.on_l1_keys_reserved_write([key(i) for i in range(1, 10002)])
        listener.on_l1_keys_write_finished([first, last])
        keys, carriers = listener.pop_pending_batch()
        assert keys == [first, last]
        assert [
            trace.get_current_span(extract_trace_context(c)).get_span_context().trace_id
            for c in carriers
        ] == [2]
    finally:
        listener.close()


@pytest.mark.parametrize(
    "error", [None, ValueError, TimeoutError, asyncio.CancelledError]
)
def test_unsampled_batch_isolates_and_restores_context(
    monkeypatch: pytest.MonkeyPatch, error: type[BaseException] | None
) -> None:
    """Nested SDK spans keep the unsampled decision without choosing a writer."""
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with trace.use_span(parent(8, False)):
        carrier = capture_trace_context()

    def submit() -> int:
        batch = trace.get_current_span().get_span_context()
        assert batch.is_valid and not batch.trace_flags.sampled
        assert batch.trace_id not in {8, 99}
        assert baggage.get_baggage("payload") is None
        event = Event(EventType.L2_STORE_SUBMITTED)
        assert event.trace_context["traceparent"].endswith("-00")
        with tracer.start_as_current_span("backend.child") as child:
            assert not child.is_recording()
            if error is not None:
                raise error("synthetic private request content")
        return 7

    token = context.attach(baggage.set_baggage("payload", "synthetic"))
    try:
        with trace.use_span(parent(99)):
            if error is None:
                assert run_with_trace_links([carrier], submit) == 7
            else:
                with pytest.raises(error):
                    run_with_trace_links([carrier], submit)
            assert trace.get_current_span().get_span_context().trace_id == 99
            assert baggage.get_baggage("payload") == "synthetic"
        assert not exporter.get_finished_spans()
    finally:
        context.detach(token)
        provider.shutdown()


@pytest.mark.parametrize(
    "cleanup",
    ["on_l1_keys_deleted_by_manager", "on_l1_keys_finish_write_and_reserve_read"],
)
def test_shared_writer_bounds_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, cleanup: str
) -> None:
    """A shared key retains bounded contributors and clears abandoned writes."""
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    listener = StoreListener()
    shared = key(1)
    try:
        for number in range(1, 13):
            with trace.use_span(parent(number)):
                listener.on_l1_keys_reserved_write([shared])
        listener.on_l1_keys_write_finished([shared])
        keys, carriers = listener.pop_pending_batch()
        assert keys == [shared]
        assert [
            trace.get_current_span(extract_trace_context(c)).get_span_context().trace_id
            for c in carriers
        ] == list(range(1, 9))
        with trace.use_span(parent(99)):
            listener.on_l1_keys_reserved_write([shared])
        getattr(listener, cleanup)([shared])
        assert listener.pending_count() == 0
        listener.on_l1_keys_write_finished([shared])
        keys, carriers = listener.pop_pending_batch()
        assert keys == [shared] and not carriers
    finally:
        listener.close()


def test_opt_out_keeps_ambient_context_and_pending_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LMCACHE_MP_TRACE_CONTEXT", raising=False)
    listener = StoreListener()
    try:
        with trace.use_span(parent(99)):
            assert (
                run_with_trace_links(
                    [], lambda: trace.get_current_span().get_span_context().trace_id
                )
                == 99
            )
            listener.on_l1_keys_reserved_write([key(1)])
            listener.on_l1_keys_write_finished([key(1)])
        keys, carriers = listener.pop_pending_batch()
        assert keys == [key(1)] and not carriers
    finally:
        listener.close()


@pytest.mark.parametrize("error", [ValueError, TimeoutError, asyncio.CancelledError])
def test_failed_batch_ends_without_exporting_error_payload(
    monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        trace, "get_tracer", lambda *args, **kwargs: provider.get_tracer("test")
    )
    with trace.use_span(parent(8)):
        carrier = capture_trace_context()

    def fail() -> None:
        raise error("synthetic private request content")

    try:
        with trace.use_span(parent(99)):
            with pytest.raises(error):
                run_with_trace_links([carrier], fail)
            assert trace.get_current_span().get_span_context().trace_id == 99
        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == trace.StatusCode.ERROR
        assert not span.status.description and not span.attributes and not span.events
        assert span.end_time is not None
        assert run_with_trace_links([carrier], lambda: 3) == 3
    finally:
        provider.shutdown()
