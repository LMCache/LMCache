# SPDX-License-Identifier: Apache-2.0
"""CPU tracing tests with real controller queues and the existing mock L2."""

# Standard
from unittest.mock import create_autospec
import select
import sys

# Third Party
from opentelemetry import trace
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

pytestmark = pytest.mark.skipif(
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

    def __init__(
        self, config: MockL2AdapterConfig, fail_operation: str | None = None
    ) -> None:
        super().__init__(config)
        self.seen: list[tuple[str, int, bool]] = []
        self.fail_operation = fail_operation
        self.failed = False

    def observe(self, operation: str) -> None:
        """Record each caller and optionally fail request 10 at an I/O boundary."""
        ctx = trace.get_current_span().get_span_context()
        self.seen.append((operation, ctx.trace_id, ctx.trace_flags.sampled))
        if operation == self.fail_operation and ctx.trace_id == 10:
            self.failed = True
            raise ValueError("test adapter submission failure")

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
        return result


@pytest.mark.parametrize("sampled", [True, False])
@pytest.mark.parametrize("fail_operation", [None, "lookup", "load"])
def test_prefetch_queue_keeps_each_parent(
    monkeypatch: pytest.MonkeyPatch, sampled: bool, fail_operation: str | None
) -> None:
    """A failed lookup/load must not contaminate the next queued request."""
    monkeypatch.setenv("LMCACHE_MP_TRACE_CONTEXT", "1")
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10)
    adapter = ObservedL2(config, fail_operation)
    keys = [key(1), key(2)]
    warm = adapter.submit_store_task(keys, [memory_obj(), memory_obj()])
    assert wait_fd(adapter.get_store_event_fd())
    assert warm in adapter.pop_completed_store_tasks()
    adapter.seen.clear()
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
    controller = PrefetchController(
        [l1],
        [L1ManagerDescriptor(0, l1config)],
        [adapter],
        [L2AdapterDescriptor(0, config)],
        DefaultPrefetchPolicy(),
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
        successful_requests = requests if fail_operation is None else requests[1:]
        for request in successful_requests:
            assert controller.wait_prefetch_result(request, timeout=5)
            assert controller.query_prefetch_result(request) is not None
        assert adapter.failed == (fail_operation is not None)
        expected = [
            ("lookup", 10, sampled),
            ("lookup", 11, sampled),
            ("load", 11, sampled),
        ]
        if fail_operation != "lookup":
            expected.append(("load", 10, sampled))
        assert sorted(adapter.seen) == sorted(expected)
    finally:
        controller.stop()
        adapter.close()
