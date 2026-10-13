# SPDX-License-Identifier: Apache-2.0
"""CPU tracing tests with real controller queues and the existing mock L2."""

# Standard
from unittest.mock import create_autospec
import select
import sys
import threading

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
@pytest.mark.parametrize("failure", [None, "lookup", "load"])
def test_prefetch_queue_keeps_each_parent(
    monkeypatch: pytest.MonkeyPatch, sampled: bool, failure: str | None
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
        if failure == "lookup" and lookup_keys == [keys[0]]:
            adapter.observe("lookup")
            raise ValueError("test lookup submission failure")
        return submit_lookup(lookup_keys, layout_descs)

    monkeypatch.setattr(adapter, "submit_lookup_and_lock_task", lookup_with_failure)
    submit_load = adapter.submit_load_task

    def load_with_failure(load_keys: list[ObjectKey], objs: list[MemoryObj]) -> int:
        if failure == "load" and load_keys == [keys[0]]:
            adapter.observe("load")
            raise ValueError("test load submission failure")
        return submit_load(load_keys, objs)

    monkeypatch.setattr(adapter, "submit_load_task", load_with_failure)
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
            with trace.use_span(
                parent(number, sampled if number == 10 else not sampled)
            ):
                requests.append(
                    controller.submit_prefetch_request(
                        PrefetchTaskSpec(key_groups=[group])
                    )
                )
        for index, request in enumerate(requests):
            assert controller.wait_prefetch_result(request, timeout=5)
            result = controller.query_prefetch_result(request)
            assert result is not None
            failed = failure is not None and index == 0
            assert result.hit_cells[0].popcount() == (0 if failed else 1)
            assert result.l1_owners == ({} if failed else {keys[index]: 0})
        expected = [
            ("lookup", 10, sampled),
            ("lookup", 11, not sampled),
            ("load", 11, not sampled),
        ]
        if failure != "lookup":
            expected.append(("load", 10, sampled))
        assert sorted(adapter.seen) == sorted(expected)
        if failure == "load":
            assert [
                call.args[0] for call in l1.finish_write_and_delete.call_args_list
            ] == [[keys[0]]]
            assert adapter.debug_get_locked_key_count() == 0
    finally:
        controller.stop()
        adapter.close()
