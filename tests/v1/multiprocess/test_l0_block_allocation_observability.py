# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM L0 block allocation observability wiring.

These tests intentionally exercise the wire boundaries around
REPORT_BLOCK_ALLOCATION rather than only subscriber metric math:

* scheduler adapter -> message queue request
* MP server management handler -> EventBus event
"""

# Future
from __future__ import annotations

# Standard
from unittest.mock import MagicMock
import threading
import types

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
)
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord
from lmcache.v1.multiprocess.protocol import RequestType


class _RecordingEventBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish(self, event) -> None:
        self.events.append(event)


def _records() -> list[BlockAllocationRecord]:
    return [
        BlockAllocationRecord(
            req_id="req-prod-1",
            new_block_ids=[7, 8, 9],
            new_token_ids=[101, 102, 103],
        )
    ]


def test_scheduler_adapter_reports_block_allocation_to_message_queue():
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "production-model"
    adapter._health_event = threading.Event()
    adapter._health_event.set()
    adapter.mq_client = MagicMock()
    records = _records()

    adapter.report_block_allocations(records)

    adapter.mq_client.submit_request.assert_called_once()
    request_type, payloads, response_class = adapter.mq_client.submit_request.call_args[
        0
    ]
    assert request_type == RequestType.REPORT_BLOCK_ALLOCATION
    assert payloads[1] == "production-model"
    assert payloads[2] == records
    assert response_class is None


def test_management_module_publishes_report_block_allocation_event():
    # First Party
    from lmcache.v1.multiprocess.modules.management import ManagementModule

    recording_bus = _RecordingEventBus()
    ctx = types.SimpleNamespace(event_bus=recording_bus)
    module = ManagementModule(ctx)  # type: ignore[arg-type]
    records = _records()

    module.report_block_allocations(
        instance_id=42,
        model_name="production-model",
        records=records,
    )

    [published] = recording_bus.events
    assert published.event_type == EventType.MP_VLLM_BLOCK_ALLOCATION
    assert published.metadata == {
        "instance_id": 42,
        "model_name": "production-model",
        "records": records,
    }
