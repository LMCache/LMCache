# SPDX-License-Identifier: Apache-2.0
"""Tests for the durability contract and for capturing a controller's
components as one consistent group."""

# Standard
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import cast
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    PersistenceType,
)
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager
from tests.v1.mp_coordinator.persistence.conftest import capture_consistently


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(chunk_hash=chunk_id.to_bytes(4, "big"), model_name="m", kv_rank=0)


def _pipeline() -> tuple[
    EventGate, QuiesceLock, CacheUsageManager, FleetEvictionController
]:
    """The real path: the usage view consumes before the controller, which
    reads it for the same batch (as ``create_app`` orders them)."""
    usage_manager = CacheUsageManager()
    controller = FleetEvictionController(usage_manager=usage_manager)
    broadcaster = CacheEventBroadcaster()
    broadcaster.register_consumer(usage_manager)
    broadcaster.register_consumer(controller)
    quiesce = QuiesceLock()
    gate = EventGate(broadcaster, quiesce)
    return gate, quiesce, usage_manager, controller


def _batch(seq: int, key: ObjectKey, size_bytes: int = 1024) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[
            CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=size_bytes)
        ],
    )


class TestTheContract:
    def test_each_component_names_itself_and_where_it_belongs(self):
        """The section name and the artifact are the component's own
        business; nothing outside it decides either."""
        *_, controller = _pipeline()

        described = {
            component.name: component.persistence_type
            for component in controller.get_durable_components()
        }

        assert described == {
            "pins": PersistenceType.METADATA,
            "quotas": PersistenceType.METADATA,
            "lru_order": PersistenceType.CHECKPOINT,
        }


class TestGroupConsistency:
    def test_a_capture_never_splits_a_batch_across_consumers(self):
        """The reason the quiesce exists. One batch is applied by the usage
        view and then by the eviction controller; a capture landing
        between the two records bytes charged to a key the policy has no
        record of -- a state no moment ever matched, and plausible enough
        to restore without noticing.
        """
        gate, quiesce, usage_manager, controller = _pipeline()
        stop = threading.Event()
        mismatches: list[str] = []

        # The real gap between the two consumers is microseconds, which a
        # test would miss by luck rather than by correctness. Widening it
        # means an unprotected capture lands inside almost every time.
        usage_consume = usage_manager.consume

        def slow_usage_consume(batch: CacheEventBatch) -> None:
            usage_consume(batch)
            time.sleep(0.001)

        usage_manager.consume = slow_usage_consume  # type: ignore[method-assign]

        def ingest() -> None:
            seq = 0
            while not stop.is_set():
                seq += 1
                gate.ingest(_batch(seq=seq, key=_key(seq)))

        def capture_repeatedly() -> None:
            for _ in range(25):
                # Short, so a barrier that stops excluding batches fails
                # fast rather than stalling the suite.
                captured = capture_consistently(
                    quiesce, [usage_manager, controller.policy], timeout=1.0
                )
                placements = cast(
                    "list[tuple[str, ObjectKey, str, str, int]]",
                    captured["cache_usage"]["placements"],
                )
                buckets = cast(
                    "Mapping[str, list[ObjectKey]]",
                    captured["lru_order"]["buckets"],
                )
                accounted = sum(size for *_, size in placements)
                tracked = len(buckets.get("", []))
                # Every key contributes the same size, so the ledger and
                # the policy must agree on how many keys exist.
                if accounted != tracked * 1024:
                    mismatches.append(f"{accounted} bytes vs {tracked} keys")

        with ThreadPoolExecutor(max_workers=2) as pool:
            ingest_future = pool.submit(ingest)
            capture_future = pool.submit(capture_repeatedly)
            try:
                capture_future.result(timeout=60.0)
            finally:
                # Without this the ingest loop outlives a failure and the
                # pool never shuts down: a hang instead of a red test.
                stop.set()
                ingest_future.result(timeout=10.0)

        assert not mismatches, f"captured a state that never existed: {mismatches[:3]}"


class TestTheCaptureContract:
    def test_every_capture_is_plain_data(self):
        """An artifact writer must not need to know what a section means,
        so no capture may hand back a domain object -- an `ObjectKey`, a
        `Placement`, a numpy array. Left unchecked this rots quietly: the
        writer grows a special case per type and the coupling is only
        visible once a new artifact backend has to reproduce it.
        """
        gate, quiesce, usage_manager, controller = _pipeline()
        gate.ingest(_batch(seq=1, key=_key(1)))
        controller.pin([_key(1)])
        controller.quota.set_quota("tenant-a", 4096)
        directory = KeyDirectory()
        directory.consume(_batch(seq=2, key=_key(2)))

        captured = capture_consistently(
            quiesce,
            [usage_manager, gate, directory, *controller.get_durable_components()],
        )

        for name, section in captured.items():
            _assert_plain(section, where=name)

    def test_a_domain_object_is_caught(self):
        """The check above is only worth having if it fails."""
        with pytest.raises(AssertionError, match="ObjectKey"):
            _assert_plain({"keys": [_key(1)]}, where="fake")


def _assert_plain(value: object, where: str) -> None:
    """Assert ``value`` is a nest of dicts, sequences, scalars and bytes."""
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_plain(key, where)
            _assert_plain(item, where)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_plain(item, where)
        return
    raise AssertionError(
        f"{where!r} captured a {type(value).__name__}; captures hold plain data"
    )
