# SPDX-License-Identifier: Apache-2.0
"""Exercise capacity decisions through the real store-loop notification path."""

# Standard
from types import SimpleNamespace
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import (
    CapacityAdmission,
    DefaultStorePolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.platform import create_event_notifier


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "delete_refused",
        "delete_raises",
        "snapshot_raises",
        "submit_raises",
        "store_fails",
    ],
)
def test_store_loop_rechecks_deleted_capacity_and_releases_every_read(
    monkeypatch, failure
):
    incoming = ObjectKey(chunk_hash=b"incoming", model_name="m", kv_rank=0)
    rejected = ObjectKey(chunk_hash=b"rejected", model_name="m", kv_rank=0)
    resident = ObjectKey(chunk_hash=b"resident", model_name="m", kv_rank=0)
    released = []
    done = threading.Event()
    calls = []
    completions = []
    listeners = []

    def finish_read(keys):
        released.extend(keys)
        if set(released) == {incoming, rejected}:
            done.set()

    l1 = SimpleNamespace(
        config=SimpleNamespace(tag="capacity-test"),
        register_listener=listeners.append,
        reserve_read=lambda keys: {
            key: (L1Error.SUCCESS, SimpleNamespace(get_size=lambda: 1)) for key in keys
        },
        finish_read=finish_read,
    )

    class Policy(DefaultStorePolicy):
        @property
        def requires_value_capacity(self):
            return True

        def plan_capacity_admission(self, keys, block_bytes, snapshot):
            return CapacityAdmission(
                (incoming,), (resident,) if snapshot.resident_sizes else ()
            )

    class Adapter:
        supports_value_capacity = True

        def __init__(self):
            self.residents = {resident: 1}
            self.notifier = create_event_notifier()

        def get_store_event_fd(self):
            return self.notifier.fileno()

        def get_capacity_snapshot(self):
            calls.append("snapshot")
            if failure == "snapshot_raises":
                raise RuntimeError("snapshot failed")
            return SimpleNamespace(resident_sizes=dict(self.residents))

        def delete_unlocked(self, keys):
            calls.append("delete")
            if failure == "delete_raises":
                raise RuntimeError("delete failed")
            if failure != "delete_refused":
                for key in keys:
                    self.residents.pop(key, None)

        def submit_store_task(self, keys, objects):
            calls.append(("submit", tuple(keys)))
            if failure == "submit_raises":
                raise RuntimeError("submit failed")
            self.notifier.notify()
            return 1

        def pop_completed_store_tasks(self):
            return {1: L2StoreResult(failure != "store_fails", 1)}

    bus = EventBus(EventBusConfig(enabled=True))
    bus.subscribe(EventType.L2_STORE_COMPLETED, lambda event: completions.append(event))
    monkeypatch.setattr(
        "lmcache.v1.distributed.storage_controllers.store_controller.get_event_bus",
        lambda: bus,
    )
    adapter = Adapter()
    controller = StoreController(
        l1,
        [adapter],
        [L2AdapterDescriptor(0, MockL2AdapterConfig(0.01, 10.0))],
        Policy(),
    )
    bus.start()
    controller.start()
    try:
        listeners[0].on_l1_keys_write_finished([incoming, rejected])
        assert done.wait(3), "capacity failure leaked a read lock or stalled loop"
    finally:
        controller.stop()
        bus.stop()
        adapter.notifier.close()
    assert sorted(released, key=lambda key: key.chunk_hash) == [incoming, rejected]
    submits = [call for call in calls if isinstance(call, tuple)]
    if failure in {"delete_refused", "delete_raises", "snapshot_raises"}:
        assert submits == [], "failed deletion/snapshot must not count as free space"
        assert completions == [], "rejection must not publish a successful store"
    else:
        assert submits == [("submit", (incoming,))]
        assert calls[:4] == ["snapshot", "delete", "snapshot", ("submit", (incoming,))]
        if failure == "submit_raises":
            assert completions == []
        else:
            assert len(completions) == 1
            assert completions[0].metadata["succeeded_count"] == (
                0 if failure == "store_fails" else 1
            )
