# SPDX-License-Identifier: Apache-2.0
"""Provider leases outlive private-backend mappings and ambiguous hotplug calls."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal
import json

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.memory_orchestrator.pooling import (
    MemoryAllocation,
    MemoryPoolController,
)

ALLOCATION = MemoryAllocation("lease-1", "/dev/dax0.1", 4096, 8192)


@dataclass
class Pool:
    """Record physical ownership transitions, including uncertain release."""

    events: list[str] = field(default_factory=list)
    acquired: list[tuple[str, str, int]] = field(default_factory=list)
    release_failures: int = 0

    def acquire(
        self, consumer_id: str, request_id: str, size_bytes: int
    ) -> MemoryAllocation:
        self.events.append("acquire")
        self.acquired.append((consumer_id, request_id, size_bytes))
        return ALLOCATION

    def release(self, allocation_id: str) -> None:
        assert allocation_id == ALLOCATION.allocation_id
        self.events.append("release")
        if self.release_failures:
            self.release_failures -= 1
            raise OSError("release acknowledgement lost")


def reply(body: object, code: int = 200, instance: str = "worker-a") -> httpx.Response:
    """Return a versioned worker reply in the existing HTTP wire format."""
    return httpx.Response(code, json=body, headers={"X-LMCache-Instance": instance})


def arena(state: str = "active") -> dict[str, object]:
    """Describe the leased range using the existing L1 status fields."""
    return dict(
        device_path="/dev/dax0.1", offset_in_bytes=4096, size_in_bytes=8192, state=state
    )


def client_for(
    pool: Pool, responses: Sequence[httpx.Response | Exception]
) -> httpx.Client:
    """Script HTTP outcomes while verifying worker fencing and byte-range routing."""
    first = True
    outcomes = iter(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal first
        if first:
            first = False
            assert request.method == "GET"
            return reply(
                {"arenas": [], "adapters": [{"status": {"hotplug_enabled": True}}]}
            )
        assert request.headers["X-LMCache-Expected-Instance"] == "worker-a"
        if request.method == "POST":
            operation = request.url.path.rsplit("/", 1)[-1]
            pool.events.append(operation)
            payload = json.loads(request.content)
            assert (payload["device_path"], payload["offset"]) == ("/dev/dax0.1", 4096)
            if "/l2/" in request.url.path:
                assert payload["adapter_index"] == 0
                if operation == "remove":
                    assert payload["mode"] == "evict" and payload["force"] is False
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.Client(base_url="http://worker", transport=httpx.MockTransport(handle))


@pytest.mark.parametrize("tier", ["l1", "l2"])
def test_acquire_precedes_attach_and_release_follows_unmap(
    tier: Literal["l1", "l2"],
) -> None:
    pool = Pool()
    added = (
        {"added": arena()}
        if tier == "l1"
        else {
            "device": dict(
                device_path="/dev/dax0.1",
                offset_bytes=4096,
                max_dax_size_bytes=8192,
                state="active",
            )
        }
    )
    removed: dict[str, object] = (
        {"removed": {"arenas": [arena("removed")]}}
        if tier == "l1"
        else {"state": "removed", "operation": "remove", "device_path": "/dev/dax0.1"}
    )
    with client_for(pool, [reply(added), reply(removed)]) as client:
        controller = MemoryPoolController(pool, client, "node-a", tier)
        assert controller.grow("grow-1", 8192) == ALLOCATION
        assert controller.grow("grow-1", 8192) == ALLOCATION
        with pytest.raises(ValueError):
            controller.grow("grow-1", 4096)
        assert controller.shrink("grow-1")
        with pytest.raises(KeyError):
            controller.shrink("grow-1")
    assert pool.acquired == [("node-a", f"worker-a/{tier}/0/grow-1", 8192)]
    assert pool.events == ["acquire", "add", "remove", "release"]


def test_l1_drain_keeps_lease_until_same_worker_confirms_disappearance() -> None:
    pool = Pool()
    responses = [
        reply({"added": arena()}),
        reply({"removed": {"arenas": [arena("draining")]}}),
        reply({"removed": {"arenas": [arena("draining")]}}),
        reply({"error": "arena already reaped"}, 404),
    ]
    with client_for(pool, responses) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l1")
        controller.grow("grow-1", 8192)
        assert not controller.shrink("grow-1")
        assert not controller.shrink("grow-1")
        assert "release" not in pool.events
        assert controller.shrink("grow-1")
    assert pool.events == ["acquire", "add", "remove", "remove", "remove", "release"]


def test_l2_busy_remove_can_retry_without_releasing_or_reacquiring() -> None:
    pool = Pool()
    added = {
        "device": dict(
            device_path="/dev/dax0.1",
            offset_bytes=4096,
            max_dax_size_bytes=8192,
            state="active",
        )
    }
    with client_for(
        pool,
        [
            reply(added),
            reply({"error": "borrowed"}, 409),
            reply(
                {
                    "state": "removed",
                    "operation": "remove",
                    "device_path": "/dev/dax0.1",
                }
            ),
        ],
    ) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l2")
        controller.grow("grow-1", 8192)
        with pytest.raises(httpx.HTTPStatusError):
            controller.shrink("grow-1")
        assert "release" not in pool.events
        assert controller.shrink("grow-1")
    assert pool.events == ["acquire", "add", "remove", "remove", "release"]


@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        (httpx.ReadTimeout("unknown add outcome"), httpx.ReadTimeout),
        (reply({"error": "mapping failed"}, 500), httpx.HTTPStatusError),
        (reply({"added": {}}), RuntimeError),
    ],
)
def test_uncertain_attach_never_rolls_back_or_repeats_hotplug(
    outcome: httpx.Response | Exception, error: type[Exception]
) -> None:
    pool = Pool()
    with client_for(pool, [outcome]) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l1")
        with pytest.raises(error):
            controller.grow("grow-1", 8192)
        with pytest.raises(RuntimeError):
            controller.grow("grow-1", 8192)
        with pytest.raises(RuntimeError):
            controller.shrink("grow-1")
    assert pool.events == ["acquire", "add"]


def test_replacement_worker_cannot_confirm_previous_mapping_is_gone() -> None:
    pool = Pool()
    responses = [
        reply({"added": arena()}),
        reply({"removed": {"arenas": [arena("draining")]}}),
        reply({"error": "arena not found"}, 404, instance="worker-b"),
    ]
    with client_for(pool, responses) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l1")
        controller.grow("grow-1", 8192)
        assert not controller.shrink("grow-1")
        with pytest.raises(RuntimeError, match="instance"):
            controller.shrink("grow-1")
        with pytest.raises(RuntimeError):
            controller.shrink("grow-1")
    assert "release" not in pool.events


def test_missing_mapping_without_an_acknowledged_drain_keeps_lease() -> None:
    pool = Pool()
    with client_for(pool, [reply({"added": arena()}), reply({}, 404)]) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l1")
        controller.grow("grow-1", 8192)
        with pytest.raises(httpx.HTTPStatusError):
            controller.shrink("grow-1")
        with pytest.raises(RuntimeError):
            controller.shrink("grow-1")
    assert "release" not in pool.events


def test_release_retry_does_not_repeat_backend_removal() -> None:
    pool = Pool(release_failures=1)
    with client_for(
        pool,
        [reply({"added": arena()}), reply({"removed": {"arenas": [arena("removed")]}})],
    ) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l1")
        controller.grow("grow-1", 8192)
        with pytest.raises(OSError, match="acknowledgement"):
            controller.shrink("grow-1")
        assert controller.shrink("grow-1")
    assert pool.events == ["acquire", "add", "remove", "release", "release"]
