# SPDX-License-Identifier: Apache-2.0
"""Keep provider leases until detach is certain; pressure E2E covers L1 success."""

# Standard
from unittest.mock import Mock, call
import json

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.memory_orchestrator.pooling import (
    MemoryAllocation,
    MemoryPoolController,
    MemoryPoolProvider,
)

ALLOCATION = MemoryAllocation("lease-1", "/dev/dax0.1", 4096, 8192)


@pytest.fixture
def pool() -> Mock:
    """Return a provider with inspectable ownership calls."""
    return Mock(spec=MemoryPoolProvider, acquire=Mock(return_value=ALLOCATION))


def reply(body: object, code: int = 200, instance: str = "worker-a") -> httpx.Response:
    """Build a response from a specific worker incarnation."""
    return httpx.Response(code, json=body, headers={"X-LMCache-Instance": instance})


def arena(state: str = "active") -> dict[str, object]:
    """Describe the leased range in the L1 wire format."""
    return dict(
        device_path=ALLOCATION.device_path,
        offset_in_bytes=4096,
        size_in_bytes=8192,
        state=state,
    )


def client_for(pool: Mock, *responses: httpx.Response | Exception) -> httpx.Client:
    """Check ownership ordering and route fields while supplying HTTP outcomes."""
    outcomes = iter(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return reply(
                {"arenas": [], "adapters": [{"status": {"hotplug_enabled": True}}]}
            )
        pool.acquire.assert_called_once()
        pool.release.assert_not_called()
        assert request.headers["X-LMCache-Expected-Instance"] == "worker-a"
        payload = json.loads(request.content)
        assert (payload["device_path"], payload["offset"]) == (
            ALLOCATION.device_path,
            4096,
        )
        if request.url.path.endswith("/add"):
            assert payload["size"] == 8192
        if "/l2/" in request.url.path:
            assert payload["adapter_index"] == 0
            if request.url.path.endswith("/remove"):
                assert payload["mode"] == "evict" and payload["force"] is False
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.Client(base_url="http://worker", transport=httpx.MockTransport(handle))


def test_l2_busy_detach_and_uncertain_release_retry_without_reacquisition(
    pool: Mock,
) -> None:
    added = dict(
        device_path=ALLOCATION.device_path,
        offset_bytes=4096,
        max_dax_size_bytes=8192,
        state="active",
    )
    removed = dict(
        state="removed", operation="remove", device_path=ALLOCATION.device_path
    )
    pool.release.side_effect = [OSError("release acknowledgement lost"), None]
    with client_for(
        pool, reply({"device": added}), reply({}, 409), reply(removed)
    ) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l2")
        assert controller.grow("grow-1", 8192) == ALLOCATION
        assert controller.grow("grow-1", 8192) == ALLOCATION
        with pytest.raises(ValueError):
            controller.grow("grow-1", 4096)
        pool.acquire.assert_called_once_with("node-a", "worker-a/l2/0/grow-1", 8192)
        with pytest.raises(httpx.HTTPStatusError):
            controller.shrink("grow-1")
        pool.release.assert_not_called()
        with pytest.raises(OSError):
            controller.shrink("grow-1")
        assert controller.shrink("grow-1")
        with pytest.raises(KeyError):
            controller.shrink("grow-1")
    assert pool.release.call_args_list == [call("lease-1")] * 2


@pytest.mark.parametrize(
    ("drain", "instance"), [(True, "worker-a"), (True, "worker-b"), (False, "worker-a")]
)
def test_l1_missing_range_requires_acknowledged_drain_on_same_worker(
    pool: Mock,
    drain: bool,
    instance: str,
) -> None:
    responses = [reply({"added": arena()})]
    if drain:
        responses += [reply({"removed": {"arenas": [arena("draining")]}})] * 2
    responses.append(reply({}, 404, instance))
    with client_for(pool, *responses) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l1")
        controller.grow("grow-1", 8192)
        if drain:
            assert not controller.shrink("grow-1")
            assert not controller.shrink("grow-1")
            pool.release.assert_not_called()
        if drain and instance == "worker-a":
            assert controller.shrink("grow-1")
            pool.release.assert_called_once_with("lease-1")
        else:
            error = RuntimeError if drain else httpx.HTTPStatusError
            with pytest.raises(error):
                controller.shrink("grow-1")
            with pytest.raises(RuntimeError):
                controller.shrink("grow-1")
            pool.release.assert_not_called()


@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        (httpx.ReadTimeout("unknown add outcome"), httpx.ReadTimeout),
        (reply({}, 500), httpx.HTTPStatusError),
        (reply({"added": {}}), RuntimeError),
    ],
)
def test_uncertain_attach_never_releases_or_repeats_hotplug(
    pool: Mock,
    outcome: httpx.Response | Exception,
    error: type[Exception],
) -> None:
    with client_for(pool, outcome) as client:
        controller = MemoryPoolController(pool, client, "node-a", "l1")
        with pytest.raises(error):
            controller.grow("grow-1", 8192)
        with pytest.raises(RuntimeError):
            controller.grow("grow-1", 8192)
        with pytest.raises(RuntimeError):
            controller.shrink("grow-1")
    pool.release.assert_not_called()
