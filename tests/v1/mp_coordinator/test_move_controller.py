# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator move controller: target-pulled copy, then source
delete, driven by the coordinator itself.

One ``httpx.MockTransport`` plays both MP servers. Each endpoint answers from
a script the test writes, so a test reads as the exact conversation the
controller had, and asserts what it reported back.
"""

# Standard
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
import asyncio
import itertools
import json
import math
import time

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.api import MovePhase, MoveStatus
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers.base import ControllerRuntime
from lmcache.v1.mp_coordinator.controllers.move_controller import (
    MoveController,
    MoveOutcome,
    MoveSettings,
    MoveSpec,
    MoveSubmitError,
)
from lmcache.v1.mp_coordinator.views import build_views
from lmcache.v1.mp_coordinator.views.instance_registry import (
    InstanceRegistry,
    MPInstance,
)
from lmcache.v1.multiprocess.cache_control.object_service import MAX_DELETE_BATCH

_SOURCE_IP = "10.0.0.1"
_TARGET_IP = "10.0.0.2"
_REQUEST_ID = "rid-1"


# -- What the MP servers may answer ------------------------------------------

_Answer = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _json(status_code: int, body: object) -> _Answer:
    async def answer(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=body)

    return answer


def _raw(status_code: int, content: bytes) -> _Answer:
    async def answer(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=content)

    return answer


def _raising(exc: Exception) -> _Answer:
    async def answer(_request: httpx.Request) -> httpx.Response:
        raise exc

    return answer


def _submitted() -> _Answer:
    return _json(202, {"request_id": _REQUEST_ID, "chunks": 1, "status": "submitted"})


def _pending() -> _Answer:
    return _json(200, {"request_id": _REQUEST_ID, "status": "pending"})


def _completed_body(total: int, missing: tuple[int, ...] = ()) -> dict[str, object]:
    return {
        "request_id": _REQUEST_ID,
        "status": "completed",
        "found_keys": total - len(missing),
        "total_keys": total,
        "missing_key_indices": list(missing),
    }


def _completed(total: int, missing: tuple[int, ...] = ()) -> _Answer:
    """The target's report: ``total`` keys, of which ``missing`` did not load."""
    return _json(200, _completed_body(total, missing))


def _ack(locked: int = 0) -> _Answer:
    """The source acknowledges a delete batch, refusing ``locked`` of its keys."""

    async def answer(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["keys"])
        return httpx.Response(200, json={"deleted": n - locked, "skipped": locked})

    return answer


def _blocking(delay_s: float, answer: _Answer) -> _Answer:
    """``answer`` after stalling the event loop for ``delay_s`` -- the whole
    loop, not just this request, as a large synchronous handler would."""

    async def blocking(request: httpx.Request) -> httpx.Response:
        time.sleep(delay_s)
        return await answer(request)

    return blocking


def _gated(gate: asyncio.Event, answer: _Answer) -> _Answer:
    """``answer``, but only once ``gate`` is set."""

    async def gated(request: httpx.Request) -> httpx.Response:
        await gate.wait()
        return await answer(request)

    return gated


class _Trickle(httpx.AsyncByteStream):
    """A body that arrives in ``pieces`` with ``gap_s`` between them -- the
    shape a per-request read timeout cannot bound."""

    def __init__(self, body: bytes, pieces: int, gap_s: float) -> None:
        self._body = body
        self._pieces = pieces
        self._gap_s = gap_s

    async def __aiter__(self) -> AsyncIterator[bytes]:
        step = max(1, len(self._body) // self._pieces)
        for start in range(0, len(self._body), step):
            yield self._body[start : start + step]
            await asyncio.sleep(self._gap_s)


def _trickled(body: dict[str, object], pieces: int, gap_s: float) -> _Answer:
    async def answer(_request: httpx.Request) -> httpx.Response:
        stream = _Trickle(json.dumps(body).encode(), pieces, gap_s)
        return httpx.Response(200, stream=stream)

    return answer


async def _script_ran_out(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no scripted answer left for {request.method} {request.url}")


class _Fleet:
    """Both MP servers behind one transport, each endpoint answering from a
    script. Records every submit body, status poll, and delete body it saw.

    Args:
        status: The target's answers to status polls, in order; a poll past
            the end fails the test.
        delete: The source's answers to delete batches, in order; past the
            end, every batch is acknowledged in full.
        submit: The target's answers to the prefetch submit; default: a
            submitted job.
        source_ip: Address the source answers deletes at.
    """

    def __init__(
        self,
        *,
        status: Iterable[_Answer] = (),
        delete: Iterable[_Answer] = (),
        submit: Iterable[_Answer] = (),
        source_ip: str = _SOURCE_IP,
    ) -> None:
        self._status = iter(status)
        self._delete = iter(delete)
        self._submit = iter(submit)
        self.source_ip = source_ip
        self.submits: list[dict[str, object]] = []
        self.status_polls = 0
        self.deletes: list[dict[str, object]] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        host, path, method = request.url.host, request.url.path, request.method
        if host == _TARGET_IP and method == "POST" and path == "/cache/prefetches":
            self.submits.append(json.loads(request.content))
            return await next(self._submit, _submitted())(request)
        if (
            host == _TARGET_IP
            and method == "GET"
            and path == f"/cache/prefetches/{_REQUEST_ID}"
        ):
            self.status_polls += 1
            return await next(self._status, _script_ran_out)(request)
        if host == self.source_ip and method == "DELETE" and path == "/cache/objects":
            self.deletes.append(json.loads(request.content))
            return await next(self._delete, _ack())(request)
        return httpx.Response(404, json={"detail": f"unexpected {method} {path}"})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


# -- Driving a move ----------------------------------------------------------


def _instance(instance_id: str, ip: str, port: int = 8080) -> MPInstance:
    now = time.time()
    return MPInstance(
        instance_id=instance_id,
        ip=ip,
        http_port=port,
        registration_time=now,
        last_heartbeat_time=now,
    )


def _registry(*instances: MPInstance) -> InstanceRegistry:
    """A registry holding ``instances`` (default: the source and the target)."""
    registry = InstanceRegistry()
    defaults = (_instance("mp-src", _SOURCE_IP), _instance("mp-dst", _TARGET_IP))
    for instance in instances or defaults:
        registry.register(instance)
    return registry


def _controller(*instances: MPInstance, **settings: float) -> MoveController:
    """A fast-polling controller over a registry holding ``instances``
    (default: the source and the target); ``settings`` override
    :class:`MoveSettings` by key."""
    return MoveController(
        _registry(*instances),
        MoveSettings.model_validate({"move_poll_interval_s": 0.005, **settings}),
    )


def _keys(n: int) -> list[ObjectKey]:
    return [
        ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)
        for i in range(n)
    ]


def _tokens(keys: list[ObjectKey]) -> list[int]:
    return list(range(4 * len(keys)))


def _spec(keys: list[ObjectKey], keep_source: bool = False) -> MoveSpec:
    return MoveSpec(
        source_instance_id="mp-src",
        target_instance_id="mp-dst",
        model_name="m",
        world_size=1,
        cache_salt="alice",
        keys=keys,
        chunks=len(keys),
        keep_source=keep_source,
    )


async def _submit(
    ctl: MoveController,
    client: httpx.AsyncClient,
    n: int = 2,
    keep_source: bool = False,
) -> str:
    """Submit a move of ``n`` keys and return its id."""
    keys = _keys(n)
    return await ctl.submit_move(_spec(keys, keep_source), _tokens(keys), client)


async def _settle(
    ctl: MoveController, move_id: str, timeout: float = 3.0
) -> MoveOutcome:
    """Poll until the move leaves ``PENDING`` and return that terminal outcome."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outcome = ctl.get_status(move_id)
        assert outcome is not None, "a pending move must stay reportable"
        if outcome.status is not MoveStatus.PENDING:
            return outcome
        await asyncio.sleep(0.005)
    raise AssertionError("move did not settle in time")


async def _run(
    fleet: _Fleet, n: int = 2, keep_source: bool = False, **settings: float
) -> MoveOutcome:
    """Move ``n`` keys through ``fleet`` on a fresh controller; return how it
    settled."""
    ctl = _controller(**settings)
    async with fleet.client() as client:
        return await _settle(ctl, await _submit(ctl, client, n, keep_source))


def _delete_hashes(delete_body: dict[str, object]) -> list[str]:
    keys = delete_body["keys"]
    assert isinstance(keys, list)
    return [key["chunk_hash_hex"] for key in keys]


# -- Construction ------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_config_overrides_take_effect():
    """extra_config values drive behaviour, not just construction: a short
    completion timeout from config fails a never-completing move at that
    timeout, and a small result limit from config bounds retention."""
    config = MPCoordinatorConfig(
        extra_config={
            "move_poll_interval_s": 0.005,
            "move_completion_timeout_s": 0.05,
            "move_result_ttl_s": 60,
            "move_result_limit": 1,
        }
    )
    views = build_views(config)
    views.get(InstanceRegistry).register(_instance("mp-dst", _TARGET_IP))
    ctl = MoveController.from_config(config, views)
    fleet = _Fleet(status=itertools.repeat(_pending()))
    async with fleet.client() as client:
        started = time.monotonic()
        first = await _submit(ctl, client)
        outcome = await _settle(ctl, first)
        elapsed = time.monotonic() - started
        second = await _submit(ctl, client)
        await _settle(ctl, second)

    assert outcome.status is MoveStatus.FAILED
    assert "0.05s" in outcome.error
    assert elapsed < 0.5  # the 600 s default would still be waiting
    assert ctl.get_status(first) is None  # evicted by move_result_limit=1
    assert ctl.get_status(second) is not None


@pytest.mark.parametrize(
    "key",
    ["move_poll_interval_s", "move_completion_timeout_s", "move_result_ttl_s"],
)
@pytest.mark.parametrize(
    "value",
    [0, -1, "10", True, math.nan, math.inf, None],
    ids=["zero", "negative", "string", "bool", "nan", "inf", "none"],
)
def test_from_config_rejects_non_finite_or_non_positive_numbers(
    key: str, value: object
):
    with pytest.raises(ValueError, match=key):
        MoveController.from_config(
            MPCoordinatorConfig(extra_config={key: value}),
            build_views(MPCoordinatorConfig()),
        )


@pytest.mark.parametrize(
    "value", [0, -3, 1.5, True, "5"], ids=["zero", "negative", "float", "bool", "str"]
)
def test_from_config_rejects_invalid_result_limit(value: object):
    with pytest.raises(ValueError, match="move_result_limit"):
        MoveController.from_config(
            MPCoordinatorConfig(extra_config={"move_result_limit": value}),
            build_views(MPCoordinatorConfig()),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"move_poll_interval_s": 0},
        {"move_completion_timeout_s": math.inf},
        {"move_result_ttl_s": math.nan},
        {"move_result_limit": 0},
    ],
    ids=["zero-interval", "inf-timeout", "nan-ttl", "zero-limit"],
)
def test_settings_apply_the_same_checks_directly(kwargs: dict[str, object]):
    with pytest.raises(ValueError):
        MoveSettings.model_validate(kwargs)


# -- The happy path and its variants -----------------------------------------


@pytest.mark.asyncio
async def test_move_deletes_only_what_the_target_loaded():
    """The target's prefetch is submitted with the tokens; once it reports,
    exactly the keys it loaded are deleted from the source's L1 (never
    forced), and the outcome carries every count."""
    keys = _keys(4)
    fleet = _Fleet(status=[_pending(), _pending(), _completed(4, missing=(1,))])

    outcome = await _run(fleet, n=4)

    assert fleet.submits == [
        {
            "model_name": "m",
            "world_size": 1,
            "token_ids": _tokens(keys),
            "cache_salt": "alice",
        }
    ]
    assert fleet.status_polls == 3  # two pending, then completed
    assert len(fleet.deletes) == 1
    assert fleet.deletes[0]["tier"] == "l1"
    assert fleet.deletes[0]["force"] is False
    assert _delete_hashes(fleet.deletes[0]) == [
        keys[i].chunk_hash.hex() for i in (0, 2, 3)
    ]
    assert outcome == MoveOutcome(
        move_id=outcome.move_id,
        source_instance_id="mp-src",
        target_instance_id="mp-dst",
        status=MoveStatus.COMPLETED,
        phase=MovePhase.DELETE,
        requested=4,
        loaded=3,
        missing=1,
        deleted=3,
        skipped=0,
        error="",
    )


@pytest.mark.asyncio
async def test_keep_source_skips_the_source_delete():
    """keep_source=True loads and reports, but never deletes from the source."""
    fleet = _Fleet(status=[_completed(3)])

    outcome = await _run(fleet, n=3, keep_source=True)

    assert fleet.deletes == []
    assert outcome.status is MoveStatus.COMPLETED
    assert (outcome.loaded, outcome.missing, outcome.deleted) == (3, 0, 0)


@pytest.mark.asyncio
async def test_nothing_loaded_means_nothing_deleted():
    """A target that found no source for any key completes the move with
    every key missing and issues no delete."""
    fleet = _Fleet(status=[_completed(2, missing=(0, 1))])

    outcome = await _run(fleet)

    assert fleet.deletes == []
    assert outcome.status is MoveStatus.COMPLETED
    assert (outcome.loaded, outcome.missing, outcome.deleted) == (0, 2, 0)


@pytest.mark.asyncio
async def test_locked_keys_are_reported_skipped():
    """Keys the source refuses (locked) are counted, not forced."""
    fleet = _Fleet(status=[_completed(3)], delete=[_ack(locked=1)])

    outcome = await _run(fleet, n=3)

    assert outcome.status is MoveStatus.COMPLETED
    assert (outcome.deleted, outcome.skipped) == (2, 1)


@pytest.mark.asyncio
async def test_source_delete_is_batched_at_the_node_cap():
    """More keys than the node accepts per delete go out in several calls."""
    n = MAX_DELETE_BATCH + 1
    fleet = _Fleet(status=[_completed(n)])

    outcome = await _run(fleet, n=n)

    assert [len(d["keys"]) for d in fleet.deletes] == [MAX_DELETE_BATCH, 1]
    assert outcome.deleted == n


# -- Observation contract ----------------------------------------------------


@pytest.mark.asyncio
async def test_status_shows_the_phase_and_the_target_report_as_they_happen():
    """A move is in the load phase until the target reports; while the
    source delete is outstanding it is pending in the delete phase with the
    target's report visible and nothing yet counted as deleted."""
    gate = asyncio.Event()
    fleet = _Fleet(status=[_completed(2)], delete=[_gated(gate, _ack())])
    ctl = _controller()
    async with fleet.client() as client:
        move_id = await _submit(ctl, client)
        fresh = ctl.get_status(move_id)
        assert fresh is not None
        assert (fresh.status, fresh.phase) == (MoveStatus.PENDING, MovePhase.LOAD)
        for _ in range(200):
            if fleet.deletes:
                break
            await asyncio.sleep(0.005)
        mid = ctl.get_status(move_id)
        assert mid is not None
        assert (mid.status, mid.phase, mid.loaded, mid.deleted) == (
            MoveStatus.PENDING,
            MovePhase.DELETE,
            2,
            0,
        )
        gate.set()
        outcome = await _settle(ctl, move_id)

    assert (outcome.status, outcome.deleted) == (MoveStatus.COMPLETED, 2)


@pytest.mark.asyncio
async def test_terminal_status_stays_readable_and_is_not_extended_by_reads():
    """A settled move can be read again until its TTL runs out; reading it
    does not extend the TTL, and an unknown id is None."""
    fleet = _Fleet(status=[_completed(1)])
    ctl = _controller(move_result_ttl_s=0.1)
    async with fleet.client() as client:
        move_id = await _submit(ctl, client, n=1)
        outcome = await _settle(ctl, move_id)
        assert ctl.get_status(move_id) == outcome
        assert ctl.retained == 1
        for _ in range(6):  # keep reading past the TTL
            await asyncio.sleep(0.025)
            ctl.get_status(move_id)

    assert ctl.get_status(move_id) is None
    assert ctl.get_status("never-submitted") is None
    assert ctl.in_flight == 0


@pytest.mark.asyncio
async def test_expired_results_are_swept_without_anyone_polling():
    """Under run(), an expired outcome is dropped on the sweep timer even if
    no client ever reads it."""
    fleet = _Fleet(status=[_completed(1)])
    ctl = _controller(move_result_ttl_s=0.05)
    async with fleet.client() as client:
        async with ctl.run(ControllerRuntime(http_client=client)):
            move_id = await _submit(ctl, client, n=1)
            await _settle(ctl, move_id)
            assert (ctl.in_flight, ctl.retained) == (0, 1)
            await asyncio.sleep(0.15)  # past the TTL and at least one sweep
            assert ctl.retained == 0
    assert ctl.get_status(move_id) is None


@pytest.mark.asyncio
async def test_result_limit_evicts_the_oldest_settled_move_only():
    """The cap applies to settled outcomes, oldest first; a move still
    running is never evicted by it."""
    fleet = _Fleet(status=[_completed(1)] * 3)
    slow = _Fleet(status=itertools.repeat(_pending()))
    ctl = _controller(move_result_limit=2)
    async with fleet.client() as client, slow.client() as slow_client:
        running = await _submit(ctl, slow_client, n=1)
        settled = []
        for _ in range(3):
            move_id = await _submit(ctl, client, n=1)
            await _settle(ctl, move_id)
            settled.append(move_id)

        assert ctl.retained == 2
        assert ctl.get_status(settled[0]) is None
        assert ctl.get_status(settled[1]) is not None
        assert ctl.get_status(settled[2]) is not None
        pending = ctl.get_status(running)
        assert pending is not None and pending.status is MoveStatus.PENDING
        assert ctl.in_flight == 1


# -- Failures never delete without proof -------------------------------------


@pytest.mark.asyncio
async def test_submit_rejected_by_target_raises_and_records_nothing():
    fleet = _Fleet(submit=[_json(503, {"detail": "busy"})])
    ctl = _controller()
    async with fleet.client() as client:
        with pytest.raises(httpx.HTTPError):
            await _submit(ctl, client)
    assert ctl.in_flight == 0


@pytest.mark.parametrize(
    "submit, reason",
    [
        (_raw(200, b"not json"), "unusable"),
        (_raw(200, b"[]"), "unusable"),
        (_json(200, {"status": "submitted"}), "without a request_id"),
        (_json(200, {"chunks": 0, "status": "noop"}), "chunk-size"),
    ],
    ids=["non-json", "not-an-object", "no-request-id", "noop"],
)
@pytest.mark.asyncio
async def test_submit_reply_with_no_job_to_drive_raises_and_records_nothing(
    submit: _Answer, reason: str
):
    """A 2xx submit reply that gives the move no prefetch job -- not JSON,
    not an object, no request_id, or a noop (the target chunked the sequence
    to nothing: a chunk-size mismatch) -- is a submit error: no move is
    recorded and nothing is polled."""
    fleet = _Fleet(submit=[submit])
    ctl = _controller()
    async with fleet.client() as client:
        with pytest.raises(MoveSubmitError, match=reason):
            await _submit(ctl, client)
    assert fleet.status_polls == 0
    assert (ctl.in_flight, ctl.retained) == (0, 0)


@pytest.mark.asyncio
async def test_unregistered_target_raises_before_any_call():
    fleet = _Fleet()
    ctl = _controller(_instance("mp-src", _SOURCE_IP))
    async with fleet.client() as client:
        with pytest.raises(MoveSubmitError, match="not registered"):
            await _submit(ctl, client)
    assert fleet.submits == []


@pytest.mark.asyncio
async def test_target_key_count_mismatch_fails_without_delete():
    fleet = _Fleet(status=[_completed(3)])

    outcome = await _run(fleet, n=2)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.LOAD
    assert "resolved to 2" in outcome.error
    assert fleet.deletes == []


@pytest.mark.asyncio
async def test_target_without_per_key_outcome_fails_without_delete():
    """A server too old to report missing_key_indices cannot be a target."""
    fleet = _Fleet(
        status=[_json(200, {"status": "completed", "found_keys": 2, "total_keys": 2})]
    )

    outcome = await _run(fleet)

    assert outcome.status is MoveStatus.FAILED
    assert "missing_key_indices" in outcome.error
    assert fleet.deletes == []


@pytest.mark.asyncio
async def test_target_dropping_the_job_fails_without_delete():
    fleet = _Fleet(status=[_json(404, {"detail": "gone"})])

    outcome = await _run(fleet)

    assert outcome.status is MoveStatus.FAILED
    assert "no longer knows" in outcome.error
    assert fleet.deletes == []


@pytest.mark.parametrize(
    "status", ["unknown", "failed", "noop", "", "unexpected-state"]
)
@pytest.mark.asyncio
async def test_unexpected_target_status_fails_without_retry_or_delete(status: str):
    """Only pending means keep waiting; other unrecognized states fail closed."""
    fleet = _Fleet(status=[_json(200, {"status": status}), _completed(2)])

    outcome = await _run(fleet)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.LOAD
    assert "unexpected prefetch status" in outcome.error
    assert repr(status) in outcome.error
    assert fleet.status_polls == 1
    assert fleet.deletes == []


@pytest.mark.asyncio
async def test_transient_poll_errors_are_retried():
    """A transport blip on a status poll is retried, and the move completes
    once the target answers."""
    blip = _raising(httpx.ConnectError("blip"))
    fleet = _Fleet(status=[blip, blip, _completed(2)])

    outcome = await _run(fleet)

    assert fleet.status_polls == 3  # two blips, then the answer
    assert outcome.status is MoveStatus.COMPLETED
    assert outcome.deleted == 2


@pytest.mark.asyncio
async def test_unreachable_target_fails_at_the_deadline_without_delete():
    fleet = _Fleet(status=itertools.repeat(_raising(httpx.ConnectError("down"))))

    outcome = await _run(fleet, move_completion_timeout_s=0.05)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.LOAD
    assert "unreachable" in outcome.error
    assert fleet.deletes == []


@pytest.mark.asyncio
async def test_completion_seen_after_the_deadline_is_not_honored():
    """A ``completed`` report the coordinator first sees after the deadline
    -- here because the event loop was stalled past it -- does not trigger
    the source delete. ``wait_for`` alone lets it through on Python 3.10 and
    3.11, where a result landing in the same loop iteration as the timeout
    wins."""
    fleet = _Fleet(status=[_blocking(0.06, _completed(1))])

    outcome = await _run(fleet, n=1, move_completion_timeout_s=0.02)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.LOAD
    assert "after the" in outcome.error and "deadline" in outcome.error
    assert fleet.deletes == []


@pytest.mark.asyncio
async def test_poll_sleep_never_crosses_the_deadline():
    """A poll sleep longer than the time left is cancelled at the deadline
    (the whole wait runs under ``wait_for``), so the move fails then instead
    of one interval later."""
    fleet = _Fleet(status=itertools.repeat(_pending()))

    started = time.monotonic()
    outcome = await _run(
        fleet, n=1, move_poll_interval_s=1.0, move_completion_timeout_s=0.02
    )
    elapsed = time.monotonic() - started

    assert outcome.status is MoveStatus.FAILED
    assert "did not finish" in outcome.error
    assert elapsed < 0.5  # well short of the 1 s interval
    assert fleet.deletes == []


@pytest.mark.asyncio
async def test_deadline_bounds_a_trickling_status_response():
    """A status reply that keeps arriving in pieces cannot outlive the
    deadline: the wait is cancelled mid-transfer, the move fails, and the
    source is untouched."""
    fleet = _Fleet(status=[_trickled(_completed_body(1), pieces=6, gap_s=0.2)])

    started = time.monotonic()
    outcome = await _run(fleet, n=1, move_completion_timeout_s=0.1)  # reply ~1.2 s
    elapsed = time.monotonic() - started

    assert outcome.status is MoveStatus.FAILED
    assert "did not finish" in outcome.error
    assert elapsed < 0.6  # cut at the 0.1 s deadline, not after the reply
    assert fleet.deletes == []


@pytest.mark.asyncio
async def test_timeout_fails_without_delete():
    fleet = _Fleet(status=itertools.repeat(_pending()))

    outcome = await _run(fleet, move_completion_timeout_s=0.05)

    assert outcome.status is MoveStatus.FAILED
    assert "did not finish" in outcome.error
    assert fleet.deletes == []


@pytest.mark.asyncio
async def test_source_unreachable_fails_but_reports_the_load():
    fleet = _Fleet(
        status=[_completed(2)], delete=[_raising(httpx.ConnectError("source down"))]
    )

    outcome = await _run(fleet)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.DELETE
    assert "ConnectError" in outcome.error
    assert (outcome.loaded, outcome.deleted) == (2, 0)


@pytest.mark.asyncio
async def test_delete_failing_mid_way_reports_the_batches_that_landed():
    """A source that rejects a later delete batch fails the move, but the
    earlier batches are not rolled back and ``deleted`` counts them."""
    n = MAX_DELETE_BATCH + 1
    fleet = _Fleet(
        status=[_completed(n)], delete=[_ack(), _json(503, {"detail": "busy"})]
    )

    outcome = await _run(fleet, n=n)

    assert outcome.status is MoveStatus.FAILED
    assert "503" in outcome.error
    assert [len(d["keys"]) for d in fleet.deletes] == [MAX_DELETE_BATCH, 1]
    assert (outcome.loaded, outcome.deleted) == (n, MAX_DELETE_BATCH)


@pytest.mark.parametrize(
    "status_body, reason",
    [
        (
            {
                "status": "completed",
                "found_keys": 0,
                "total_keys": 2,
                "missing_key_indices": [],
            },
            "counts disagree",
        ),
        (
            {
                "status": "completed",
                "found_keys": 1,
                "total_keys": 2,
                "missing_key_indices": [1, 1],
            },
            "repeated",
        ),
        (
            {
                "status": "completed",
                "found_keys": True,
                "total_keys": 2,
                "missing_key_indices": [0],
            },
            "found_keys",
        ),
        (
            {
                "status": "completed",
                "found_keys": 1,
                "total_keys": 2.0,
                "missing_key_indices": [1],
            },
            "total_keys",
        ),
        (
            {
                "status": "completed",
                "found_keys": 1,
                "total_keys": 2,
                "missing_key_indices": [2],
            },
            "out-of-range",
        ),
        (
            {
                "status": "completed",
                "found_keys": 1,
                "total_keys": 2,
                "missing_key_indices": [-1],
            },
            "missing_key_indices",
        ),
    ],
    ids=[
        "contradictory-counts",
        "repeated-index",
        "bool-count",
        "float-total",
        "out-of-range",
        "negative-index",
    ],
)
@pytest.mark.asyncio
async def test_inconsistent_target_report_fails_without_delete(
    status_body: dict[str, object], reason: str
):
    """A completed report whose counts and positions do not agree is not
    acted on: the move fails and the source is untouched."""
    fleet = _Fleet(status=[_json(200, status_body)])

    outcome = await _run(fleet)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.LOAD
    assert reason in outcome.error
    assert fleet.deletes == []


@pytest.mark.parametrize(
    "delete_body, reason",
    [
        ({}, "deleted"),
        ({"deleted": -7, "skipped": True}, "deleted"),
        ({"deleted": 1, "skipped": True}, "skipped"),
        ({"deleted": 2, "skipped": 1}, "batch of 2"),
        ({"deleted": 0, "skipped": 0, "ok": False, "error": "boom"}, "boom"),
    ],
    ids=["empty", "negative", "bool", "over-batch", "ok-false"],
)
@pytest.mark.asyncio
async def test_malformed_delete_reply_fails_the_move(
    delete_body: dict[str, object], reason: str
):
    """A delete reply is validated in full before any count is added: a
    missing, negative, boolean or impossible count, or ``ok: false``, fails
    the move with ``deleted`` untouched -- the batch itself may or may not
    have been applied, which is exactly why nothing is assumed."""
    fleet = _Fleet(status=[_completed(2)], delete=[_json(200, delete_body)])

    outcome = await _run(fleet)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.DELETE
    assert reason in outcome.error
    assert (outcome.loaded, outcome.deleted, outcome.skipped) == (2, 0, 0)


@pytest.mark.asyncio
async def test_lost_delete_reply_counts_nothing_as_acknowledged():
    """When the source applied a delete but its reply never arrived, the
    move fails and ``deleted`` stays at what was acknowledged (nothing):
    the caller must treat the source's state as unknown, not as intact."""
    lost = _raising(httpx.ReadError("reply lost after the source applied it"))
    fleet = _Fleet(status=[_completed(2)], delete=[lost])

    outcome = await _run(fleet)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.DELETE
    assert "ReadError" in outcome.error
    assert len(fleet.deletes) == 1
    assert (outcome.loaded, outcome.deleted) == (2, 0)


@pytest.mark.asyncio
async def test_source_address_is_resolved_when_the_delete_is_sent():
    """The source is addressed by instance id: one that re-registered at a
    new address while the target was loading gets the delete there, not at
    the address it had when the move was submitted."""
    fleet = _Fleet(status=[_pending(), _pending(), _completed(2)], source_ip="10.0.0.9")
    registry = _registry()
    ctl = MoveController(registry, MoveSettings(move_poll_interval_s=0.005))
    async with fleet.client() as client:
        move_id = await _submit(ctl, client)
        registry.register(_instance("mp-src", "10.0.0.9"))
        outcome = await _settle(ctl, move_id)

    assert outcome.status is MoveStatus.COMPLETED
    assert len(fleet.deletes) == 1  # answered only at the new address
    assert (outcome.loaded, outcome.deleted) == (2, 2)


@pytest.mark.asyncio
async def test_source_gone_from_the_registry_fails_without_delete():
    """A source that deregistered before the delete could be sent fails the
    move in the delete phase; nothing is sent to a stale address."""
    fleet = _Fleet(status=[_pending(), _pending(), _completed(2)])
    registry = _registry()
    ctl = MoveController(registry, MoveSettings(move_poll_interval_s=0.005))
    async with fleet.client() as client:
        move_id = await _submit(ctl, client)
        registry.deregister("mp-src")
        outcome = await _settle(ctl, move_id)

    assert outcome.status is MoveStatus.FAILED
    assert outcome.phase is MovePhase.DELETE
    assert "no longer registered" in outcome.error
    assert (outcome.loaded, outcome.deleted) == (2, 0)
    assert fleet.deletes == []


# -- Lifecycle ---------------------------------------------------------------


@pytest.mark.parametrize(
    "let_it_start",
    [False, True],
    ids=["cancelled-before-first-run", "cancelled-mid-poll"],
)
@pytest.mark.asyncio
async def test_run_cancels_in_flight_moves_on_exit(let_it_start: bool):
    """Leaving run() cancels a move still waiting on its target -- whether
    its task already started polling or never got a turn -- and it ends
    failed in the load phase with the source untouched."""
    fleet = _Fleet(status=itertools.repeat(_pending()))
    ctl = _controller()
    async with fleet.client() as client:
        async with ctl.run(ControllerRuntime(http_client=client)):
            move_id = await _submit(ctl, client)
            assert ctl.in_flight == 1
            if let_it_start:
                await asyncio.sleep(0.03)
                assert fleet.status_polls > 0
        outcome = ctl.get_status(move_id)

    assert outcome is not None
    assert (outcome.status, outcome.phase) == (MoveStatus.FAILED, MovePhase.LOAD)
    assert "cancelled" in outcome.error
    # The cancelled move went through the same retirement as any other.
    assert (ctl.in_flight, ctl.retained) == (0, 1)
    assert ctl.get_status(move_id) == outcome
    assert fleet.deletes == []
