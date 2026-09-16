# SPDX-License-Identifier: Apache-2.0
"""Coordinator-driven cache move: a target-pulled copy, then a source delete.

MP servers never write into a peer, so a move is driven from the target: the
coordinator submits a warm prefetch there, polls it itself until the target
reports which keys it loaded, and only then deletes those keys -- and only
those -- from the source's L1. The coordinator polls rather than the client
because the delete is its own action and must not depend on whether a client
ever asks. Nothing is rolled back and nothing holds the target's copy; see
``docs/design/v1/mp_coordinator/cache_move.md`` for the contract and the
failure table.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING
import asyncio
import time
import uuid

# Third Party
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import MovePhase, MoveStatus
from lmcache.v1.mp_coordinator.controllers.base import Controller, ControllerRuntime
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
from lmcache.v1.multiprocess.cache_control.object_service import (
    MAX_DELETE_BATCH,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.discovery import Registry
    from lmcache.v1.mp_coordinator.views.base import View
    from lmcache.v1.mp_coordinator.views.instance_registry import MPInstance

logger = init_logger(__name__)

# Expired results are swept at least this often even when the TTL is longer.
_MAX_SWEEP_INTERVAL_S = 30.0

# Wire statuses kept local to avoid coupling to the warm-prefetch implementation.
_TARGET_PENDING = "pending"
_TARGET_COMPLETED = "completed"
_TARGET_NOOP = "noop"

_CANCELLED_ERROR = "cancelled: the coordinator shut down"


class MoveSettings(BaseModel):
    """The ``MPCoordinatorConfig.extra_config`` keys a move reads.

    Strict: each value must be a finite positive number (a positive integer
    for the limit), never a string or a bool. Keys this model does not name
    belong to other controllers and are ignored.

    Attributes:
        move_poll_interval_s: Seconds between polls of the target's prefetch
            status.
        move_completion_timeout_s: Seconds the coordinator waits for the
            target to report; a move that times out deletes nothing. Covers
            the status requests, their replies and the sleeps between them
            -- not the submit, the source delete, or the target's own load.
        move_result_ttl_s: Seconds a settled move's outcome stays readable.
        move_result_limit: Most settled outcomes kept at once.
    """

    model_config = ConfigDict(
        strict=True, extra="ignore", allow_inf_nan=False, frozen=True
    )

    move_poll_interval_s: PositiveFloat = 0.5
    move_completion_timeout_s: PositiveFloat = 600.0
    move_result_ttl_s: PositiveFloat = 600.0
    move_result_limit: PositiveInt = 1000


_DEFAULT_SETTINGS = MoveSettings()


class MoveSubmitError(Exception):
    """The target gave the move no prefetch job to drive.

    Raised by :meth:`MoveController.submit_move` when the target is not
    registered, or its reply to the prefetch submit is not a JSON object
    carrying a ``request_id`` -- a ``noop`` included: the coordinator
    resolved whole chunks but the target submitted nothing, so the two
    disagree on chunking. No move is recorded; the API reports it as an
    upstream (502) error.
    """


class _MoveFailed(Exception):
    """The move cannot continue; the message says why."""


class _SubmitReply(BaseModel):
    """The target's ``POST /cache/prefetches`` body, as far as a move reads it."""

    model_config = ConfigDict(strict=True, extra="ignore")

    request_id: str = ""
    status: str = ""


class _StatusReply(BaseModel):
    """The target's ``GET /cache/prefetches/{id}`` body, as far as a move reads it.

    Strict, so a count that arrives as a bool or a float is refused rather
    than coerced.

    Attributes:
        status: The prefetch's state; only ``completed`` is acted on.
        found_keys: Keys the prefetch loaded.
        total_keys: Keys the target resolved the sequence to.
        missing_key_indices: Positions not loaded; ``None`` on a server too
            old to report per-key outcome.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    status: str
    found_keys: NonNegativeInt = 0
    total_keys: NonNegativeInt = 0
    missing_key_indices: list[NonNegativeInt] | None = None


class _DeleteReply(BaseModel):
    """The source's ``DELETE /cache/objects`` body, as far as a move reads it.

    Attributes:
        deleted: Keys the source removed.
        skipped: Keys the source refused because they were locked.
        ok: ``False`` when the source hit a structured failure (``error``).
        error: Why, when ``ok`` is ``False``.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    deleted: NonNegativeInt
    skipped: NonNegativeInt
    ok: bool = True
    error: str = ""


def _missing_positions(reply: _StatusReply, total: int) -> set[int]:
    """Return the positions the target did not load, once its report checks out.

    Args:
        reply: The target's completed status.
        total: Keys the coordinator resolved. The target must report the same
            number, or its positions mean nothing here. Matching counts do not
            establish matching key identities; fleet key resolution must agree.

    Returns:
        The positions, in resolved key order, the target did not load.

    Raises:
        _MoveFailed: ``total_keys`` disagrees with ``total`` (a chunk-size
            mismatch), ``missing_key_indices`` is absent (a server too old
            to report per key), a position repeats or is out of range, or
            the counts disagree with the positions.
    """
    if reply.total_keys != total:
        raise _MoveFailed(
            f"target reports {reply.total_keys} key(s) for a sequence the "
            f"coordinator resolved to {total}; check that --chunk-size and "
            "--hash-algorithm match the coordinator's"
        )
    if reply.missing_key_indices is None:
        raise _MoveFailed(
            "target did not report per-key outcome (missing_key_indices); "
            "its LMCache is too old to be a move target"
        )
    missing = set(reply.missing_key_indices)
    if len(missing) != len(reply.missing_key_indices) or any(
        i >= total for i in missing
    ):
        raise _MoveFailed(
            "target reported a repeated or out-of-range position in missing_key_indices"
        )
    if reply.found_keys + len(missing) != total:
        raise _MoveFailed(
            f"target's counts disagree: found_keys={reply.found_keys}, "
            f"{len(missing)} missing, total_keys={total}"
        )
    return missing


def _acknowledged(reply: _DeleteReply, batch_size: int) -> tuple[int, int]:
    """Return ``(deleted, skipped)`` once the source's reply to a batch checks out.

    Args:
        reply: The source's reply.
        batch_size: Keys the batch asked to delete.

    Returns:
        The counts the source acknowledged.

    Raises:
        _MoveFailed: The source flagged ``ok: false``, or the two counts
            exceed the batch (a key the source never held counts in neither,
            so they may fall short, never over).
    """
    if not reply.ok:
        raise _MoveFailed(f"source reported the delete failed: {reply.error!r}")
    if reply.deleted + reply.skipped > batch_size:
        raise _MoveFailed(
            f"source acknowledged {reply.deleted} deleted + {reply.skipped} "
            f"skipped for a batch of {batch_size} key(s)"
        )
    return reply.deleted, reply.skipped


@dataclass(frozen=True)
class MoveSpec:
    """What one move is asked to do.

    Attributes:
        source_instance_id: The MP server the chunks leave. Its address is
            resolved when the delete is sent, so it must be registered then.
        target_instance_id: The MP server the chunks arrive at; must be
            registered at submit.
        model_name: Model whose layout the target uses to allocate L1.
        world_size: World size selecting the layout and the per-rank fan-out.
        cache_salt: Per-tenant isolation salt applied to the produced keys.
        keys: The keys the token sequence resolves to, in resolved order
            (chunk-major, then rank) -- the order the target's status reports
            positions in.
        chunks: Whole chunks the sequence resolved to.
        keep_source: When ``True`` the source keeps its chunks: a copy.
    """

    source_instance_id: str
    target_instance_id: str
    model_name: str
    world_size: int
    cache_salt: str
    keys: list[ObjectKey]
    chunks: int
    keep_source: bool


@dataclass(frozen=True)
class MoveOutcome:
    """Where a move stands, as reported by :meth:`MoveController.get_status`.

    Key counts are per-rank keys (chunks times the per-rank fan-out);
    ``requested`` counts chunks.

    Attributes:
        move_id: The move.
        source_instance_id: The server the chunks leave.
        target_instance_id: The server the chunks arrive at.
        status: ``PENDING`` while the move runs, then ``COMPLETED`` or
            ``FAILED``.
        phase: ``LOAD`` until the target reports (the source is untouched),
            ``DELETE`` from then on. For a failed move, the phase it failed
            in.
        requested: Whole chunks the move resolved to.
        loaded: Keys the target reported loading into its L1. A report of
            what the prefetch loaded, not proof the keys are still there.
        missing: Keys the target did not load -- not found on any source it
            can read, or already resident there. Never deleted from the
            source.
        deleted: Keys whose removal the source acknowledged (``0`` for a
            copy). A delete whose reply was lost may have removed more.
        skipped: Keys the source refused to delete because they were locked.
        error: Why the move failed; empty unless ``status`` is ``FAILED``.
    """

    move_id: str
    source_instance_id: str
    target_instance_id: str
    status: MoveStatus
    phase: MovePhase
    requested: int
    loaded: int = 0
    missing: int = 0
    deleted: int = 0
    skipped: int = 0
    error: str = ""


@dataclass
class _MoveJob:
    """A running move: what drives it, and the outcome it reports meanwhile.

    Attributes:
        spec: What to move.
        target: The target as registered at submit; the prefetch job belongs
            to that incarnation.
        prefetch_request_id: The target's warm-prefetch id.
        outcome: The live report, replaced as the move advances.
        last_poll_error: The last transport error a status poll hit, kept so
            a timeout can say the target was unreachable; cleared by a poll
            that gets through.
    """

    spec: MoveSpec
    target: MPInstance
    prefetch_request_id: str
    outcome: MoveOutcome
    last_poll_error: str = ""


class MoveController(Controller):
    """Drive cross-server moves and report where each stands.

    Everything runs on the coordinator's event loop -- the tables are touched
    only by request handlers and by the move tasks -- so no lock is needed.
    A running move holds its spec (the resolved keys, not the token ids);
    once it settles, only its outcome is kept, re-readable for
    ``move_result_ttl_s`` seconds (reads do not extend that) and bounded to
    ``move_result_limit`` outcomes, oldest evicted first. Expired outcomes
    are swept while :meth:`run` is active, so nothing depends on a client
    polling. In-flight moves are bounded only by ``move_completion_timeout_s``
    plus the source delete.

    Args:
        registry: Fleet membership, read at submit for the target and when
            a delete is sent for the source, so the source is addressed at
            its current registration.
        settings: Timing and retention; see :class:`MoveSettings`.
    """

    def __init__(
        self, registry: InstanceRegistry, settings: MoveSettings = _DEFAULT_SETTINGS
    ) -> None:
        self._registry = registry
        self._settings = settings
        # Running moves only.
        self._jobs: dict[str, _MoveJob] = {}
        # Settled moves: outcome and the monotonic time it expires, oldest first.
        self._results: OrderedDict[str, tuple[MoveOutcome, float]] = OrderedDict()
        # Only tasks still running: each removes itself when it finishes.
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @classmethod
    def from_config(
        cls, config: MPCoordinatorConfig, views: Registry[View]
    ) -> MoveController:
        """Build from ``config.extra_config`` and the fleet's registry.

        Args:
            config: The coordinator configuration; its ``extra_config`` is
                read as a :class:`MoveSettings`.
            views: Supplies the ``InstanceRegistry`` the servers are
                resolved from.

        Returns:
            The controller.

        Raises:
            pydantic.ValidationError: If a setting is set but invalid (a
                ``ValueError``).
        """
        return cls(
            views.get(InstanceRegistry),
            MoveSettings.model_validate(dict(config.extra_config)),
        )

    @property
    def in_flight(self) -> int:
        """Moves not yet in a terminal state.

        An observability counter (for tests and metrics); driving a move
        never needs it.

        Returns:
            How many moves are still ``PENDING``.
        """
        return len(self._jobs)

    @property
    def retained(self) -> int:
        """Settled moves whose outcome is still readable.

        An observability counter (for tests and metrics); driving a move
        never needs it.

        Returns:
            How many outcomes are kept, expired ones not yet swept included.
        """
        return len(self._results)

    async def submit_move(
        self, spec: MoveSpec, token_ids: list[int], http_client: httpx.AsyncClient
    ) -> str:
        """Submit the target's warm prefetch and start driving the move.

        Returns once the target has accepted the prefetch; the rest --
        polling the target, then deleting from the source -- runs as a task.
        Poll :meth:`get_status` with the returned id.

        Args:
            spec: What to move. ``spec.target_instance_id`` must be
                registered now; ``spec.source_instance_id`` when the delete
                is sent, and a source that re-registered meanwhile is
                reached at its new address.
            token_ids: The tokens ``spec.keys`` were resolved from, forwarded
                to the target's warm prefetch verbatim and not kept.
            http_client: Shared async client for outbound coordinator calls;
                the move keeps using it after this call returns.

        Returns:
            The move id.

        Raises:
            httpx.HTTPError: If the target is unreachable or rejects the
                submit. No move is recorded.
            MoveSubmitError: If the target is not registered, or its reply
                gives the move no prefetch job to drive. No move is recorded.
        """
        target = self._registry.get(spec.target_instance_id)
        if target is None:
            raise MoveSubmitError(
                f"target {spec.target_instance_id!r} is not registered"
            )
        url = f"http://{target.ip}:{target.http_port}/cache/prefetches"
        body = {
            "model_name": spec.model_name,
            "world_size": spec.world_size,
            "token_ids": token_ids,
            "cache_salt": spec.cache_salt,
        }
        resp = await http_client.post(url, json=body)
        resp.raise_for_status()
        try:
            reply = _SubmitReply.model_validate(resp.json())
        except ValueError as exc:
            raise MoveSubmitError(
                f"target {target.instance_id!r} answered the prefetch submit "
                f"with an unusable body: {exc}"
            ) from None
        if reply.status == _TARGET_NOOP:
            raise MoveSubmitError(
                f"target {target.instance_id!r} submitted no prefetch for a "
                f"sequence the coordinator resolved to {spec.chunks} chunk(s); "
                "check that its --chunk-size matches the coordinator's"
            )
        if not reply.request_id:
            raise MoveSubmitError(
                f"target {target.instance_id!r} answered the prefetch submit "
                "without a request_id"
            )

        move_id = uuid.uuid4().hex
        job = _MoveJob(
            spec=spec,
            target=target,
            prefetch_request_id=reply.request_id,
            outcome=MoveOutcome(
                move_id=move_id,
                source_instance_id=spec.source_instance_id,
                target_instance_id=spec.target_instance_id,
                status=MoveStatus.PENDING,
                phase=MovePhase.LOAD,
                requested=spec.chunks,
            ),
        )
        self._jobs[move_id] = job
        task = asyncio.create_task(self._drive(job, http_client))
        self._tasks[move_id] = task
        task.add_done_callback(lambda _done: self._tasks.pop(move_id, None))
        logger.debug(
            "Move %s submitted: %s -> %s, %d chunk(s), %d key(s), keep_source=%s",
            move_id,
            spec.source_instance_id,
            spec.target_instance_id,
            spec.chunks,
            len(spec.keys),
            spec.keep_source,
        )
        return move_id

    def get_status(self, move_id: str) -> MoveOutcome | None:
        """Report where a move stands.

        A running move is reported live. A settled move's outcome stays
        readable until its TTL runs out or the result limit evicts it;
        reading it does not extend the TTL.

        Args:
            move_id: The id :meth:`submit_move` returned.

        Returns:
            The outcome, or ``None`` for an id that is unknown, expired, or
            evicted.
        """
        job = self._jobs.get(move_id)
        if job is not None:
            return job.outcome
        entry = self._results.get(move_id)
        if entry is None:
            return None
        outcome, expires_at = entry
        return outcome if time.monotonic() < expires_at else None

    @asynccontextmanager
    async def run(self, runtime: ControllerRuntime) -> AsyncIterator[None]:
        """Serve, sweeping expired outcomes, then cancel every move in flight.

        A cancelled move ends ``FAILED`` with ``error`` saying so, in the
        phase it was in; its source was not touched unless the delete had
        already begun.

        Args:
            runtime: Unused -- a move keeps the client it was submitted with.
        """
        sweeper = asyncio.create_task(self._sweep_loop())
        try:
            yield
        finally:
            sweeper.cancel()
            tasks = list(self._tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(sweeper, *tasks, return_exceptions=True)
            # A task cancelled before it first ran never reached its own
            # handler, and nothing will finish its job now.
            for job in list(self._jobs.values()):
                self._fail(job, _CANCELLED_ERROR)
                self._finish(job)

    async def _sweep_loop(self) -> None:
        """Drop expired outcomes on a timer, so idle results do not linger."""
        interval = min(self._settings.move_result_ttl_s, _MAX_SWEEP_INTERVAL_S)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            for move_id, (_outcome, expires_at) in list(self._results.items()):
                if now >= expires_at:
                    del self._results[move_id]

    def _finish(self, job: _MoveJob) -> None:
        """Retire a settled job: keep its outcome, drop everything else.

        The single exit for every terminal path -- completion, failure,
        cancellation -- so the spec is always released and the outcome
        always retained under the same TTL and limit.
        """
        outcome = job.outcome
        if outcome.status is MoveStatus.PENDING:
            raise RuntimeError(f"move {outcome.move_id} finished while still pending")
        self._jobs.pop(outcome.move_id, None)
        self._results[outcome.move_id] = (
            outcome,
            time.monotonic() + self._settings.move_result_ttl_s,
        )
        while len(self._results) > self._settings.move_result_limit:
            self._results.popitem(last=False)

    async def _drive(self, job: _MoveJob, http_client: httpx.AsyncClient) -> None:
        """Wait for the target, then delete from the source what it loaded."""
        try:
            missing = await self._await_target(job, http_client)
            loaded = [key for i, key in enumerate(job.spec.keys) if i not in missing]
            job.outcome = replace(
                job.outcome,
                phase=MovePhase.DELETE,
                loaded=len(loaded),
                missing=len(missing),
            )
            if loaded and not job.spec.keep_source:
                await self._delete_from_source(job, http_client, loaded)
            job.outcome = replace(job.outcome, status=MoveStatus.COMPLETED)
            logger.debug(
                "Move %s completed: %d/%d key(s) loaded on %s; "
                "%d deleted from %s, %d skipped",
                job.outcome.move_id,
                job.outcome.loaded,
                len(job.spec.keys),
                job.spec.target_instance_id,
                job.outcome.deleted,
                job.spec.source_instance_id,
                job.outcome.skipped,
            )
        except _MoveFailed as exc:
            self._fail(job, str(exc))
        except asyncio.CancelledError:
            self._fail(job, _CANCELLED_ERROR)
            raise
        except Exception as exc:  # a bug fails the move, not the app
            logger.exception("Move %s hit an unexpected error", job.outcome.move_id)
            self._fail(job, f"internal error: {exc}")
        finally:
            self._finish(job)

    async def _await_target(
        self, job: _MoveJob, http_client: httpx.AsyncClient
    ) -> set[int]:
        """Poll the target until its warm prefetch completes.

        The deadline bounds the coordinator's wait -- status requests, their
        replies and the sleeps between them -- by running the poll loop
        under ``asyncio.wait_for``, which cancels it mid-request,
        mid-response or mid-sleep when time is up; a per-request HTTP
        timeout only bounds the gap between two pieces of a reply. It does
        not cancel the load the target already started.

        Returns:
            Positions, in ``spec.keys`` order, the target did not load.

        Raises:
            _MoveFailed: The target dropped the job, answered non-2xx or
                unusably, its report does not line up with the resolved
                keys, or the deadline passed (whether the target was still
                loading, unreachable, or finished too late).
        """
        timeout = self._settings.move_completion_timeout_s
        try:
            return await asyncio.wait_for(
                self._poll_target(job, http_client), timeout=timeout
            )
        except asyncio.TimeoutError:
            if job.last_poll_error:
                raise _MoveFailed(
                    f"target unreachable for status polls until the {timeout:g}s "
                    f"deadline: {job.last_poll_error}"
                ) from None
            raise _MoveFailed(
                f"target did not finish loading within {timeout:g}s"
            ) from None
        except httpx.HTTPError as exc:
            raise _MoveFailed(
                f"target status poll failed: {type(exc).__name__}: {exc}"
            ) from None

    async def _poll_target(
        self, job: _MoveJob, http_client: httpx.AsyncClient
    ) -> set[int]:
        """The poll loop :meth:`_await_target` runs under its timeout.

        A transport error on one poll is not a verdict on the load, so it is
        noted on the job and retried on the next interval.

        A completion first seen after the deadline is refused here as well,
        because ``wait_for`` alone does not guarantee it: on Python 3.10 and
        3.11 a result that lands in the same loop iteration as the timeout
        wins, and a loop stalled by other work (a large token-hashing
        request, say) makes that iteration arbitrarily late; on 3.12+ the
        same holds for a reply parsed in this task's own synchronous tail.

        Returns:
            Positions, in ``spec.keys`` order, the target did not load.

        Raises:
            _MoveFailed: The target dropped the job, answered unusably, its
                report does not line up with the resolved keys, or it
                completed after the deadline.
            httpx.HTTPError: The target answered non-2xx (other than 404).
        """
        url = (
            f"http://{job.target.ip}:{job.target.http_port}"
            f"/cache/prefetches/{job.prefetch_request_id}"
        )
        timeout = self._settings.move_completion_timeout_s
        deadline = time.monotonic() + timeout
        while True:
            try:
                resp = await http_client.get(url)
            except httpx.TransportError as exc:
                job.last_poll_error = f"{type(exc).__name__}: {exc}"
                logger.debug(
                    "Move %s: status poll failed (%s); retrying",
                    job.outcome.move_id,
                    job.last_poll_error,
                )
                await asyncio.sleep(self._settings.move_poll_interval_s)
                continue
            job.last_poll_error = ""
            if resp.status_code == 404:
                raise _MoveFailed(
                    "target no longer knows the prefetch job "
                    "and may have loaded the keys, but this move has not "
                    "requested source deletion"
                )
            resp.raise_for_status()
            try:
                reply = _StatusReply.model_validate(resp.json())
            except ValueError as exc:
                raise _MoveFailed(
                    f"target returned a malformed status: {exc}"
                ) from None
            if reply.status == _TARGET_COMPLETED:
                if time.monotonic() >= deadline:
                    raise _MoveFailed(
                        f"target finished loading only after the {timeout:g}s "
                        "deadline; it keeps what it loaded and the source was "
                        "not touched"
                    )
                return _missing_positions(reply, len(job.spec.keys))
            if reply.status != _TARGET_PENDING:
                raise _MoveFailed(
                    f"target returned an unexpected prefetch status: {reply.status!r}"
                )
            await asyncio.sleep(self._settings.move_poll_interval_s)

    async def _delete_from_source(
        self, job: _MoveJob, http_client: httpx.AsyncClient, keys: list[ObjectKey]
    ) -> None:
        """``DELETE /cache/objects`` (L1, never forced) on the source, batched.

        The source's address is read from the registry for every batch, so
        the delete follows a re-registration and stops at a deregistration.
        Each reply is validated in full before its counts are added, so a
        failure part-way leaves ``deleted`` / ``skipped`` at exactly what
        the earlier batches acknowledged.

        Raises:
            _MoveFailed: The source is no longer registered, unreachable,
                rejected a batch, or answered unusably. The batch that
                failed may or may not have been applied.
        """
        for start in range(0, len(keys), MAX_DELETE_BATCH):
            source = self._registry.get(job.spec.source_instance_id)
            if source is None:
                raise _MoveFailed(
                    f"source {job.spec.source_instance_id!r} is no longer "
                    "registered; the delete stopped before this batch"
                )
            batch = keys[start : start + MAX_DELETE_BATCH]
            url = f"http://{source.ip}:{source.http_port}/cache/objects"
            payload = {
                "keys": [asdict(key.to_encoded_object_key()) for key in batch],
                "tier": Tier.L1.value,
                "force": False,
            }
            try:
                # httpx ``.delete`` can't take ``json=``; use ``request(...)``.
                resp = await http_client.request("DELETE", url, json=payload)
                resp.raise_for_status()
                reply = _DeleteReply.model_validate(resp.json())
            except httpx.HTTPError as exc:
                raise _MoveFailed(
                    f"source delete failed: {type(exc).__name__}: {exc}"
                ) from None
            except ValueError as exc:
                raise _MoveFailed(
                    f"source returned a malformed delete reply: {exc}"
                ) from None
            deleted, skipped = _acknowledged(reply, len(batch))
            job.outcome = replace(
                job.outcome,
                deleted=job.outcome.deleted + deleted,
                skipped=job.outcome.skipped + skipped,
            )

    @staticmethod
    def _fail(job: _MoveJob, error: str) -> None:
        """Mark ``job`` failed, in whatever phase it is in, with ``error``."""
        job.outcome = replace(job.outcome, status=MoveStatus.FAILED, error=error)
        logger.warning(
            "Move %s failed in the %s phase: %s",
            job.outcome.move_id,
            job.outcome.phase.value,
            error,
        )
