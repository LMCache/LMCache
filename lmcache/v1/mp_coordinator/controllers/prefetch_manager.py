# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side warm-prefetch dispatch.

Forwards a client's warm-prefetch request to one named MP server and proxies
its status. The MP server's ``POST /cache/prefetches`` submits the load and returns
a ``request_id`` immediately (the load runs in the server's storage-manager
thread); the coordinator relays that id back to the client, which then polls
``GET /cache/prefetches/{instance_id}/{request_id}`` on the coordinator until the
server reports completion. There is no background polling on either side -- the
submit and status calls are quick and the client drives completion on demand.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
import threading
import time

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.controllers.base import Controller
from lmcache.v1.mp_coordinator.metric_sources import register_metric_source
from lmcache.v1.mp_coordinator.utils.histogram import (
    CHUNK_COUNT_BOUNDS,
    DURATION_SECONDS_BOUNDS,
    Histogram,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.views.instance_registry import MPInstance

logger = init_logger(__name__)


@dataclass
class PrefetchStats:
    """The prefetches relayed since this process started.

    Attributes:
        requests_submitted: Prefetch requests by result: ``submitted``,
            ``noop`` (shorter than one chunk), ``unknown_server`` (not
            registered) or ``server_error`` (refused or unreachable).
            Every result is present, from 0.
        chunks_requested_per_request: Chunks each answered request asked
            to warm.
        submit_duration_seconds: Time for the server to accept a prefetch.
        requests_in_flight: Prefetches submitted and not yet seen
            completed, by ``instance_id``.
        completion_duration_seconds: Submit to the first poll that saw it
            completed; includes the caller's poll interval, so an upper
            bound.
        keys_requested: Keys the completed prefetches asked to warm.
        keys_found: Keys they found in L2 and loaded into L1.
        status_polls: Status polls by result: ``pending``, ``completed``,
            ``unknown_request`` (the server lost the job or it was already
            collected) or ``server_error``. Every result is present, from 0.
        requests_abandoned: Prefetches never polled to completion before
            being forgotten: the caller gave up, or the server left.
    """

    requests_submitted: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            ("submitted", "noop", "unknown_server", "server_error"), 0
        )
    )
    chunks_requested_per_request: Histogram = field(
        default_factory=lambda: Histogram.with_bounds(CHUNK_COUNT_BOUNDS)
    )
    submit_duration_seconds: Histogram = field(
        default_factory=lambda: Histogram.with_bounds(DURATION_SECONDS_BOUNDS)
    )
    requests_in_flight: dict[str, int] = field(default_factory=dict)
    completion_duration_seconds: Histogram = field(
        default_factory=lambda: Histogram.with_bounds(
            (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
        )
    )
    keys_requested: int = 0
    keys_found: int = 0
    status_polls: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            ("pending", "completed", "unknown_request", "server_error"), 0
        )
    )
    requests_abandoned: int = 0


class PrefetchManager(Controller):
    """Submit warm-prefetch requests to MP servers and proxy their status.

    The coordinator only relays, so to time prefetches it remembers each
    submitted one, ``(instance_id, request_id)`` to its submit time, until
    a poll sees it completed or the server no longer knows it. One never
    polled to completion is forgotten after ``abandon_after_seconds`` and
    counted as abandoned. The table is as big as the prefetches in flight
    and is not checkpointed.

    Args:
        abandon_after_seconds: How long a submitted prefetch is remembered
            without a completed poll.
    """

    def __init__(self, abandon_after_seconds: float = 600.0) -> None:
        self._abandon_after_seconds = abandon_after_seconds
        # Locked: handlers record from the event loop while metrics read
        # from another thread, and reading expires old entries.
        self._lock = threading.Lock()
        self._submitted_at: dict[tuple[str, str], float] = {}
        self._stats = PrefetchStats()
        register_metric_source(self)

    async def submit_prefetch(
        self,
        target: MPInstance,
        http_client: httpx.AsyncClient,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
    ) -> dict[str, Any]:
        """``POST /cache/prefetches`` to ``target`` and return its JSON reply.

        Args:
            target: The MP server to warm.
            http_client: Shared async client for outbound coordinator calls.
            model_name: Model whose layout the target uses to allocate L1.
            world_size: World size selecting the layout and per-rank fan-out.
            token_ids: Prompt tokens whose complete chunks should be warmed.
            cache_salt: Per-tenant isolation salt applied to the produced keys.

        Returns:
            The server's reply, e.g. ``{"request_id", "chunks", "status"}`` or
            ``{"chunks": 0, "status": "noop"}``.

        Raises:
            httpx.HTTPError: If the target is unreachable or returns non-2xx.
            ValueError: If the target's reply is not JSON.
        """
        url = f"http://{target.ip}:{target.http_port}/cache/prefetches"
        body = {
            "model_name": model_name,
            "world_size": world_size,
            "token_ids": token_ids,
            "cache_salt": cache_salt,
        }
        submitted_at = time.monotonic()
        try:
            resp = await http_client.post(url, json=body)
            resp.raise_for_status()
            reply = resp.json()
        except (httpx.HTTPError, ValueError):
            self._count_submit("server_error")
            raise
        accepted_at = time.monotonic()
        submit_result = "noop" if reply.get("status") == "noop" else "submitted"
        request_id = reply.get("request_id")
        with self._lock:
            self._forget_abandoned_locked()
            self._stats.requests_submitted[submit_result] += 1
            self._stats.chunks_requested_per_request.observe(reply.get("chunks", 0))
            self._stats.submit_duration_seconds.observe(accepted_at - submitted_at)
            # Without a request id it cannot be polled, so there is nothing
            # to track.
            if submit_result == "submitted" and request_id:
                self._submitted_at[(target.instance_id, request_id)] = accepted_at
        logger.info(
            "Prefetch submitted to %s: %d tokens", target.instance_id, len(token_ids)
        )
        return reply

    def record_unknown_server(self) -> None:
        """Count a prefetch asked of a server that is not registered.

        Called by the submit endpoint, which resolves the server before
        anything reaches :meth:`submit_prefetch`.
        """
        self._count_submit("unknown_server")

    async def get_status(
        self,
        target: MPInstance,
        http_client: httpx.AsyncClient,
        request_id: str,
    ) -> tuple[int, dict[str, Any]]:
        """Proxy ``GET /cache/prefetches/{request_id}`` on ``target``.

        Args:
            target: The MP server holding the job.
            http_client: Shared async client for outbound coordinator calls.
            request_id: The id returned by :meth:`submit_prefetch`.

        Returns:
            ``(status_code, body)`` from the server (e.g. 200 with a status
            body, or 404 for an unknown id), relayed verbatim to the caller.

        Raises:
            httpx.HTTPError: If the target is unreachable (transport error).
        """
        url = f"http://{target.ip}:{target.http_port}/cache/prefetches/{request_id}"
        try:
            resp = await http_client.get(url)
        except httpx.HTTPError:
            self._count_poll(target.instance_id, request_id, "server_error", {})
            raise
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        if resp.status_code == 404:
            poll_result = "unknown_request"
        elif resp.status_code == 200:
            poll_result = (
                "completed" if body.get("status") == "completed" else "pending"
            )
        else:
            poll_result = "server_error"
        self._count_poll(target.instance_id, request_id, poll_result, body)
        return resp.status_code, body

    def stats(self) -> PrefetchStats:
        """Return a snapshot of :class:`PrefetchStats`, first forgetting
        prefetches older than ``abandon_after_seconds``."""
        with self._lock:
            self._forget_abandoned_locked()
            in_flight: dict[str, int] = {}
            for instance_id, _ in self._submitted_at:
                in_flight[instance_id] = in_flight.get(instance_id, 0) + 1
            return replace(
                self._stats,
                requests_submitted=dict(self._stats.requests_submitted),
                chunks_requested_per_request=(
                    self._stats.chunks_requested_per_request.copy()
                ),
                submit_duration_seconds=self._stats.submit_duration_seconds.copy(),
                requests_in_flight=in_flight,
                completion_duration_seconds=(
                    self._stats.completion_duration_seconds.copy()
                ),
                status_polls=dict(self._stats.status_polls),
            )

    def _count_submit(self, submit_result: str) -> None:
        """Count one submit that reached no prefetch on a server."""
        with self._lock:
            self._stats.requests_submitted[submit_result] += 1

    def _count_poll(
        self,
        instance_id: str,
        request_id: str,
        poll_result: str,
        body: dict[str, Any],
    ) -> None:
        """Count one status poll, and stop tracking a prefetch that is done
        or that the server no longer knows."""
        now = time.monotonic()
        with self._lock:
            self._forget_abandoned_locked()
            self._stats.status_polls[poll_result] += 1
            if poll_result not in ("completed", "unknown_request"):
                return
            submitted_at = self._submitted_at.pop((instance_id, request_id), None)
            if poll_result == "completed" and submitted_at is not None:
                self._stats.completion_duration_seconds.observe(now - submitted_at)
                self._stats.keys_requested += body.get("total_keys", 0)
                self._stats.keys_found += body.get("found_keys", 0)

    def _forget_abandoned_locked(self) -> None:
        """Forget, and count, prefetches older than ``abandon_after_seconds``.
        Call holding ``self._lock``."""
        submitted_before = time.monotonic() - self._abandon_after_seconds
        abandoned_prefetches = [
            prefetch
            for prefetch, submitted_at in self._submitted_at.items()
            if submitted_at < submitted_before
        ]
        for prefetch in abandoned_prefetches:
            del self._submitted_at[prefetch]
        self._stats.requests_abandoned += len(abandoned_prefetches)
