# SPDX-License-Identifier: Apache-2.0
"""Per-request lookup state for the LMCache multiprocess scheduler adapter."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, replace
import enum
import functools
import threading
import time

# First Party
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.token_codec import num_packed_tokens
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)

_QueryPrefetchStatusFuture = MessagingFuture[int | None]
_RpcFuture = MessagingFuture[None] | _QueryPrefetchStatusFuture


class _Phase(enum.Enum):
    """Where one server is in the LOOKUP -> QUERY_PREFETCH_STATUS exchange."""

    LOOKUP_IN_FLIGHT = enum.auto()
    """LOOKUP sent; its reply has not been handled."""

    QUERY_PREFETCH_STATUS_IN_FLIGHT = enum.auto()
    """QUERY_PREFETCH_STATUS sent; its reply has not been handled."""

    NEEDS_REQUERY = enum.auto()
    """The last QUERY_PREFETCH_STATUS reply was None or failed; the next
    scheduler check re-sends it."""

    RESOLVED = enum.auto()
    """The server reported its final hit chunk count."""

    TIMED_OUT = enum.auto()
    """An RPC to the server passed its deadline."""


@dataclass
class _Server:
    """One server's progress through a lookup."""

    client: RequestClient

    phase: _Phase = _Phase.LOOKUP_IN_FLIGHT

    future: _RpcFuture | None = None
    """The RPC in flight, set only in LOOKUP_IN_FLIGHT and
    QUERY_PREFETCH_STATUS_IN_FLIGHT."""

    deadline: float = 0.0
    """``time.monotonic()`` deadline of the RPC in flight, counted from when it
    was sent."""

    scheduler_checked: bool = False
    """Whether a scheduler check ran since the QUERY_PREFETCH_STATUS in flight
    was sent."""

    hit_chunks: int = 0
    """Final hit chunk count, set in RESOLVED."""


class LookupState:
    """One request's lookup on every LMCache server, from LOOKUP to the
    matched token count.

    Each server runs its own chain: LOOKUP, then QUERY_PREFETCH_STATUS as soon
    as that server acknowledges the LOOKUP, re-queried while the server answers
    None. Once every server reports a hit count, the result is the minimum
    across servers; servers that reported more get their excess tail locks
    released.

    RPC completions advance the state through ``on_lookup_finished`` and
    ``on_query_prefetch_status_finished``, which run as future done-callbacks.
    ``on_scheduler_check`` re-sends deferred QUERY_PREFETCH_STATUS requests,
    optionally waits for their replies, and expires RPCs past their deadline.

    Note:
        - A QUERY_PREFETCH_STATUS is sent to a server **only after** that
          server's LOOKUP completed, and at most one is in flight per server.
        - After a None or failed reply, the query is re-sent from the reply
          callback only if a scheduler check ran since it was sent; otherwise
          the next check re-sends it. So each server gets **at most one** new
          QUERY_PREFETCH_STATUS per scheduler check.
        - An RPC's deadline counts from when it is sent, so time spent waiting
          for a scheduler check to re-send a query never counts against the
          server.
        - Once the result is final (all servers resolved, a deadline passed,
          or ``close``), no further RPC is sent.
        - RPC operations are thread-safe because they may happen in different
          threads.
    """

    def __init__(
        self,
        key: IPCCacheServerKey,
        clients: Mapping[str, RequestClient],
        *,
        tp_size: int,
        tokens_per_chunk: int,
        timeout: float,
        blocking_lookup: bool,
    ) -> None:
        """
        Args:
            key: The LOOKUP key; its ``request_id`` names the prefetch job.
            clients: Request client of every server, keyed by server URL.
            tp_size: Tensor parallel size sent with LOOKUP and lock releases.
            tokens_per_chunk: LMCache chunk size in tokens.
            timeout: Seconds each RPC may stay unanswered.
            blocking_lookup: Whether ``on_scheduler_check`` waits for the
                QUERY_PREFETCH_STATUS replies in flight. LOOKUP replies are
                never waited for.
        """
        self._key = key
        self._tp_size = tp_size
        self._tokens_per_chunk = tokens_per_chunk
        self._timeout = timeout
        self._blocking_lookup = blocking_lookup

        # Progress of every server, keyed by URL.
        self._servers = {url: _Server(client) for url, client in clients.items()}

        # Matched token count; None until final, then never changes.
        self._result: int | None = None

        # Guards ``_servers`` and ``_result``; every send happens under it.
        self._lock = threading.Lock()

    def submit(self) -> None:
        """Send LOOKUP to every server."""
        deadline = time.monotonic() + self._timeout
        sent_lookups: list[tuple[str, MessagingFuture[None]]] = []
        with self._lock:
            for url, server in self._servers.items():
                future = server.client.lookup(self._key, self._tp_size)
                server.future = future
                server.deadline = deadline
                sent_lookups.append((url, future))

        for url, future in sent_lookups:
            future.add_done_callback(functools.partial(self.on_lookup_finished, url))

    def close(self) -> None:
        """Stop sending RPCs; an unresolved result becomes 0."""
        with self._lock:
            if self._result is None:
                self._result = 0

    def drain(self) -> str | None:
        """Close, then wait for the LOOKUP and QUERY_PREFETCH_STATUS RPCs
        still in flight.

        Each RPC is waited on until its own deadline.

        Returns:
            The URL of the first server whose RPC missed its deadline, or None
            if every RPC completed.
        """
        self.close()
        with self._lock:
            in_flight = [
                (url, server.future, server.deadline)
                for url, server in self._servers.items()
                if server.future is not None
            ]

        for url, future, deadline in in_flight:
            if not future.wait(timeout=max(0.0, deadline - time.monotonic())):
                return url
        return None

    def on_lookup_finished(self, url: str, future: MessagingFuture[None]) -> None:
        """Handle a completed LOOKUP by querying that server's prefetch status.

        A failed LOOKUP is logged and still followed by a QUERY_PREFETCH_STATUS.
        Stale futures and calls after the result is final are ignored.

        Args:
            url: The server the LOOKUP went to.
            future: The completed LOOKUP future.
        """
        with self._lock:
            server = self._servers[url]
            if (
                self._result is not None
                or server.phase is not _Phase.LOOKUP_IN_FLIGHT
                or server.future is not future
            ):
                return

            sent_queries = self._handle_lookup_reply(url, server, future)

        self._add_query_prefetch_status_callbacks(sent_queries)

    def on_query_prefetch_status_finished(
        self, url: str, future: _QueryPrefetchStatusFuture
    ) -> None:
        """Handle a completed QUERY_PREFETCH_STATUS.

        An int reply resolves the server, and the last one resolves the
        lookup. A None or failed reply is re-queried. Stale futures and calls
        after the result is final are ignored.

        Args:
            url: The server the query went to.
            future: The completed QUERY_PREFETCH_STATUS future.
        """
        with self._lock:
            server = self._servers[url]
            if (
                self._result is not None
                or server.phase is not _Phase.QUERY_PREFETCH_STATUS_IN_FLIGHT
                or server.future is not future
            ):
                return

            sent_queries = self._handle_query_prefetch_status_reply(url, server, future)

        self._add_query_prefetch_status_callbacks(sent_queries)

    def on_scheduler_check(self) -> list[str]:
        """Advance the lookup for one scheduler check.

        Handles replies whose callbacks have not run yet, re-sends deferred
        QUERY_PREFETCH_STATUS requests, waits for the QUERY_PREFETCH_STATUS
        replies in flight when ``blocking_lookup`` is set, and expires RPCs
        past their deadline, which makes the result 0.

        Returns:
            URLs of the servers whose RPC passed its deadline during this
            call. A server is reported at most once.
        """
        with self._lock:
            if self._result is not None:
                return []

            for server in self._servers.values():
                if server.phase is _Phase.QUERY_PREFETCH_STATUS_IN_FLIGHT:
                    server.scheduler_checked = True

            sent_queries = self._handle_completed_rpcs()

            for url, server in self._servers.items():
                if server.phase is _Phase.NEEDS_REQUERY:
                    sent_queries.append(self._send_query_prefetch_status(url, server))

            queries_to_wait: list[tuple[_RpcFuture, float]] = []
            if self._blocking_lookup:
                queries_to_wait = [
                    (server.future, server.deadline)
                    for server in self._servers.values()
                    if server.phase is _Phase.QUERY_PREFETCH_STATUS_IN_FLIGHT
                    and server.future is not None
                ]

        self._add_query_prefetch_status_callbacks(sent_queries)

        for future, deadline in queries_to_wait:
            future.wait(timeout=max(0.0, deadline - time.monotonic()))

        with self._lock:
            if self._result is not None:
                return []

            sent_queries = self._handle_completed_rpcs()

            expired_urls = self._get_expired_servers(time.monotonic())
            for url in expired_urls:
                self._servers[url].phase = _Phase.TIMED_OUT
                self._servers[url].future = None
            if expired_urls:
                self._result = 0

        self._add_query_prefetch_status_callbacks(sent_queries)
        return expired_urls

    def get_result(self) -> int | None:
        """Return the matched token count.

        Returns:
            None while the lookup is unresolved and open; otherwise the
            matched token count, which is 0 after a timeout or ``close``.
        """
        with self._lock:
            return self._result

    def _handle_completed_rpcs(self) -> list[tuple[str, _QueryPrefetchStatusFuture]]:
        """Handle every in-flight RPC that has completed; caller holds the lock."""
        sent_queries: list[tuple[str, _QueryPrefetchStatusFuture]] = []
        for url, server in self._servers.items():
            if self._result is not None:
                break

            future = server.future
            if future is None or not future.query():
                continue

            if server.phase is _Phase.LOOKUP_IN_FLIGHT:
                sent_queries += self._handle_lookup_reply(url, server, future)
            else:
                sent_queries += self._handle_query_prefetch_status_reply(
                    url, server, future
                )
        return sent_queries

    def _handle_lookup_reply(
        self, url: str, server: _Server, future: _RpcFuture
    ) -> list[tuple[str, _QueryPrefetchStatusFuture]]:
        """Send QUERY_PREFETCH_STATUS after the LOOKUP; caller holds the lock."""
        if future.exception_ is not None:
            logger.warning(
                "LOOKUP for request %s to %s failed: %s; querying its status anyway",
                self._key.request_id,
                url,
                future.exception_,
            )

        return [self._send_query_prefetch_status(url, server)]

    def _handle_query_prefetch_status_reply(
        self, url: str, server: _Server, future: _RpcFuture
    ) -> list[tuple[str, _QueryPrefetchStatusFuture]]:
        """Resolve the server or re-query it after a QUERY_PREFETCH_STATUS
        reply; caller holds the lock."""
        if future.exception_ is not None:
            logger.warning(
                "QUERY_PREFETCH_STATUS for request %s to %s failed: %s; re-querying",
                self._key.request_id,
                url,
                future.exception_,
            )
        else:
            reply = future.result(timeout=0)
            if reply is not None:
                server.phase = _Phase.RESOLVED
                server.future = None
                server.hit_chunks = int(reply)
                self._maybe_resolve_lookup_result()
                return []

        if server.scheduler_checked:
            return [self._send_query_prefetch_status(url, server)]

        server.phase = _Phase.NEEDS_REQUERY
        server.future = None
        return []

    def _send_query_prefetch_status(
        self, url: str, server: _Server
    ) -> tuple[str, _QueryPrefetchStatusFuture]:
        """Send QUERY_PREFETCH_STATUS to one server; caller holds the lock."""
        future = server.client.query_prefetch_status(self._key.request_id)
        server.phase = _Phase.QUERY_PREFETCH_STATUS_IN_FLIGHT
        server.future = future
        server.deadline = time.monotonic() + self._timeout
        server.scheduler_checked = False
        return url, future

    def _maybe_resolve_lookup_result(self) -> None:
        """Finalize the final hit tokens once all the servers have returned
        their hit tokens. Free the extra lookup locks if needed. Callers should
        hold the lock."""
        if any(s.phase is not _Phase.RESOLVED for s in self._servers.values()):
            return

        hit_chunks = {url: server.hit_chunks for url, server in self._servers.items()}
        min_chunks = min(hit_chunks.values())
        if min_chunks != max(hit_chunks.values()):
            logger.warning(
                "[req=%s] LMCache hit mismatch across servers: %s → take min=%d",
                self._key.request_id,
                hit_chunks,
                min_chunks,
            )

            # Chunks past the minimum stay locked on the servers that
            # reported them but will never be retrieved.
            for url, server in self._servers.items():
                if server.hit_chunks <= min_chunks:
                    continue

                tail_key = replace(
                    self._key,
                    start=min_chunks * self._tokens_per_chunk,
                    end=min(
                        server.hit_chunks * self._tokens_per_chunk,
                        num_packed_tokens(self._key.token_bytes),
                    ),
                )
                server.client.free_lookup_locks(tail_key, self._tp_size)

        self._result = min_chunks * self._tokens_per_chunk

    def _get_expired_servers(self, now: float) -> list[str]:
        """Return the servers whose LOOKUP or QUERY_PREFETCH_STATUS is still
        unanswered at its deadline; caller holds the lock."""
        return [
            url
            for url, server in self._servers.items()
            if server.future is not None
            and not server.future.query()
            and now >= server.deadline
        ]

    def _add_query_prefetch_status_callbacks(
        self, sent_queries: list[tuple[str, _QueryPrefetchStatusFuture]]
    ) -> None:
        """Register reply callbacks for sent QUERY_PREFETCH_STATUS requests;
        caller does not hold the lock."""
        for url, future in sent_queries:
            future.add_done_callback(
                functools.partial(self.on_query_prefetch_status_finished, url)
            )
