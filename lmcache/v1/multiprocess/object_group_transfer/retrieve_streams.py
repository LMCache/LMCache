# SPDX-License-Identifier: Apache-2.0
"""The extra streams of one retrieve call, tied to the caller's stream,
which is the one the request is ordered on.

- gate: the scatter stream waits for the caller's stream, so no kernel runs
  before the block IDs are staged and the engine has released the blocks.
- join: the caller's stream waits for every stream the call used, so what
  the caller queues next runs after the last load and the last kernel.
"""

# Standard
from contextlib import contextmanager
from typing import Any, Iterator

# First Party
from lmcache.v1.multiprocess.object_group_transfer.engines import (
    TransferEngine,
    staging_lane,
)
from lmcache.v1.platform import stream as platform_stream
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.staging_lane import StagingLane
import lmcache.lmcache_native as lmcache_native


class RetrieveStreams:
    """Hands out staging lanes for one retrieve call and tracks which
    streams the call used.

    Args:
        cache_context: Cache context of the KV cache.
        scatter_stream: The context's scatter stream.
    """

    def __init__(self, cache_context: BaseCacheContext, scatter_stream: Any) -> None:
        self._cache_context = cache_context
        self._caller_stream = platform_stream.current_stream(cache_context.device)
        self.scatter_stream = scatter_stream
        self._lanes: dict[TransferEngine, StagingLane] = {}

    def lane(self, engine: TransferEngine) -> StagingLane:
        """Return ``engine``'s staging lane; the call's first lane gates the
        scatter stream behind the caller's stream."""
        lane = self._lanes.get(engine)
        if lane is None:
            if not self._lanes:
                platform_stream.record_completion_event(
                    self._cache_context.device, self._caller_stream
                ).wait_on(self.scatter_stream)
            lane = staging_lane(self._cache_context, engine)
            self._lanes[engine] = lane
        return lane

    def join(self) -> None:
        """Make the caller's stream wait for every stream the call used."""
        if not self._lanes:
            return
        streams = [lane.stream for lane in self._lanes.values()]
        streams.append(self.scatter_stream)
        for stream in streams:
            platform_stream.record_completion_event(
                self._cache_context.device, stream
            ).wait_on(self._caller_stream)


@contextmanager
def open_retrieve_streams(
    cache_context: BaseCacheContext, direction: "lmcache_native.TransferDirection"
) -> Iterator[RetrieveStreams | None]:
    """Open the streams of one transfer call and join them on exit, also
    when the call raises: a queued load must finish before the caller
    releases the objects it reads.

    Args:
        cache_context: Cache context of the KV cache.
        direction: H2D (retrieve) or D2H (store).

    Yields:
        The call's streams, or None for a store or a context without a
        scatter stream.
    """
    scatter_stream = (
        cache_context.get_scatter_stream()
        if direction == lmcache_native.TransferDirection.H2D
        else None
    )
    if scatter_stream is None:
        yield None
        return
    streams = RetrieveStreams(cache_context, scatter_stream)
    try:
        yield streams
    finally:
        streams.join()
