# SPDX-License-Identifier: Apache-2.0
"""The extra streams of one pipelined retrieve call, tied to the caller's
stream at both ends.

The caller's stream is the one the request is ordered on: the engine's
event is waited on it before the transfer, and the completion event and
the lock release are queued on it afterwards. A pipelined retrieve queues
its loads and block kernels on other streams, so it has to keep both
orderings itself::

    caller stream   | wait engine | gate ............................ | join | done
    load stream     |    load b0  | load b1 | load b2 |                  ^
    scatter stream  |       (gate)| scatter b0 | scatter b1 | scatter b2 |

- gate: the scatter stream waits for an event recorded on the caller's
  stream, so no kernel runs before the block IDs are staged and the engine
  has released the blocks. Loads only touch staging banks and start at once.
- join: the caller's stream waits for every stream the call used, so what
  the caller queues next runs after the last kernel and the last load.
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
        self._scatter_stream = scatter_stream
        self._lanes: dict[TransferEngine, StagingLane] = {}

    @property
    def scatter_stream(self) -> Any:
        """The stream the block kernels are queued on."""
        return self._scatter_stream

    def lane(self, engine: TransferEngine) -> StagingLane:
        """Return ``engine``'s staging lane; the first call of a retrieve
        gates the scatter stream behind the caller's stream.

        Args:
            engine: The engine about to stage.

        Returns:
            The engine's lane on this cache context.
        """
        lane = self._lanes.get(engine)
        if lane is None:
            if not self._lanes:
                platform_stream.record_completion_event(
                    self._cache_context.device, self._caller_stream
                ).wait_on(self._scatter_stream)
            lane = staging_lane(self._cache_context, engine)
            self._lanes[engine] = lane
        return lane

    def join(self) -> None:
        """Make the caller's stream wait for every stream the call used."""
        if not self._lanes:
            return
        streams = [lane.stream for lane in self._lanes.values()]
        streams.append(self._scatter_stream)
        for stream in streams:
            platform_stream.record_completion_event(
                self._cache_context.device, stream
            ).wait_on(self._caller_stream)


@contextmanager
def open_retrieve_streams(
    cache_context: BaseCacheContext, scatter_stream: Any
) -> Iterator[RetrieveStreams]:
    """Open the streams of one pipelined retrieve call and join them on
    exit, also when the call raises: a load that was queued must finish
    before the caller releases the objects it reads.

    Args:
        cache_context: Cache context of the KV cache.
        scatter_stream: The context's scatter stream.

    Yields:
        The call's streams.
    """
    streams = RetrieveStreams(cache_context, scatter_stream)
    try:
        yield streams
    finally:
        streams.join()
