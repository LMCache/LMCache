# SPDX-License-Identifier: Apache-2.0

"""EventBus subscriber that turns L1 evictions into Dynamo ``BlockRemoved``.

Subscribes to :attr:`EventType.L1_KEYS_EVICTED` and forwards the evicted
chunks' hashes to a :class:`DynamoKvPublisher`. Like the store hook, this is a
bypass: a failure here must never disturb the EventBus drain loop, so the
callback swallows all exceptions.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_observability.dynamo.publisher import DynamoKvPublisher

logger = init_logger(__name__)


class DynamoEvictSubscriber(EventSubscriber):
    """Forwards ``L1_KEYS_EVICTED`` events to a Dynamo KV publisher.

    On each eviction the evicted keys' ``chunk_hash`` values are handed to the
    publisher, which looks up and emits the corresponding ``BlockRemoved``.
    """

    def __init__(self, publisher: DynamoKvPublisher) -> None:
        """Create the subscriber.

        Args:
            publisher: The Dynamo KV publisher to forward evictions to.
        """
        self._publisher = publisher

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Subscribe only to L1 key evictions."""
        return {EventType.L1_KEYS_EVICTED: self._on_evict}

    def _on_evict(self, event: Event) -> None:
        """Emit a ``BlockRemoved`` for the evicted chunks, never raising.

        Args:
            event: An ``L1_KEYS_EVICTED`` event whose ``metadata["keys"]``
                holds the evicted object keys (each with a ``chunk_hash``).
        """
        try:
            chunk_hashes = [key.chunk_hash for key in event.metadata["keys"]]
            self._publisher.on_evict(chunk_hashes)
        except Exception:
            logger.warning(
                "Dynamo KV event publish (evict) failed; ignoring", exc_info=True
            )

    def shutdown(self) -> None:
        """Close the underlying publisher (and its ZMQ socket)."""
        self._publisher.close()
