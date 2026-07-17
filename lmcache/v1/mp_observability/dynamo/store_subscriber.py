# SPDX-License-Identifier: Apache-2.0

"""EventBus subscriber that turns committed stores into Dynamo ``BlockStored``.

Subscribes to :attr:`EventType.MP_KEYS_STORED` and forwards the committed
store's full token prefix and chunk hashes to a :class:`DynamoKvPublisher`.
Like the evict subscriber, this is a bypass: a failure here must never disturb
the EventBus drain loop, so the callback swallows all exceptions.
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


class DynamoStoreSubscriber(EventSubscriber):
    """Forwards ``MP_KEYS_STORED`` events to a Dynamo KV publisher.

    On each committed store the key's full token prefix and the stored range's
    ``chunk_hash`` values are handed to the publisher, which recomputes and
    emits the corresponding ``BlockStored``.

    Closing the publisher is owned by the evict subscriber's ``shutdown``; this
    subscriber shares the same publisher and must not close it as well.
    """

    def __init__(self, publisher: DynamoKvPublisher) -> None:
        """Create the subscriber.

        Args:
            publisher: The Dynamo KV publisher to forward stores to.
        """
        self._publisher = publisher

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Subscribe only to committed-store events."""
        return {EventType.MP_KEYS_STORED: self._on_store}

    def _on_store(self, event: Event) -> None:
        """Emit a ``BlockStored`` for the committed range, never raising.

        Args:
            event: An ``MP_KEYS_STORED`` event whose ``metadata["key"]`` holds
                the committed cache key (supplying ``token_ids``, ``start``,
                and ``end``) and whose ``metadata["object_keys"]`` holds the
                resolved object keys for the stored range, in token order (each
                with a ``chunk_hash``).
        """
        try:
            key = event.metadata["key"]
            object_keys = event.metadata["object_keys"]
            token_ids = list(key.token_ids)
            chunk_hashes = [ok.chunk_hash for ok in object_keys]
            self._publisher.on_store(token_ids, key.start, key.end, chunk_hashes)
        except Exception:
            logger.warning(
                "Dynamo KV event publish (store) failed; ignoring", exc_info=True
            )
