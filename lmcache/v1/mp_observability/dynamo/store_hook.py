# SPDX-License-Identifier: Apache-2.0

"""Bypass hook that turns a committed store into a Dynamo ``BlockStored``.

This is the single call site the MP server's store paths use to publish KV
events. It is deliberately defensive: emitting events must never disturb the
real store. Every failure -- and the disabled case (``publisher is None``) --
is swallowed here so the caller's success path is unchanged.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.mp_observability.dynamo.publisher import DynamoKvPublisher
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

logger = init_logger(__name__)


def publish_store_from_key(
    publisher: DynamoKvPublisher | None,
    key: IPCCacheServerKey,
    object_keys: list[ObjectKey],
) -> None:
    """Publish a ``BlockStored`` for a just-committed store, never raising.

    No-op when ``publisher`` is ``None`` (feature disabled). Any exception is
    logged at warning level and suppressed so the store path is unaffected.

    Args:
        publisher: The Dynamo KV publisher, or ``None`` when the feature is
            disabled.
        key: The committed store's cache key. Supplies the full token prefix
            (``token_ids``) and the newly stored range (``start``, ``end``).
        object_keys: The resolved object keys for the stored range, in token
            order. Their ``chunk_hash`` values are used as eviction keys.
    """
    if publisher is None:
        return
    try:
        token_ids = list(key.token_ids)
        chunk_hashes = [ok.chunk_hash for ok in object_keys]
        publisher.on_store(token_ids, key.start, key.end, chunk_hashes)
    except Exception:
        logger.warning(
            "Dynamo KV event publish (store) failed; ignoring", exc_info=True
        )
