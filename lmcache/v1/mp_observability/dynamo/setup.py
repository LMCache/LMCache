# SPDX-License-Identifier: Apache-2.0

"""Factory that wires up Dynamo KV-event publishing for the MP server.

Kept separate from ``server.py`` so the wiring can be unit-tested with a fake
context and a real EventBus, without standing up a server. The heavy imports
(the ZMQ/msgspec/vLLM-backed emitter) are deferred into the function body so
importing this module stays cheap.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_observability.config import ObservabilityConfig
    from lmcache.v1.mp_observability.dynamo.publisher import DynamoKvPublisher
    from lmcache.v1.mp_observability.event_bus import EventBus
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext

logger = init_logger(__name__)


def setup_dynamo_kv_publishing(
    ctx: MPCacheServerContext,
    obs_config: ObservabilityConfig,
    bus: EventBus,
) -> DynamoKvPublisher | None:
    """Build and install Dynamo KV-event publishing, if enabled.

    When :attr:`ObservabilityConfig.enable_dynamo_kv_events` is off, this is a
    no-op returning ``None``. When on, it builds the emitter and publisher
    (reusing the context's real token hasher and chunk size), registers the
    store and evict subscribers on ``bus``, attaches the publisher to ``ctx`` so
    the store paths can reach it, and returns the publisher.

    Args:
        ctx: Server context. Supplies ``token_hasher`` and ``chunk_size``, and
            receives the built publisher via its ``dynamo_kv_publisher`` slot.
        obs_config: Observability configuration carrying the Dynamo settings.
        bus: The EventBus to register the store and evict subscribers on. They
            may be registered after ``bus.start()`` -- the drain loop reads its
            subscriber map fresh each cycle.

    Returns:
        The installed :class:`DynamoKvPublisher`, or ``None`` when disabled.
    """
    if not obs_config.enable_dynamo_kv_events:
        return None

    # First Party
    from lmcache.v1.mp_observability.dynamo.emitter import DynamoKvEventEmitter
    from lmcache.v1.mp_observability.dynamo.evict_subscriber import (
        DynamoEvictSubscriber,
    )
    from lmcache.v1.mp_observability.dynamo.publisher import DynamoKvPublisher
    from lmcache.v1.mp_observability.dynamo.store_subscriber import (
        DynamoStoreSubscriber,
    )

    emitter = DynamoKvEventEmitter(
        zmq_bind=obs_config.dynamo_zmq_bind,
        medium=obs_config.dynamo_medium,
        data_parallel_rank=obs_config.dynamo_dp_rank,
    )
    publisher = DynamoKvPublisher(
        emitter,
        ctx.token_hasher,
        obs_config.dynamo_kv_block_size,
        ctx.chunk_size,
    )
    bus.register_subscriber(DynamoStoreSubscriber(publisher))
    bus.register_subscriber(DynamoEvictSubscriber(publisher))
    ctx.dynamo_kv_publisher = publisher
    logger.info(
        "Dynamo KV event publishing enabled (bind=%s, block_size=%d)",
        obs_config.dynamo_zmq_bind,
        obs_config.dynamo_kv_block_size,
    )
    return publisher
