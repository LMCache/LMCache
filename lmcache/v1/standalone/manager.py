# SPDX-License-Identifier: Apache-2.0
"""
StandaloneLMCacheManager: A specialized manager for LMCache standalone mode.

This class extends LMCacheManager to handle standalone mode specifically,
removing vLLM dependencies and simplifying the initialization logic.
"""

# Standard
from typing import TYPE_CHECKING, Any, Callable, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # Fir
    pass

logger = init_logger(__name__)


class StandaloneLMCacheManager(LMCacheManager):
    """
    LMCacheManager specialized for standalone mode.

    This class handles the standalone mode without vLLM dependencies,
    providing a cleaner and more focused implementation.
    """

    def __init__(
        self,
        config: Any,
        metadata: LMCacheMetadata,
        gpu_connector: Any,
        broadcast_fn: Callable,
        broadcast_object_fn: Callable,
        connector: Optional[Any] = None,
    ):
        """
        Initialize StandaloneLMCacheManager with standalone-specific parameters.

        Args:
            config: LMCache engine configuration
            metadata: Pre-constructed LMCacheMetadata
            gpu_connector: GPU connector instance
            broadcast_fn: Broadcast function for tensor parallel
            broadcast_object_fn: Broadcast function for objects
            connector: Reference to LMCacheConnectorV1Impl for internal API server
        """
        # Store standalone-specific parameters before parent __init__
        # (needed by _init_components which is called in parent __init__)
        self._metadata = metadata
        self._gpu_connector = gpu_connector
        self._broadcast_fn = broadcast_fn
        self._broadcast_object_fn = broadcast_object_fn

        # Call parent __init__ (vllm_config=None for standalone mode)
        super().__init__(
            config=config,
            vllm_config=None,
            role="worker",
            connector=connector,
        )

    def _init_components(self) -> None:
        """Initialize components specifically for standalone mode."""
        if self._role != "worker":
            raise NotImplementedError(
                "Standalone mode currently only supports 'worker' role, not 'scheduler'"
            )

        # Initialize LMCache engine for standalone mode
        instance_id = self._config.lmcache_instance_id

        self._lmcache_engine = LMCacheEngineBuilder.get_or_create(
            instance_id=instance_id,
            config=self._config,
            metadata=self._metadata,
            gpu_connector=self._gpu_connector,
            broadcast_fn=self._broadcast_fn,
            broadcast_object_fn=self._broadcast_object_fn,
        )
        self._lmcache_engine_metadata = self._lmcache_engine.metadata

        # Initialize API server
        self._api_server = InternalAPIServer(self)

    def post_init(self) -> None:
        """Post-initialization for standalone mode."""
        if self._lmcache_engine is not None:
            # Standalone mode post-init is simpler (no async_lookup_server)
            self._lmcache_engine.post_init()

        # Initialize health monitor after engine post_init completes (if engine exists)
        # This also sets up PeriodicThreadRegistry metrics
        self._init_health_monitor()

    def stop_services(self) -> None:
        """Shutdown for standalone mode with simplified logic."""
        logger.info("Starting StandaloneLMCacheManager shutdown...")

        # Let parent handle common shutdown logic
        super().stop_services()

        logger.info("StandaloneLMCacheManager shutdown completed")
