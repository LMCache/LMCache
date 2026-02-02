# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)
from lmcache.v1.lookup_client.hit_limit_lookup_client import HitLimitLookupClient
from lmcache.v1.lookup_client.lmcache_lookup_client_bypass import (
    LMCacheBypassLookupClient,
)
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)


class LookupClientFactory:
    """Factory for creating lookup clients and servers based on configuration."""

    @staticmethod
    def create_lookup_client(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        lmcache_engine: Optional[LMCacheEngine] = None,
    ) -> LookupClientInterface:
        """
        Create a lookup client based on the configuration.

        Args:
            config: The LMCache engine configuration
            metadata: The LMCache engine metadata (includes engine_id,
                world_size, kv_connector_extra_config)
            lmcache_engine: Optional LMCacheEngine instance for
                bypass lookup client

        Returns:
            A lookup client instance
        """

        client: LookupClientInterface
        # Check if external_lookup_client is configured
        if config.external_lookup_client is not None:
            if config.enable_async_loading:
                raise ValueError(
                    "Asynchronous loading is not supported for external lookup clients."
                )
            client = LookupClientFactory._create_external_lookup_client(
                config.external_lookup_client, config, metadata
            )
        else:
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupClient,
            )
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupClient,
            )

            # Check if bypass lookup is enabled and lmcache_engine is provided
            if config.enable_scheduler_bypass_lookup and lmcache_engine is not None:
                client = LMCacheBypassLookupClient(config, metadata, lmcache_engine)
            elif config.enable_async_loading:
                client = LMCacheAsyncLookupClient(config, metadata)
            else:
                client = LMCacheLookupClient(config, metadata)

        if config.hit_miss_ratio is not None and 0 <= config.hit_miss_ratio <= 1:
            client = HitLimitLookupClient(client, config)

        # Wrap with ChunkStatisticsLookupClient if enabled
        if config.enable_chunk_statistics:
            client = ChunkStatisticsLookupClient(
                client,
                config,
            )
        return client

    @staticmethod
    def create_lookup_server(
        lmcache_engine: LMCacheEngine,
        metadata: LMCacheMetadata,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        """
        Create a lookup server based on the configuration.

        Args:
            lmcache_engine: The LMCache engine instance
            metadata: The LMCache engine metadata (includes engine_id,
                world_size, kv_connector_extra_config, worker_id)

        Returns:
            A lookup server instance, or None if no server should be created
        """
        config = lmcache_engine.config
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 config is expected for lookup server and client"
        )

        lookup_server_worker_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )

        if config.external_lookup_client is None and (
            len(lookup_server_worker_ids) == 0
            or metadata.worker_id in lookup_server_worker_ids
        ):
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupServer,
            )
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupServer,
            )

            if config.enable_async_loading:
                return LMCacheAsyncLookupServer(lmcache_engine, metadata)
            else:
                return LMCacheLookupServer(lmcache_engine, metadata)

        return None

    @staticmethod
    def _create_external_lookup_client(
        external_lookup_uri: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> LookupClientInterface:
        """
        Create an external lookup client based on the URI format.

        Args:
            external_lookup_uri: URI in format <scheme>://<address>
            config: The LMCache engine configuration
            metadata: The LMCache engine metadata

        Returns:
            A lookup client instance

        Raises:
            ValueError: If the URI format is unsupported
        """
        # Parse URI scheme and address
        if "://" not in external_lookup_uri:
            raise ValueError(
                f"Invalid external lookup client URI format: {external_lookup_uri}. "
                "Expected format: <scheme>://<address>"
            )

        scheme, address = external_lookup_uri.split("://", 1)

        # Route to appropriate client based on scheme
        if scheme == "mooncakestore":
            return LookupClientFactory._create_mooncake_lookup_client(
                address, config, metadata
            )
        else:
            raise ValueError(
                f"Unsupported external lookup client scheme: {scheme}. "
                "Supported schemes: mooncakestore"
            )

    @staticmethod
    def _create_mooncake_lookup_client(
        master_address: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> "MooncakeLookupClient":
        """Create a MooncakeLookupClient instance."""
        # First Party
        from lmcache.v1.lookup_client.mooncake_lookup_client import (
            MooncakeLookupClient,
        )

        return MooncakeLookupClient(config, metadata, master_address)
