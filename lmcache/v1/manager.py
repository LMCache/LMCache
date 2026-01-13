# SPDX-License-Identifier: Apache-2.0
"""
LMCacheManager: A unified manager for LMCache internal components.

This module provides a clean interface to manage LMCache components lifecycle,
decoupling adapters from internal LMCache implementation details.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import TYPE_CHECKING, Any, Optional, Union
import time

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)

# Engine name constant
ENGINE_NAME = "LMCacheEngine"


class LMCacheManager:
    """
    LMCacheManager manages the lifecycle of LMCache internal components.

    This class encapsulates the initialization and shutdown of:
    - LMCacheEngine
    - LookupClient / LookupServer
    - OffloadServer
    - InternalAPIServer
    - RuntimePluginLauncher

    The manager is serving-engine-agnostic and relies on metadata
    provided by the integration adapter.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        role: str = "worker",
        connector: Optional[Any] = None,
    ):
        """
        Initialize LMCacheManager.

        Args:
            config: LMCache engine configuration
            metadata: Engine metadata extracted from serving engine
            role: The role string ("scheduler" or "worker")
            connector: Reference to adapter for internal API server
        """
        self._config = config
        self._metadata = metadata
        self._role = role
        self._connector: Any = connector

        # Components (initialized later)
        self._lmcache_engine: Optional[LMCacheEngine] = None
        self._lmcache_engine_metadata: Optional[LMCacheEngineMetadata] = None
        self._lookup_client: Optional[LookupClientInterface] = None
        self._lookup_server: Optional[
            Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]
        ] = None
        self._offload_server: Optional[ZMQOffloadServer] = None
        self._api_server: Optional[InternalAPIServer] = None
        self._runtime_plugin_launcher: Optional[RuntimePluginLauncher] = None

        # Initialize components based on role
        self._init_components()

    def _init_components(self) -> None:
        """Initialize components based on the role."""
        if self._role == "scheduler":
            self._init_scheduler_components()
        else:
            self._init_worker_components()
        # Initialize API server and plugin launcher only on DP rank 0
        if self._metadata.data_parallel_rank_local == 0:
            self._init_dp_rank0_components()

    def _init_scheduler_components(self) -> None:
        """Initialize components for scheduler role."""
        # First Party
        from lmcache.observability import PrometheusLogger
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        if self._config.enable_scheduler_bypass_lookup:
            # Create LMCacheEngine for scheduler when bypass is enabled
            self._lmcache_engine = self._create_lmcache_engine(self._metadata)
            self._lmcache_engine_metadata = self._lmcache_engine.metadata
        else:
            self._lmcache_engine = None
            # Use provided metadata for prometheus logger
            self._lmcache_engine_metadata = self._metadata
            PrometheusLogger.GetOrCreate(self._lmcache_engine_metadata)

        # Create lookup client
        self._lookup_client = LookupClientFactory.create_lookup_client(
            self._config,
            self._lmcache_engine_metadata,
            self._lmcache_engine,
        )

    def _init_worker_components(self) -> None:
        """Initialize components for worker role."""
        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        # Create LMCacheEngine
        self._lmcache_engine = self._create_lmcache_engine(self._metadata)
        self._lmcache_engine_metadata = self._lmcache_engine.metadata

        # Create lookup server
        self._lookup_server = LookupClientFactory.create_lookup_server(
            self._lmcache_engine, self._lmcache_engine_metadata
        )

        # Create offload server
        self._offload_server = ZMQOffloadServer(
            self._lmcache_engine,
            self._metadata.worker_id,
        )

    def _init_dp_rank0_components(self) -> None:
        """Initialize components that only run on DP rank 0."""
        # Start internal API server
        self._api_server = InternalAPIServer(self)

        # Create plugin launcher
        worker_id = (
            -1
            if self._lmcache_engine is None
            else self._lmcache_engine.metadata.worker_id
        )
        self._runtime_plugin_launcher = RuntimePluginLauncher(
            self._config,
            self._role,
            self._metadata.tensor_parallel_size,
            worker_id,
        )

    def _create_lmcache_engine(self, metadata: LMCacheEngineMetadata) -> LMCacheEngine:
        """
        Create and return an LMCacheEngine instance.

        Args:
            metadata: Engine metadata

        Returns:
            LMCacheEngine instance
        """
        if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
            return curr_engine

        self._validate_mla_config(metadata.use_mla)

        # Create GPU connector
        gpu_connector = self._create_gpu_connector(metadata)

        engine = LMCacheEngineBuilder.get_or_create(
            ENGINE_NAME,
            self._config,
            metadata,
            gpu_connector,
            metadata.broadcast_fn,
            metadata.broadcast_object_fn,
        )

        if metadata.role == "scheduler" and self._config.enable_scheduler_bypass_lookup:
            assert engine.save_only_first_rank or self._config.get_extra_config_value(
                "remote_enable_mla_worker_id_as0", metadata.use_mla
            ), (
                "enable_scheduler_bypass_lookup is only supported with "
                "save_only_first_rank or remote_enable_mla_worker_id_as0"
            )

        return engine

    def _validate_mla_config(self, use_mla: bool) -> None:
        """Validate MLA-related configuration."""
        if use_mla and (
            self._config.remote_serde != "naive"
            and self._config.remote_serde is not None
        ):
            raise ValueError("MLA only works with naive serde mode..")

        if use_mla and self._config.use_layerwise and self._config.enable_blending:
            raise ValueError(
                "We haven't supported MLA with Cacheblend yet. Please disable blending."
            )

    def _create_gpu_connector(self, metadata: LMCacheEngineMetadata):
        """Create the GPU connector based on configuration and metadata."""
        # First Party
        from lmcache.v1.gpu_connector import (
            VLLMBufferLayerwiseGPUConnector,
            VLLMPagedMemGPUConnectorV2,
            VLLMPagedMemGPUConnectorV3,
            VLLMPagedMemLayerwiseGPUConnector,
        )
        from lmcache.v1.xpu_connector import VLLMPagedMemXPUConnectorV2

        use_gpu = self._need_gpu_interm_buffer()

        if metadata.role == "scheduler":
            return None

        if self._config.use_layerwise:
            if self._config.enable_blending:
                return VLLMBufferLayerwiseGPUConnector.from_metadata(metadata, use_gpu)
            else:
                return VLLMPagedMemLayerwiseGPUConnector.from_metadata(
                    metadata, use_gpu
                )

        if metadata.is_cuda_alike():
            if self._config.use_gpu_connector_v3:
                return VLLMPagedMemGPUConnectorV3.from_metadata(metadata, use_gpu)
            else:
                return VLLMPagedMemGPUConnectorV2.from_metadata(metadata, use_gpu)
        elif metadata.is_xpu():
            return VLLMPagedMemXPUConnectorV2.from_metadata(metadata, use_gpu)
        else:
            raise RuntimeError(
                f"No supported connector found for platform: {metadata.device_name}"
            )

    def _need_gpu_interm_buffer(self) -> bool:
        """Check if GPU intermediate buffer is needed."""
        return not self._config.enable_pd

    def start_services(self) -> None:
        """
        Start all managed services.

        Managed services include:
        - InternalAPIServer: HTTP server exposing internal APIs for
          monitoring and management (e.g., cache stats, flush operations).
        - RuntimePluginLauncher: Launches external plugin processes defined
          in the configuration (e.g., custom telemetry, cache warming scripts).
        """
        if self._api_server is not None:
            self._api_server.start()

        if self._runtime_plugin_launcher is not None:
            self._runtime_plugin_launcher.launch_plugins()

    def post_init(self) -> None:
        """
        Post-initialization after KV caches are registered.
        """
        if self._lmcache_engine is None:
            return

        # Post-init for engines with async loading
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupServer,
        )

        async_lookup_server = None
        if self._config.enable_async_loading and self._lookup_server is not None:
            assert isinstance(self._lookup_server, LMCacheAsyncLookupServer)
            async_lookup_server = self._lookup_server

        self._lmcache_engine.post_init(async_lookup_server=async_lookup_server)

    def stop_services(self) -> None:
        """Stop all managed components gracefully."""
        logger.info("Stopping LMCacheManager services...")
        start_time = time.time()
        errors: list[tuple[str, Union[str, Exception]]] = []

        def _safe_close(name: str, close_fn, timeout: float = 10.0):
            """Helper to close a resource with timeout protection."""
            try:
                logger.info("Closing %s...", name)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(close_fn)
                    try:
                        future.result(timeout=timeout)
                        logger.info("%s closed successfully", name)
                    except TimeoutError:
                        logger.error(
                            "%s close operation timed out after %ss. "
                            "Continuing with shutdown...",
                            name,
                            timeout,
                        )
                        errors.append((name, "Timeout"))
            except Exception as e:
                logger.error("Error closing %s: %s", name, e)
                errors.append((name, e))

        # Close offload server
        if self._offload_server is not None:
            _safe_close("offload_server", self._offload_server.close, timeout=10.0)

        # Stop plugins
        if self._runtime_plugin_launcher is not None:
            _safe_close(
                "runtime_plugin_launcher",
                self._runtime_plugin_launcher.stop_plugins,
                timeout=10.0,
            )

        # Stop API server
        if self._api_server is not None:
            _safe_close("api_server", self._api_server.stop, timeout=10.0)

        # Close lookup server
        if self._lookup_server is not None:
            _safe_close("lookup_server", self._lookup_server.close, timeout=10.0)

        # Close lookup client
        if self._lookup_client is not None:
            _safe_close("lookup_client", self._lookup_client.close, timeout=10.0)

        # Destroy cache engine
        try:
            logger.info("Destroying LMCache engine: %s", ENGINE_NAME)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(LMCacheEngineBuilder.destroy, ENGINE_NAME)
                try:
                    future.result(timeout=15.0)
                    logger.info("LMCache engine destroyed successfully")
                except TimeoutError:
                    logger.error(
                        "Cache engine destroy timed out after 15s. "
                        "Continuing with shutdown..."
                    )
                    errors.append(("cache_engine", "Timeout"))
        except Exception as e:
            logger.error("Error destroying cache engine: %s", e)
            errors.append(("cache_engine", e))

        elapsed = time.time() - start_time
        if errors:
            logger.warning(
                "Shutdown completed with %d errors in %.2fs: %s",
                len(errors),
                elapsed,
                errors,
            )
        else:
            logger.info(
                "LMCacheManager services stopped successfully in %.2fs", elapsed
            )

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance."""
        return self._lmcache_engine

    @property
    def lmcache_engine_metadata(self) -> Optional[LMCacheEngineMetadata]:
        """Get the LMCache engine metadata."""
        return self._lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional[LookupClientInterface]:
        """Get the lookup client instance."""
        return self._lookup_client

    @property
    def lookup_server(
        self,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        """Get the lookup server instance."""
        return self._lookup_server

    @property
    def offload_server(self) -> Optional[ZMQOffloadServer]:
        """Get the offload server instance."""
        return self._offload_server

    @property
    def api_server(self) -> Optional[InternalAPIServer]:
        """Get the API server instance."""
        return self._api_server

    @property
    def config(self) -> LMCacheEngineConfig:
        """Get the LMCache engine configuration."""
        return self._config

    def get_inference_info(self) -> dict:
        """Get inference information by delegating to the connector.

        Returns:
            dict: Dictionary containing inference information,
                  or empty dict if connector is not available.
        """
        if self._connector is not None and hasattr(
            self._connector, "get_inference_info"
        ):
            return self._connector.get_inference_info()
        return {}
