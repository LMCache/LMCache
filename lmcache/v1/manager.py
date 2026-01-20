# SPDX-License-Identifier: Apache-2.0
"""
LMCacheManager: A unified manager for LMCache internal components.

This module provides a clean interface to manage LMCache components lifecycle,
decoupling northbound integrations from internal LMCache implementation details.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import TYPE_CHECKING, Any, Optional, Union
import time

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.health_monitor.constants import (
    DEFAULT_PING_INTERVAL,
    PING_INTERVAL_CONFIG_KEY,
)
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher

if TYPE_CHECKING:
    # Third Party

    # First Party
    from lmcache.config import LMCacheEngineMetadata
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)

# Engine name constant
ENGINE_NAME = "LMCacheEngine"


class LMCacheManager:
    """
    LMCacheManager bundles LMCache internal components.

    The main service is the LMCacheEngine.

    Auxiliary services are:
    - LookupClient / LookupServer (specific to vLLM for
        disaggregated scheduler-worker processes)
    - OffloadServer
    - InternalAPIServer
    - RuntimePluginLauncher
    - HealthMonitor

    The northbound consumer of LMCacheManager is responsible for
    constructing LMCacheEngineMetadata which should toggle on/off
    all of the services inside of the LMCacheManager
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        northbound: Optional[Any] = None,
    ):
        """
        Args:
            config: LMCache configuration
            metadata: LMCache metadata (extracted from northbound use case)
            northbound: Reference to northbound connector/adapter
            (e.g. LMCacheConnectorV1Impl for vLLM)
        """
        self._config = config
        self._metadata = metadata
        self._northbound = northbound

        self._lmcache_engine: Optional[LMCacheEngine] = None
        self._lookup_client: Optional[LookupClientInterface] = None
        self._lookup_server = None
        self._offload_server: Optional[ZMQOffloadServer] = None
        self._api_server: Optional[InternalAPIServer] = None
        self._runtime_plugin_launcher: Optional[RuntimePluginLauncher] = None
        self._health_monitor: Optional[HealthMonitor] = None

        self._init_components()

    def _init_components(self):
        if self._metadata.needs_cache_engine(self._config):
            self._lmcache_engine = self._create_lmcache_engine()
        else:
            # the cache engine has a StatLogger with a prometheus logger
            # components without the cache engine should still have a prometheus logger
            PrometheusLogger.GetOrCreate(self._metadata)
        if self._metadata.needs_lookup_client:
            self._lookup_client = self._create_lookup_client()
        if self._metadata.needs_lookup_server:
            self._lookup_server = self._create_lookup_server()
        if self._metadata.needs_offload_server:
            self._offload_server = self._create_offload_server()
        if self._metadata.needs_api_server:
            self._api_server = self._create_api_server()
        if self._metadata.needs_runtime_plugin_launcher:
            self._runtime_plugin_launcher = self._create_runtime_plugin_launcher()

    def _create_lmcache_engine(self) -> LMCacheEngine:
        """
        Create and return an LMCacheEngine instance using pre-built metadata.

        Returns:
            LMCacheEngine instance
        """
        engine_name = self._metadata.engine_name
        return LMCacheEngineBuilder.get_or_create(
            engine_name,
            self._config,
            self._metadata,
        )

    def _create_lookup_client(self) -> LookupClientInterface:
        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        return LookupClientFactory.create_lookup_client(
            self._config,
            self._metadata,
            self._lmcache_engine,
        )

    def _create_lookup_server(self):
        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        return LookupClientFactory.create_lookup_server(
            self._lmcache_engine, self._metadata
        )

    def _create_offload_server(self):
        return ZMQOffloadServer(
            self._lmcache_engine,
            self._metadata.tensor_model_parallel_rank,
        )

    def _create_api_server(self):
        return InternalAPIServer(self)

    def _create_runtime_plugin_launcher(self):
        worker_id = -1 if self._lmcache_engine is None else self._metadata.worker_id
        return RuntimePluginLauncher(
            self._config,
            self._metadata.role,
            self._metadata.tensor_parallel_size,
            worker_id,
        )

    def _init_health_monitor(self) -> None:
        """
        Initialize the health monitor for the LMCacheManager.

        This is called during post_init after all components are initialized.
        The HealthMonitor automatically discovers and instantiates all
        HealthCheck subclasses based on the manager's role and components.
        """
        # First Party
        from lmcache.observability import PrometheusLogger

        if not self._metadata.needs_health_monitor:
            return

        # Get ping interval from config
        ping_interval = self._config.get_extra_config_value(
            PING_INTERVAL_CONFIG_KEY, DEFAULT_PING_INTERVAL
        )

        # Create health monitor with manager - it will auto-discover health checks
        self._health_monitor = HealthMonitor(
            manager=self,
            ping_interval=ping_interval,
        )

        # Inject health monitor into engine (if exists)
        if self._lmcache_engine is not None:
            self._lmcache_engine.set_health_monitor(self._health_monitor)

        # Start the health monitor
        self._health_monitor.start()
        logger.info(
            "Health monitor initialized and started at manager level (role=%s)",
            self._metadata.role,
        )

        # Setup metrics callback for health status
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.lmcache_is_healthy.set_function(
                lambda: 1 if self.is_healthy() else 0
            )

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
            # Initialize health monitor for scheduler (even without engine)
            self._init_health_monitor()
            return

        # vLLM mode post-init
        # First Party
        from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupServer,
        )

        async_lookup_server = None
        if self._config.enable_async_loading and self._lookup_server is not None:
            assert isinstance(self._lookup_server, LMCacheAsyncLookupServer)
            async_lookup_server = self._lookup_server

        self._lmcache_engine.post_init(async_lookup_server=async_lookup_server)

        # Initialize health monitor after engine post_init completes
        self._init_health_monitor()

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

        # Stop health monitor first
        if self._health_monitor is not None:
            _safe_close("health_monitor", self._health_monitor.stop, timeout=5.0)

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
            # In vLLM mode, use ENGINE_NAME constant
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
        return self._metadata

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
    def health_monitor(self) -> Optional[HealthMonitor]:
        """Get the health monitor instance."""
        return self._health_monitor

    @property
    def role(self) -> Optional[str]:
        """Get the role of this manager (scheduler or worker)."""
        return self._metadata.role

    def is_healthy(self) -> bool:
        """
        Check if the LMCacheManager is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._health_monitor is None:
            return True
        return self._health_monitor.is_healthy()

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
        if self._northbound is not None and hasattr(
            self._northbound, "get_inference_info"
        ):
            return self._northbound.get_inference_info()
        return {}
