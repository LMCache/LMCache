# SPDX-License-Identifier: Apache-2.0
"""
StandaloneServiceFactory: Service factory for LMCache standalone mode.

Creates LMCache service components without vLLM dependencies.
"""

# Standard
from typing import TYPE_CHECKING, Any, Callable, Optional

# First Party
from lmcache.integration.base_service_factory import BaseServiceFactory
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.health_monitor.constants import (
    DEFAULT_PING_INTERVAL,
    PING_INTERVAL_CONFIG_KEY,
)
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager

logger = init_logger(__name__)


class StandaloneServiceFactory(BaseServiceFactory):
    """Service factory for standalone LMCache mode (no vLLM)."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Any,
        broadcast_fn: Callable,
        broadcast_object_fn: Callable,
    ):
        self._config = config
        self._metadata = metadata
        self._gpu_connector = gpu_connector
        self._broadcast_fn = broadcast_fn
        self._broadcast_object_fn = broadcast_object_fn
        self._engine: Optional[LMCacheEngine] = None

    def get_or_create_metadata(self) -> Optional[LMCacheMetadata]:
        return self._metadata

    def get_or_create_lmcache_engine(self) -> Optional[LMCacheEngine]:
        if self._engine is not None:
            return self._engine

        instance_id = self._config.lmcache_instance_id
        self._engine = LMCacheEngineBuilder.get_or_create(
            instance_id=instance_id,
            config=self._config,
            metadata=self._metadata,
            gpu_connector=self._gpu_connector,
            broadcast_fn=self._broadcast_fn,
            broadcast_object_fn=self._broadcast_object_fn,
        )
        return self._engine

    def maybe_create_lookup_client(self):
        return None

    def maybe_create_prometheus_logger(self):
        return None

    def maybe_create_lookup_server(self):
        return None

    def maybe_create_offload_server(self):
        return None

    def maybe_create_runtime_plugin_launcher(self):
        return None

    def maybe_create_internal_api_server(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional[InternalAPIServer]:
        return InternalAPIServer(lmcache_manager)

    def maybe_create_health_monitor(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional[HealthMonitor]:
        # First Party
        from lmcache.observability import PrometheusLogger
        from lmcache.v1.periodic_thread import (
            PeriodicThreadRegistry,
            ThreadLevel,
        )

        ping_interval = self._config.get_extra_config_value(
            PING_INTERVAL_CONFIG_KEY, DEFAULT_PING_INTERVAL
        )
        health_monitor = HealthMonitor(
            manager=lmcache_manager,
            ping_interval=ping_interval,
        )

        if self._engine is not None:
            self._engine.set_health_monitor(health_monitor)

        health_monitor.start()
        logger.info("Health monitor initialized and started (standalone mode)")

        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.lmcache_is_healthy.set_function(
                lambda: 1 if lmcache_manager.is_healthy() else 0
            )

            registry = PeriodicThreadRegistry.get_instance()
            prometheus_logger.periodic_threads_total_count.set_function(
                lambda: len(registry.get_all())
            )
            prometheus_logger.periodic_threads_running_count.set_function(
                lambda: registry.get_running_count()
            )
            prometheus_logger.periodic_threads_active_count.set_function(
                lambda: registry.get_active_count()
            )

            for level in ThreadLevel:
                level_name = level.value
                total_attr = f"periodic_threads_{level_name}_total"
                running_attr = f"periodic_threads_{level_name}_running"
                active_attr = f"periodic_threads_{level_name}_active"

                if hasattr(prometheus_logger, total_attr):
                    getattr(prometheus_logger, total_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["total"]
                    )
                if hasattr(prometheus_logger, running_attr):
                    getattr(prometheus_logger, running_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["running"]
                    )
                if hasattr(prometheus_logger, active_attr):
                    getattr(prometheus_logger, active_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["active"]
                    )

        return health_monitor
