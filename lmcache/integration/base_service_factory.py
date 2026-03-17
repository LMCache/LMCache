# SPDX-License-Identifier: Apache-2.0
"""
BaseServiceFactory: Abstract interface for creating LMCache service components.

Each serving engine integration (e.g., vLLM) should implement a concrete
ServiceFactory that determines which components to create for each role.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    # First Party
    from lmcache.observability import PrometheusLogger
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.health_monitor.base import HealthMonitor
    from lmcache.v1.internal_api_server.api_server import InternalAPIServer
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer
    from lmcache.v1.manager import LMCacheManager
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
    from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher


class BaseServiceFactory(ABC):
    """Abstract base for creating LMCache service components.

    Subclasses must implement all methods to provide the appropriate
    components for their serving engine integration.
    """

    @abstractmethod
    def get_engine_instance_id(self) -> str:
        """Return the instance_id used to register the engine with
        LMCacheEngineBuilder. Used by LMCacheManager for engine destruction."""
        raise NotImplementedError

    @abstractmethod
    def get_or_create_metadata(self) -> Optional["LMCacheMetadata"]:
        raise NotImplementedError

    @abstractmethod
    def get_or_create_lmcache_engine(self) -> Optional["LMCacheEngine"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_lookup_client(self) -> Optional["LookupClientInterface"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_prometheus_logger(self) -> Optional["PrometheusLogger"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_lookup_server(
        self,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_offload_server(self) -> Optional["ZMQOffloadServer"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_runtime_plugin_launcher(
        self,
    ) -> Optional["RuntimePluginLauncher"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_internal_api_server(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional["InternalAPIServer"]:
        raise NotImplementedError

    @abstractmethod
    def maybe_create_health_monitor(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional["HealthMonitor"]:
        raise NotImplementedError
