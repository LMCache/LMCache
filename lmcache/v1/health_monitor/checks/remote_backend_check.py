# SPDX-License-Identifier: Apache-2.0
"""
Health check for RemoteBackend.
"""

# Standard
from typing import TYPE_CHECKING, List
import asyncio
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.health_monitor.base import HealthCheck
from lmcache.v1.health_monitor.constants import (
    DEFAULT_PING_TIMEOUT,
    PING_GENERIC_ERROR_CODE,
    PING_TIMEOUT_CONFIG_KEY,
    PING_TIMEOUT_ERROR_CODE,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager
    from lmcache.v1.storage_backend.remote_backend import RemoteBackend

logger = init_logger(__name__)


class RemoteBackendHealthCheck(HealthCheck):
    """
    Health check for RemoteBackend by pinging the remote connector.

    This check verifies that the remote backend is reachable and responsive
    by sending periodic ping requests.
    """

    def __init__(
        self,
        backend: "RemoteBackend",
    ):
        self.backend = backend
        # Get ping timeout from backend config
        self.ping_timeout = backend.config.get_extra_config_value(
            PING_TIMEOUT_CONFIG_KEY, DEFAULT_PING_TIMEOUT
        )
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()

    @classmethod
    def create(cls, manager: "LMCacheManager") -> List[HealthCheck]:
        """
        Create RemoteBackendHealthCheck instances from a LMCacheManager.

        This method finds all RemoteBackend instances in the storage manager
        and creates a health check for each one.

        Args:
            manager: The LMCacheManager instance

        Returns:
            List of RemoteBackendHealthCheck instances
        """
        # Import here to avoid circular imports
        # First Party
        from lmcache.v1.storage_backend.remote_backend import RemoteBackend

        instances: List[HealthCheck] = []

        # Get engine from manager
        engine = manager.lmcache_engine
        if engine is None or engine.storage_manager is None:
            return instances

        for backend_name, backend in engine.storage_manager.storage_backends.items():
            if isinstance(backend, RemoteBackend):
                check = cls(backend)
                instances.append(check)
                logger.info(f"Created RemoteBackendHealthCheck for {backend_name}")

        return instances

    def name(self) -> str:
        return f"RemoteBackendHealthCheck({self.backend.remote_url})"

    def should_skip(self) -> bool:
        """Check if we should skip ping for this connector"""
        connector = self.backend.connection
        if connector is None:
            logger.warning("Connector is None, should retry.")
            return False

        if not connector.support_ping():
            logger.info(
                f"Connector {connector} does not support ping, skipping ping loop"
            )
            return True

        return False

    def _try_reinitialize_connection(self) -> bool:
        """
        Try to reinitialize the connection if connector is None.

        Returns:
            bool: True if connection was successfully initialized, False otherwise
        """
        if self.backend.connection is not None:
            return True

        logger.warning("Connector is None, re-initializing connection.")
        self.backend.init_connection()

        return self.backend.connection is not None

    def check(self) -> bool:
        """
        Perform a ping check on the remote backend.

        Returns:
            bool: True if ping succeeds, False otherwise
        """
        # Try to reinitialize connection if needed
        if not self._try_reinitialize_connection():
            return False

        # At this point, connector is guaranteed to be not None
        connector = self.backend.connection
        assert connector is not None

        # If connector doesn't support ping, assume it's healthy
        if not connector.support_ping():
            return True

        try:
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(
                connector.ping(), self.backend.loop
            )
            error_code = future.result(timeout=self.ping_timeout)
            latency = (time.perf_counter() - start_time) * 1000

            # Record ping latency
            self._stats_monitor.update_remote_ping_latency(latency)
            # Record error code (0 means success)
            self._stats_monitor.update_remote_ping_error_code(error_code)

            if error_code != 0:
                logger.warning(f"Ping failed with error code: {error_code}")
                return False

            return True

        except asyncio.TimeoutError:
            logger.warning("Ping timeout")
            self._stats_monitor.update_remote_ping_error_code(PING_TIMEOUT_ERROR_CODE)
            return False
        except Exception as e:
            logger.error(f"Ping error: {e}")
            self._stats_monitor.update_remote_ping_error_code(PING_GENERIC_ERROR_CODE)
            return False
