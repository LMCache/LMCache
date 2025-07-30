# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector.blackhole_connector import BlackholeConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

logger = init_logger(__name__)

# Ping error codes
PING_TIMEOUT_ERROR_CODE = -1
PING_GENERIC_ERROR_CODE = -2

# Configuration constants
PING_TIMEOUT_CONFIG_KEY = "ping_timeout"
PING_INTERVAL_CONFIG_KEY = "ping_interval"
DEFAULT_PING_TIMEOUT = 5.0
DEFAULT_PING_INTERVAL = 30.0


class RemoteMonitor:
    """
    Remote monitor class, encapsulating the monitor logic
    """

    def __init__(self, backend: "RemoteBackend"):
        self.backend = backend

        # Lock for connector switching
        self.connector_lock = threading.RLock()

        # Store the original connector
        self.original_connector = backend.connection

        # Create a blackhole connector for fallback
        self.blackhole_connector = InstrumentedRemoteConnector(BlackholeConnector())

    def _should_skip_ping(self) -> bool:
        """
        Check if we should skip ping for this connector
        """
        if self.original_connector is None:
            logger.warning("Original connector is None, should retry.")
            return False

        if not self.original_connector.support_ping():
            logger.info(
                f"Connector {self.original_connector} "
                f"does not support ping, skipping ping loop"
            )
            return True

        return False

    def start(self):
        """
        Start the monitor thread
        """
        # Check if we should skip starting the thread
        if self._should_skip_ping():
            return None

        thread = threading.Thread(
            target=self.run_loop,
            daemon=True,
            name=f"{self.original_connector}-monitor-thread",
        )
        thread.start()
        return thread

    def _safe_switch_connector(self, new_connector):
        """Thread-safe connector switching"""
        with self.connector_lock:
            if self.backend.connection != new_connector:
                self.backend.connection = new_connector

    def run_loop(self):
        """
        Run the monitor loop
        """
        # Get configuration from extra_config
        extra_config = (
            self.backend.config.extra_config
            if self.backend.config.extra_config is not None
            else {}
        )
        ping_timeout = extra_config.get(PING_TIMEOUT_CONFIG_KEY, DEFAULT_PING_TIMEOUT)
        ping_interval = extra_config.get(
            PING_INTERVAL_CONFIG_KEY, DEFAULT_PING_INTERVAL
        )
        logger.info(
            f"Starting remote monitor thread {threading.current_thread().name} "
            f"with interval {ping_interval}s and timeout {ping_timeout}s"
        )

        connection_healthy = True
        while True:
            time.sleep(ping_interval)
            # Check if original_connector is still uninitialized
            if self.original_connector is None:
                # Double-checked locking for initialization
                with self.connector_lock:
                    if self.original_connector is None:
                        logger.warning(
                            "original_connector is None, re-initializing connection."
                        )
                        self.backend.init_connection()
                        self.original_connector = self.backend.connection
                        if self.original_connector is None:
                            continue
                        if not self.original_connector.support_ping():
                            logger.info(
                                f"Connector {self.original_connector} "
                                "does not support ping, break RemoteMonitor thread."
                            )
                            break

            try:
                start_time = time.perf_counter()
                future = asyncio.run_coroutine_threadsafe(
                    self.original_connector.ping(), self.backend.loop
                )
                error_code = future.result(timeout=ping_timeout)
                latency = (time.perf_counter() - start_time) * 1000
                # Record ping latency
                self.backend.stats_monitor.update_remote_ping_latency(latency)
                connection_healthy = error_code == 0
                # Record error code (0 means success)
                self.backend.stats_monitor.update_remote_ping_error_code(error_code)
                if error_code != 0:
                    logger.warning(f"Ping failed with error code: {error_code}")
            except asyncio.TimeoutError:
                connection_healthy = False
                logger.warning("Ping timeout")
                # Set timeout error code (-1)
                self.backend.stats_monitor.update_remote_ping_error_code(
                    PING_TIMEOUT_ERROR_CODE
                )
            except Exception as e:
                connection_healthy = False
                logger.error(f"Ping error: {e}")
                # Set generic exception error code (-2)
                self.backend.stats_monitor.update_remote_ping_error_code(
                    PING_GENERIC_ERROR_CODE
                )

            # Update connector based on health status
            if connection_healthy:
                self._safe_switch_connector(self.original_connector)
            else:
                self._safe_switch_connector(self.blackhole_connector)
