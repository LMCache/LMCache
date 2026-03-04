# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.config import PrometheusConfig
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)

logger = init_logger(__name__)


class PrometheusController:
    def __init__(self, config: PrometheusConfig):
        self._config = config
        self._lock = threading.Lock()
        self.all_loggers: List[PrometheusLogger] = []

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def register_logger(self, prom_logger: PrometheusLogger) -> None:
        """Register a logger for periodic flushing.

        Thread-safe: may be called after start().

        Args:
            prom_logger: The PrometheusLogger to register.
        """
        with self._lock:
            self.all_loggers.append(prom_logger)

    def start(self) -> None:
        """Start the periodic flush thread.

        No-op when ``config.enabled`` is False.
        """
        if not self._config.enabled:
            return

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="PrometheusController",
        )
        logger.info(
            "Starting PrometheusController (interval=%.1fs)...",
            self._config.log_interval,
        )
        with self._lock:
            all_logger_names = [type(pl).__name__ for pl in self.all_loggers]
        logger.info("Registered PrometheusLogger: %s.", all_logger_names)
        self._thread.start()

    def stop(self) -> None:
        """Stop the flush thread. Safe to call when not started."""
        self._stop_flag.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
            for pl in self.all_loggers:
                pl.unregister()

    def _run(self) -> None:
        while not self._stop_flag.wait(timeout=self._config.log_interval):
            with self._lock:
                snapshot = list(self.all_loggers)
            for pl in snapshot:
                try:
                    pl.log_prometheus()
                except Exception:
                    logger.exception(
                        "PrometheusController: error logging %s",
                        type(pl).__name__,
                    )


_global_controller = PrometheusController(PrometheusConfig(enabled=False))


def get_prometheus_controller() -> PrometheusController:
    """Return the current global PrometheusController singleton."""
    return _global_controller


def init_prometheus_controller(config: PrometheusConfig) -> PrometheusController:
    """Replace the global singleton with a new controller built from *config*."""
    global _global_controller
    _global_controller = PrometheusController(config)
    return _global_controller
