# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, List
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controller import (
    StorageControllerInterface,
)
from lmcache.v1.mp_observability.logger.l1_stats_logger import (
    L1ManagerStatsLogger,
)
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)
from lmcache.v1.mp_observability.logger.storage_manager_stats_logger import (
    StorageManagerStatsLogger,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)


class PrometheusController(StorageControllerInterface):
    def __init__(
        self,
        storage_manager: "StorageManager",
        l1_manager: L1Manager,
        log_interval: float,
    ):
        super().__init__(storage_manager, l1_manager)

        self._log_interval = log_interval
        self.all_loggers: List[PrometheusLogger] = []

        self.sm_stats_logger: StorageManagerStatsLogger = StorageManagerStatsLogger()
        self.get_storage_manager().register_listener(self.sm_stats_logger)
        self.all_loggers.append(self.sm_stats_logger)

        self.l1_stats_logger: L1ManagerStatsLogger = L1ManagerStatsLogger()
        self.get_l1_manager().register_listener(self.l1_stats_logger)
        self.all_loggers.append(self.l1_stats_logger)

        # TODO: adding more stats loggers, e.g., integrator logger or mp server logger

        self._stop_flag = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="PrometheusController",
        )

    def start(self):
        logger.info(
            "Starting PrometheusController (interval=%.1fs)...", self._log_interval
        )
        all_logger_names = [
            type(prometheus_logger).__name__ for prometheus_logger in self.all_loggers
        ]
        logger.info(f"Registered PrometheusLogger: {all_logger_names}.")
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()
        for prometheus_logger in self.all_loggers:
            prometheus_logger.unregister()

    def _run(self):
        while not self._stop_flag.wait(timeout=self._log_interval):
            for prometheus_logger in self.all_loggers:
                try:
                    prometheus_logger.log_prometheus()
                except Exception:
                    logger.exception(
                        "PrometheusController: error logging %s",
                        type(prometheus_logger).__name__,
                    )
