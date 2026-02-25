# SPDX-License-Identifier: Apache-2.0

# Standard
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.observability.storage_stats_listener import (
    StorageStatsListener,
)
from lmcache.v1.distributed.storage_controller import (
    StorageControllerInterface,
)

logger = init_logger(__name__)


class PrometheusController(StorageControllerInterface):
    def __init__(self, l1_manager: L1Manager, eviction_config: EvictionConfig):
        super().__init__(l1_manager)

        self.stats_listener: StorageStatsListener = StorageStatsListener()
        self.get_l1_manager().register_listener(self.stats_listener)

        self._stop_flag = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

    def start(self):
        logger.info("Starting ProemtheusController...")
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    def _run(self):

        while not self._stop_flag.is_set():
            time.sleep(1)  # Trigger every second
