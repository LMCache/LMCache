# SPDX-License-Identifier: Apache-2.0

# Standard
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controller import (
    StorageControllerInterface,
)

logger = init_logger(__name__)


class EvictionController(StorageControllerInterface):
    def __init__(
        self,
        l1_manager: L1Manager,
        eviction_config: EvictionConfig,
    ):
        super().__init__(l1_manager)

        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(self._eviction_config)
        self.get_l1_manager().register_listener(self._eviction_policy)

        self._stop_flag = threading.Event()

        self._thread = threading.Thread(
            target=self._eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting EvictionController...")
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    def _eviction_loop(self):
        l1_manager = self.get_l1_manager()
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)  # Trigger every second
            used_bytes, total_bytes = l1_manager.get_memory_usage()
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                logger.debug(
                    "Memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage,
                    watermark,
                )
                continue

            # Trigger eviction
            logger.info(
                "Memory usage %.2f above watermark %.2f; triggering eviction.",
                usage,
                watermark,
            )
            actions = self._eviction_policy.get_eviction_actions(eviction_ratio)
            for action in actions:
                self._execute_eviction_action(action)

    def _execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.DISCARD:
            self.get_l1_manager().delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self.get_l1_manager().delete(action.keys)
