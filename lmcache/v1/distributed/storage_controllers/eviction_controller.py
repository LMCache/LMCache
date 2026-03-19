# SPDX-License-Identifier: Apache-2.0

# Standard
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
    L1ManagerListener,
    L2AdapterListener,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controller import StorageControllerInterface

logger = init_logger(__name__)


class EvictionController(StorageControllerInterface):
    """
    Abstract base class for eviction controllers.

    Provides the shared eviction loop structure: background thread, stop flag,
    and eviction policy. Subclasses implement _eviction_loop and
    _execute_eviction_action for their specific tier (L1 or L2).
    """

    def __init__(self, eviction_config: EvictionConfig):
        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(eviction_config)
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    def report_status(self) -> dict:
        raise NotImplementedError

    def _eviction_loop(self):
        raise NotImplementedError

    def _execute_eviction_action(self, action: EvictionAction):
        raise NotImplementedError


class L1EvictionController(EvictionController, L1ManagerListener):
    """
    Eviction controller for L1 cache.

    Registers itself as an L1ManagerListener to keep the eviction policy
    up-to-date, and periodically triggers eviction based on L1 memory usage.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        eviction_config: EvictionConfig,
    ):
        super().__init__(eviction_config)
        self._l1_manager = l1_manager
        self._l1_manager.register_listener(self)

    def report_status(self) -> dict:
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
        }

    # ------------------------------------------------------------------
    # L1ManagerListener — delegate to the eviction policy
    # ------------------------------------------------------------------

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]):
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]):
        self._eviction_policy.on_keys_touched(keys)

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]):
        pass

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]):
        self._eviction_policy.on_keys_created(keys)

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]):
        self._eviction_policy.on_keys_removed(keys)

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]):
        self._eviction_policy.on_keys_created(keys)

    # ------------------------------------------------------------------
    # Eviction loop
    # ------------------------------------------------------------------

    def _eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                logger.debug(
                    "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage,
                    watermark,
                )
                continue

            logger.info(
                "L1 memory usage %.2f above watermark %.2f; triggering eviction.",
                usage,
                watermark,
            )
            actions = self._eviction_policy.get_eviction_actions(eviction_ratio)
            for action in actions:
                self._execute_eviction_action(action)

    def _execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.DISCARD:
            self._l1_manager.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self._l1_manager.delete(action.keys)


class L2EvictionController(EvictionController, L2AdapterListener):
    """
    Eviction controller for L2 storage.

    Acts as a L2AdapterListener to keep the eviction policy up-to-date,
    and periodically triggers eviction based on L2 storage usage reported
    by the adapter.

    Does NOT require or hold an L1Manager.
    """

    def __init__(
        self,
        l2_adapter: L2AdapterInterface,
        eviction_config: EvictionConfig,
    ):
        super().__init__(eviction_config)
        self._l2_adapter = l2_adapter
        self._l2_adapter.register_listener(self)

    def report_status(self) -> dict:
        current_usage, usage_after_eviction = self._l2_adapter.get_usage()
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
            "current_usage": current_usage,
            "usage_after_ongoing_eviction": usage_after_eviction,
        }

    # ------------------------------------------------------------------
    # L2AdapterListener — delegate to the eviction policy
    # ------------------------------------------------------------------

    def on_l2_keys_stored(self, keys: list[ObjectKey]):
        self._eviction_policy.on_keys_created(keys)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        self._eviction_policy.on_keys_touched(keys)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        self._eviction_policy.on_keys_removed(keys)

    # ------------------------------------------------------------------
    # Eviction loop
    # ------------------------------------------------------------------

    def _eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            current_usage, _ = self._l2_adapter.get_usage()
            if current_usage < watermark:
                logger.debug(
                    "L2 usage %.2f below watermark %.2f; skipping eviction.",
                    current_usage,
                    watermark,
                )
                continue

            logger.info(
                "L2 usage %.2f above watermark %.2f; triggering eviction.",
                current_usage,
                watermark,
            )
            actions = self._eviction_policy.get_eviction_actions(eviction_ratio)
            for action in actions:
                self._execute_eviction_action(action)

    def _execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.DISCARD:
            self._l2_adapter.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self._l2_adapter.delete(action.keys)
