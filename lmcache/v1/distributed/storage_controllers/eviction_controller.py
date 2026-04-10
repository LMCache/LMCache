# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import abstractmethod
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L1EvictionPolicy, L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.eviction_policy.user_lru import (
    UserLRUEvictionPolicy,
)
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.storage_controller import StorageControllerInterface

logger = init_logger(__name__)


class EvictionController(StorageControllerInterface):
    """
    Abstract base class for eviction controllers.

    Provides the shared eviction loop structure: background thread and stop
    flag. Subclasses implement eviction_loop and execute_eviction_action
    for their specific tier (L1 or L2).
    """

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self.eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    @abstractmethod
    def report_status(self) -> dict:
        """Return a status dict for this controller.

        The child class needs to override this function to report
        controller-specific health and configuration information.
        """
        pass

    @abstractmethod
    def eviction_loop(self):
        """Run the eviction loop.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass

    @abstractmethod
    def execute_eviction_action(self, action: EvictionAction):
        """Execute a single eviction action.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass


class L1EvictionController(EvictionController):
    """
    Eviction controller for L1 cache.

    Uses an L1EvictionPolicy bridge to keep the eviction policy up-to-date
    with L1 manager events, and periodically triggers eviction based on
    L1 memory usage.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        eviction_config: EvictionConfig,
    ):
        super().__init__()
        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(eviction_config)
        self._l1_manager = l1_manager
        self._listener = L1EvictionPolicy(self._eviction_policy)
        self._l1_manager.register_listener(self._listener)

    def report_status(self) -> dict:
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
        }

    def eviction_loop(self):
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
                self.execute_eviction_action(action)

    def execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.DISCARD:
            self._l1_manager.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self._l1_manager.delete(action.keys)


class L2AdapterEvictionState:
    """Per-adapter eviction state: its own policy, listener, and config."""

    def __init__(
        self,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig,
    ):
        self.adapter = adapter
        self.eviction_config = eviction_config
        self.eviction_policy = CreateEvictionPolicy(eviction_config)
        self.listener = L2EvictionPolicy(self.eviction_policy)
        adapter.register_listener(self.listener)


class L2EvictionController(StorageControllerInterface):
    """
    Unified eviction controller for all L2 adapters.

    Each adapter gets its own eviction policy and listener bridge, but a
    single background thread loops over all of them.

    When a ``quota_manager`` is provided and the adapter uses
    ``UserLRUEvictionPolicy``, eviction is triggered per-user based on
    individual quota limits instead of global aggregate usage.
    """

    def __init__(
        self,
        l2_adapter_states: list[L2AdapterEvictionState],
        quota_manager: QuotaManager | None = None,
    ):
        """Initialize the L2 eviction controller.

        Args:
            l2_adapter_states: Per-adapter eviction state objects.
            quota_manager: Optional per-user quota registry.  When
                provided together with a UserLRU policy, enables
                per-user watermark-based eviction.
        """
        self._adapter_states = l2_adapter_states
        self._quota_manager = quota_manager
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
        adapter_statuses = []
        for state in self._adapter_states:
            current_usage, usage_after_eviction = state.adapter.get_usage()
            adapter_statuses.append(
                {
                    "eviction_policy": state.eviction_config.eviction_policy,
                    "trigger_watermark": state.eviction_config.trigger_watermark,
                    "eviction_ratio": state.eviction_config.eviction_ratio,
                    "current_usage": current_usage,
                    "usage_after_ongoing_eviction": usage_after_eviction,
                }
            )
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "adapters": adapter_statuses,
        }

    def _eviction_loop(self):
        while not self._stop_flag.is_set():
            time.sleep(1)
            for state in self._adapter_states:
                self._check_and_evict(state)

    def _check_and_evict(self, state: L2AdapterEvictionState) -> None:
        """Check usage and evict if above watermark.

        For ``UserLRUEvictionPolicy`` with a quota manager, eviction is
        triggered per-user based on individual quota limits.  For all
        other policies, eviction uses the global aggregate usage.

        Args:
            state: The per-adapter eviction state to check.
        """
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio
        policy = state.eviction_policy

        if isinstance(policy, UserLRUEvictionPolicy) and self._quota_manager:
            # UserLRU: per-user watermark check
            per_user_usage = state.adapter.get_per_user_usage()
            for user_id, (user_bytes, _) in per_user_usage.items():
                limit = self._quota_manager.get_limit_bytes(user_id)
                if user_bytes <= watermark * limit:
                    continue

                logger.info(
                    "User %s L2 usage %.2f GB exceeds watermark "
                    "(%.0f%%) of quota %.2f GB; evicting.",
                    user_id,
                    user_bytes / (1024**3),
                    watermark * 100,
                    limit / (1024**3),
                )
                actions = policy.get_eviction_actions(eviction_ratio, user_id=user_id)
                for action in actions:
                    self._execute_eviction_action(state.adapter, action)
        else:
            # LRU/NoOp: global aggregate watermark (unchanged)
            current_usage, _ = state.adapter.get_usage()
            if current_usage < 0 or current_usage < watermark:
                logger.debug(
                    "L2 usage %.2f below watermark %.2f; skipping eviction.",
                    current_usage,
                    watermark,
                )
                return

            logger.info(
                "L2 usage %.2f above watermark %.2f; triggering eviction.",
                current_usage,
                watermark,
            )
            actions = policy.get_eviction_actions(eviction_ratio)
            for action in actions:
                self._execute_eviction_action(state.adapter, action)

    def _execute_eviction_action(
        self, adapter: L2AdapterInterface, action: EvictionAction
    ):
        if action.destination == EvictionDestination.DISCARD:
            adapter.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            adapter.delete(action.keys)
