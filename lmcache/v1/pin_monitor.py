# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryObj


logger = init_logger(__name__)


class PinMonitor:
    """
    Global monitor (singleton per process, shared across all cache engines)
    for pinned TensorMemoryObj instances to handle timeout detection.
    This class runs a background thread that periodically checks for pinned objects
    that have exceeded their timeout duration.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, config: "LMCacheEngineConfig"):
        # obj_id is the virtual memory address given by Python's id() function
        self._pinned_objects: dict[
            int, tuple["MemoryObj", float]
        ] = {}  # {obj_id: (memory_obj, register_time)}
        self._objects_lock = threading.Lock()
        self._monitor_thread = None
        self._running = False
        self._check_interval = config.pin_check_interval_sec
        self._pin_timeout_sec = config.pin_timeout_sec

        # Auto-start the monitor on first instance creation
        self.start_monitoring()

    @staticmethod
    def GetOrCreate(config: Optional["LMCacheEngineConfig"] = None) -> "PinMonitor":
        """Get or create the singleton instance.

        Args:
            config: Required for first-time initialization.
                Optional for subsequent calls.

        Raises:
            ValueError: If config is None when creating the instance
                for the first time.
        """
        if PinMonitor._instance is None:
            with PinMonitor._lock:
                if PinMonitor._instance is None:
                    assert config is not None, "config is required"
                    PinMonitor._instance = PinMonitor(config)
        return PinMonitor._instance

    def on_pin(self, memory_obj: "MemoryObj"):
        """Register a pinned memory object for timeout monitoring.

        Note: The same memory_obj can be pinned multiple times, so this
        function may be called multiple times with the same object.
        Each call updates the register time, effectively resetting the
        timeout countdown.
        """
        obj_id = id(memory_obj)
        with self._objects_lock:
            current_time = time.time()
            self._pinned_objects[obj_id] = (memory_obj, current_time)
            logger.debug(
                "Registered pinned object %s for timeout monitoring at time %.2f",
                obj_id,
                current_time,
            )

    def on_unpin(self, memory_obj: "MemoryObj"):
        """Unregister a memory object from timeout monitoring."""
        obj_id = id(memory_obj)
        with self._objects_lock:
            if obj_id in self._pinned_objects:
                del self._pinned_objects[obj_id]
                logger.debug(
                    "Unregistered pinned object %s from timeout monitoring",
                    obj_id,
                )

    def _check_timeouts(self):
        """Check all registered pinned objects for timeout."""
        current_time = time.time()
        timeout_objects = []

        with self._objects_lock:
            pinned_count = len(self._pinned_objects)
            for obj_id, (memory_obj, register_time) in list(
                self._pinned_objects.items()
            ):
                # Check if object is still pinned and has exceeded timeout
                if memory_obj.meta.pin_count > 0:
                    elapsed_time = current_time - register_time
                    if elapsed_time > self._pin_timeout_sec:
                        timeout_objects.append((memory_obj, elapsed_time))

        # Force unpin timeout objects outside the lock to avoid deadlocks
        force_unpin_success_count = 0
        for memory_obj, elapsed_time in timeout_objects:
            try:
                self._force_unpin_timeout_object(memory_obj, elapsed_time)
                force_unpin_success_count += 1
            except Exception as e:
                logger.error(
                    "Error forcing unpin for timeout object %s: %s", id(memory_obj), e
                )

        logger.info(
            "PinMonitor check: pinned_objects=%d, timeout_objects=%d, "
            "force_unpin_success=%d",
            pinned_count,
            len(timeout_objects),
            force_unpin_success_count,
        )

    def _force_unpin_timeout_object(self, memory_obj: "MemoryObj", elapsed_time: float):
        """Force unpin a timeout object and log the event."""
        # Get current pin_count without holding the lock for unpin calls
        # Use nullcontext if memory_obj doesn't have a lock attribute
        obj_lock = getattr(memory_obj, "lock", None) or nullcontext()
        with obj_lock:
            current_pin_count = memory_obj.meta.pin_count
            if current_pin_count <= 0:
                return

            logger.warning(
                "Pin timeout detected for MemoryObj %s. "
                "Pin count: %s, Elapsed time: %.2fs. Forcing unpin to 0.",
                memory_obj.meta.address,
                current_pin_count,
                elapsed_time,
            )

        # Update forced unpin statistics
        LMCStatsMonitor.GetOrCreate().update_forced_unpin_count(1)

        # Call unpin() while pin_count > 0 to properly release resources
        while memory_obj.meta.pin_count > 0:
            memory_obj.unpin()

    def _monitor_loop(self):
        """Background thread loop for monitoring pinned objects."""
        logger.info("Starting PinMonitor background thread")
        while self._running:
            try:
                self._check_timeouts()
                time.sleep(self._check_interval)
            except Exception as e:
                logger.error("Error in PinMonitor loop: %s", e)
                time.sleep(self._check_interval)  # Continue after error
        logger.info("PinMonitor background thread stopped")

    def start_monitoring(self):
        """Start the background monitoring thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="PinMonitor-thread"
        )
        self._monitor_thread.start()
        logger.info("PinMonitor started")

        # Setup metrics callback
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.pin_monitor_pinned_objects_count.set_function(
                lambda: len(self._pinned_objects)
            )

    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        if not self._running:
            return

        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("PinMonitor stopped")

    def get_monitored_count(self) -> int:
        """Get the number of currently monitored pinned objects."""
        with self._objects_lock:
            return len(self._pinned_objects)

    @staticmethod
    def DestroyInstance():
        """Destroy the singleton instance and stop monitoring.
        This is mainly used for testing to ensure clean state between tests.
        """
        with PinMonitor._lock:
            if PinMonitor._instance is not None:
                PinMonitor._instance.stop_monitoring()
                PinMonitor._instance = None
