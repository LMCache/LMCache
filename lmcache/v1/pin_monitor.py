# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import TensorMemoryObj


logger = init_logger(__name__)


class PinMonitor:
    """
    Global monitor for pinned TensorMemoryObj instances to handle timeout detection.
    This class runs a background thread that periodically checks for pinned objects
    that have exceeded their timeout duration.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, config: "LMCacheEngineConfig"):
        self._pinned_objects: dict[
            int, tuple["TensorMemoryObj", float]
        ] = {}  # {obj_id: (memory_obj, register_time)}
        self._objects_lock = threading.RLock()
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

    def register_pinned_object(self, memory_obj: "TensorMemoryObj"):
        """Register a pinned memory object for timeout monitoring.

        Note: The same memory_obj can be pinned multiple times, so this
        function may be called multiple times with the same object.
        Each call updates the register time, effectively resetting the
        timeout countdown.
        """
        with self._objects_lock:
            obj_id = id(memory_obj)
            current_time = time.time()
            self._pinned_objects[obj_id] = (memory_obj, current_time)
            logger.debug(
                "Registered pinned object %s for timeout monitoring at time %.2f",
                obj_id,
                current_time,
            )

    def unregister_pinned_object(self, memory_obj: "TensorMemoryObj"):
        """Unregister a memory object from timeout monitoring."""
        with self._objects_lock:
            obj_id = id(memory_obj)
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
            for obj_id, (memory_obj, register_time) in list(
                self._pinned_objects.items()
            ):
                # Check if object is still pinned and has exceeded timeout
                if memory_obj.meta.pin_count > 0:
                    elapsed_time = current_time - register_time
                    if elapsed_time > self._pin_timeout_sec:
                        timeout_objects.append((memory_obj, elapsed_time))
                else:
                    self.unregister_pinned_object(memory_obj)

        # Force unpin timeout objects outside the lock to avoid deadlocks
        for memory_obj, elapsed_time in timeout_objects:
            try:
                self._force_unpin_timeout_object(memory_obj, elapsed_time)
            except Exception as e:
                logger.error(
                    "Error forcing unpin for timeout object %s: %s", id(memory_obj), e
                )

    def _force_unpin_timeout_object(
        self, memory_obj: "TensorMemoryObj", elapsed_time: float
    ):
        """Force unpin a timeout object and log the event."""
        # Get current pin_count without holding the lock for unpin calls
        with memory_obj.lock:
            current_pin_count = memory_obj.meta.pin_count
            if current_pin_count <= 0:
                self.unregister_pinned_object(memory_obj)
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
