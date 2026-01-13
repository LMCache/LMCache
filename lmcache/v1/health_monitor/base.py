# SPDX-License-Identifier: Apache-2.0
"""
Base classes for health monitoring.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.health_monitor.constants import DEFAULT_PING_INTERVAL

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager

logger = init_logger(__name__)


class HealthCheck(ABC):
    """
    Abstract base class for health checks.

    Subclasses should implement the check() method to perform specific
    health checks. Each health check represents one aspect of system health.

    Subclasses must also implement the create() classmethod
    to create instances from a LMCacheManager.

    Example:
        class DatabaseHealthCheck(HealthCheck):
            def __init__(self, db_connection):
                self.db = db_connection

            def name(self) -> str:
                return "DatabaseHealthCheck"

            def check(self) -> bool:
                return self.db.ping()

            @classmethod
            def create(
                cls, manager: "LMCacheManager"
            ) -> List["HealthCheck"]:
                # Create instances from manager's components
                return [cls(manager.lmcache_engine.db_connection)]
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of this health check"""
        pass

    @abstractmethod
    def check(self) -> bool:
        """
        Perform the health check.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    def should_skip(self) -> bool:
        """
        Check if this health check should be skipped.

        Override this method to conditionally skip checks
        (e.g., if the component doesn't support health checks).

        Returns:
            bool: True if the check should be skipped, False otherwise
        """
        return False

    @classmethod
    @abstractmethod
    def create(cls, manager: "LMCacheManager") -> List["HealthCheck"]:
        """
        Create health check instance(s) from a LMCacheManager.

        This method should extract the necessary components from the manager
        and create one or more health check instances.

        Args:
            manager: The LMCacheManager instance

        Returns:
            List[HealthCheck]: List of health check instances.
                              Return empty list if the check is not applicable.
        """
        pass


class HealthMonitor:
    """
    Health monitor for the entire LMCache system.

    This is the unified health monitor for the entire LMCache system.
    It supports extensible health checks and provides a centralized way
    to check the health status of the LMCacheManager.

    The monitor automatically discovers and instantiates all HealthCheck
    subclasses using their create() method.

    The monitor runs in a background thread and periodically executes
    all registered health checks. If any check fails, the system is
    marked as unhealthy.

    Usage:
        # Create a health monitor with manager
        health_monitor = HealthMonitor(
            manager=manager,
            ping_interval=30.0
        )

        # Start monitoring
        health_monitor.start()

        # Check health status
        if health_monitor.is_healthy():
            # Perform normal operations
            pass

        # Stop monitoring when done
        health_monitor.stop()
    """

    def __init__(
        self,
        manager: "LMCacheManager",
        ping_interval: float = DEFAULT_PING_INTERVAL,
    ):
        self._manager = manager
        self._health_checks: List[HealthCheck] = []

        # Health status
        self._healthy = True
        self._health_lock = threading.RLock()

        # Monitor thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Configuration
        self._ping_interval = ping_interval

        # Auto-discover and instantiate health checks
        self._discover_health_checks()

    def _discover_health_checks(self) -> None:
        """
        Discover all HealthCheck subclasses and instantiate them.

        This method dynamically scans all modules in the checks package,
        finds all HealthCheck subclasses and calls their `create()`
        method to create instances.
        """
        # Standard
        import importlib
        import inspect
        import pkgutil

        # First Party
        # Import the checks package
        import lmcache.v1.health_monitor.checks as checks_pkg

        # Discover all modules in the checks package
        for _, module_name, _ in pkgutil.iter_modules(checks_pkg.__path__):
            # Skip private modules
            if module_name.startswith("_"):
                continue

            try:
                module = importlib.import_module(f"{checks_pkg.__name__}.{module_name}")

                # Find all HealthCheck subclasses in the module
                for _, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, HealthCheck)
                        and obj != HealthCheck
                    ):
                        try:
                            instances = obj.create(self._manager)
                            for instance in instances:
                                self._health_checks.append(instance)
                                logger.info(
                                    f"Registered health check: {instance.name()}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to create health check {obj.__name__}: {e}"
                            )
            except ImportError as e:
                logger.warning(f"Failed to import module {module_name}: {e}")

    def get_health_checks(self) -> List[HealthCheck]:
        """Get all registered health checks"""
        return list(self._health_checks)

    def is_healthy(self) -> bool:
        """
        Check if the system is currently healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        with self._health_lock:
            return self._healthy

    def _set_healthy(self, healthy: bool) -> None:
        """Set the health status"""
        with self._health_lock:
            if self._healthy != healthy:
                if healthy:
                    logger.info(
                        "HealthMonitor: System recovered, restoring normal operations"
                    )
                else:
                    logger.warning(
                        "HealthMonitor: System unhealthy, entering degraded mode"
                    )
                self._healthy = healthy

    def _run_all_checks(self) -> bool:
        """
        Run all health checks.

        Returns:
            bool: True if all checks pass, False if any check fails

        Raises:
            IrrecoverableException: If any check raises an irrecoverable error
        """
        for check in self._health_checks:
            if check.should_skip():
                continue
            try:
                if not check.check():
                    logger.warning(f"Health check failed: {check.name()}")
                    return False
            except IrrecoverableException:
                logger.error(
                    f"Health check {check.name()} raised IrrecoverableException"
                )
                raise
            except Exception as e:
                logger.error(f"Health check {check.name()} raised exception: {e}")
                return False
        return True

    def start(self) -> Optional[threading.Thread]:
        """
        Start the health monitor thread.

        Returns:
            Optional[threading.Thread]: The started thread,
                or None if no checks need monitoring
        """
        # Check if we have any health checks that need to run
        active_checks = [c for c in self._health_checks if not c.should_skip()]
        if not active_checks:
            logger.info("No active health checks to monitor, skipping monitor thread")
            return None

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="health-monitor-thread",
        )
        self._thread.start()
        logger.info(
            f"Started health monitor thread with {len(active_checks)} active checks, "
            f"interval: {self._ping_interval}s"
        )
        return self._thread

    def stop(self) -> None:
        """Stop the health monitor thread"""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Health monitor thread did not terminate within timeout")

    def _run_loop(self) -> None:
        """Main monitoring loop"""
        logger.info(
            f"Starting health monitor loop with interval {self._ping_interval}s"
        )

        while not self._stop_event.is_set():
            # Sleep with interruptible wait
            if self._stop_event.wait(timeout=self._ping_interval):
                break

            try:
                # Run all health checks
                is_healthy = self._run_all_checks()
                self._set_healthy(is_healthy)
            except IrrecoverableException as e:
                logger.error(f"Irrecoverable error in health monitor, stopping: {e}")
                self._set_healthy(False)
                break

        logger.info("Health monitor loop stopped")
