
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from typing import Any, Callable
import os
import threading
import logging

logger = logging.getLogger(__name__)

class OperationTimeoutError(Exception):
    """Exception raised when operations timeout."""
    pass

class OperationManager:
    """Manages execution of operations with timeouts and tracks failures."""
    def __init__(
        self,
        num_threads: int = 4,
    ):
        self.timeout_pool = ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="fs-timeout"
        )
        self._failure_count = 0
        self._failure_lock = threading.Lock()

    def run_with_timeout(
        self,
        func: Callable[[], Any],
        timeout_seconds: float,
        label: str = "default_label",
        metadata: Any = None,
    ) -> Any:
        future = self.timeout_pool.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except ConcurrentTimeoutError as err:
            count = self.increment_failure_count()
            raise OperationTimeoutError(
                f"Operation '{label}' timed out after {timeout_seconds} seconds",
                metadata,
                count,
            ) from err

    def shutdown(self):
        self.timeout_pool.shutdown(wait=True)

    def increment_failure_count(self) -> int:
        with self._failure_lock:
            self._failure_count += 1
            return self._failure_count

    def get_failure_count(self) -> int:
        """Get the current count of timed-out operations."""
        with self._failure_lock:
            return self._failure_count

    def reset_failure_count(self) -> int:
        """Reset the timeout counter and return the previous count."""
        with self._failure_lock:
            old_count = self._failure_count
            self._failure_count = 0
            return old_count
