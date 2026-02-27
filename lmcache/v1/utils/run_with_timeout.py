from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class OperationStats:
    """Statistics for a single operation type. Lock-free for real-time use."""
    total_count: int = 0
    success_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    
    def record_success(self, latency_ms: float) -> None:
        self.total_count += 1
        self.success_count += 1
        self.total_latency_ms += latency_ms
        if latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms
    
    def record_timeout(self, latency_ms: float) -> None:
        self.total_count += 1
        self.timeout_count += 1
        self.total_latency_ms += latency_ms
        if latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms
    
    def record_error(self, latency_ms: float) -> None:
        self.total_count += 1
        self.error_count += 1
        self.total_latency_ms += latency_ms
        if latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_latency_ms / self.total_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "success_count": self.success_count,
            "timeout_count": self.timeout_count,
            "error_count": self.error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "min_latency_ms": round(self.min_latency_ms, 3) if self.total_count > 0 else 0.0,
            "max_latency_ms": round(self.max_latency_ms, 3),
        }


@dataclass  
class OperationManagerStats:
    """Aggregated statistics for the OperationManager. Lock-free for real-time use."""
    # Current state (simple counters - GIL provides sufficient atomicity for stats)
    in_flight_count: int = 0
    peak_in_flight_count: int = 0
    queued_count: int = 0
    peak_queued_count: int = 0
    
    # Per-operation stats
    operation_stats: Dict[str, OperationStats] = field(default_factory=dict)
    
    def get_or_create_op_stats(self, label: str) -> OperationStats:
        if label not in self.operation_stats:
            self.operation_stats[label] = OperationStats()
        return self.operation_stats[label]
    
    def get_summary_line(self) -> str:
        """Get a concise summary line for logging."""
        op_summaries = []
        for label, stats in self.operation_stats.items():
            op_summaries.append(
                f"{label}={stats.timeout_count}/{stats.total_count}to"
            )
        ops_str = ", ".join(op_summaries) if op_summaries else "none"
        return (
            f"in_flight={self.in_flight_count} (peak={self.peak_in_flight_count}) | "
            f"queued={self.queued_count} (peak={self.peak_queued_count}) | "
            f"ops: [{ops_str}]"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_flight_count": self.in_flight_count,
            "peak_in_flight_count": self.peak_in_flight_count,
            "queued_count": self.queued_count,
            "peak_queued_count": self.peak_queued_count,
            "operations": {
                label: stats.to_dict() 
                for label, stats in self.operation_stats.items()
            },
        }
    
    def reset(self) -> Dict[str, Any]:
        """Reset stats and return the old values."""
        old_stats = self.to_dict()
        self.in_flight_count = 0
        self.peak_in_flight_count = 0
        self.queued_count = 0
        self.peak_queued_count = 0
        self.operation_stats.clear()
        return old_stats


class OperationTimeoutError(Exception):
    """Exception raised when operations timeout."""
    pass

class OperationManager:
    """Manages execution of operations with timeouts and tracks failures."""
    def __init__(
        self,
        num_threads: int = 4,
    ):
        self.num_threads = num_threads
        self.timeout_pool = ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="fs-timeout"
        )
        # Lock-free: only updated from caller thread
        self._failure_count = 0
        self._stats = OperationManagerStats()

    def run_with_timeout(
        self,
        func: Callable[[], Any],
        timeout_seconds: float,
        label: str = "default_label",
        metadata: Any = None,
    ) -> Any:
        start_time = time.perf_counter()
        
        # Track queued state (lock-free - GIL sufficient for stats)
        self._stats.queued_count += 1
        if self._stats.queued_count > self._stats.peak_queued_count:
            self._stats.peak_queued_count = self._stats.queued_count
        
        future = self.timeout_pool.submit(func)
        
        # Transition from queued to in-flight
        self._stats.queued_count -= 1
        self._stats.in_flight_count += 1
        if self._stats.in_flight_count > self._stats.peak_in_flight_count:
            self._stats.peak_in_flight_count = self._stats.in_flight_count
        
        try:
            result = future.result(timeout=timeout_seconds)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            self._stats.in_flight_count -= 1
            op_stats = self._stats.get_or_create_op_stats(label)
            op_stats.record_success(latency_ms)
            
            return result
        except ConcurrentTimeoutError as err:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._failure_count += 1
            
            self._stats.in_flight_count -= 1
            op_stats = self._stats.get_or_create_op_stats(label)
            op_stats.record_timeout(latency_ms)
            
            # Log stats on timeout to help diagnose bottlenecks
            logger.error(
                f"TIMEOUT: Operation '{label}' after {timeout_seconds}s | "
                f"latency={latency_ms:.1f}ms | "
                f"total_failures={self._failure_count} | "
                f"OpManager: {self._stats.get_summary_line()}"
            )
            
            raise OperationTimeoutError(
                f"Operation '{label}' timed out after {timeout_seconds} seconds",
                metadata,
                self._failure_count,
            ) from err
        except Exception as err:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            self._stats.in_flight_count -= 1
            op_stats = self._stats.get_or_create_op_stats(label)
            op_stats.record_error(latency_ms)
            
            raise

    def shutdown(self):
        self.timeout_pool.shutdown(wait=True)

    def get_failure_count(self) -> int:
        """Get the current count of timed-out operations."""
        return self._failure_count

    def reset_failure_count(self) -> int:
        """Reset the timeout counter and return the previous count."""
        old_count = self._failure_count
        self._failure_count = 0
        return old_count

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics as a dictionary."""
        stats = self._stats.to_dict()
        stats["num_threads"] = self.num_threads
        stats["failure_count"] = self._failure_count
        return stats
    
    def reset_stats(self) -> Dict[str, Any]:
        """Reset statistics and return the old values."""
        old_stats = self._stats.reset()
        old_stats["failure_count"] = self.reset_failure_count()
        return old_stats
