import pytest
import time
import threading
from unittest.mock import Mock
from lmcache.v1.utils.run_with_timeout import (
    OperationManager,
    OperationTimeoutError,
)


class TestOperationManager:
    """Test suite for OperationManager."""

    def test_successful_operation(self):
        """Test that a successful operation completes and returns the result."""
        manager = OperationManager(num_threads=2)
        
        def successful_func():
            return "success"
        
        result = manager.run_with_timeout(
            successful_func,
            timeout_seconds=1.0,
            label="test_op"
        )
        
        assert result == "success"
        assert manager.get_failure_count() == 0
        manager.shutdown()

    def test_operation_timeout(self):
        """Test that a slow operation times out and raises OperationTimeoutError."""
        manager = OperationManager(num_threads=2)
        
        def slow_func():
            time.sleep(2.0)
            return "should not reach here"
        
        with pytest.raises(OperationTimeoutError) as exc_info:
            manager.run_with_timeout(
                slow_func,
                timeout_seconds=0.5,
                label="slow_operation",
                metadata={"test": "data"}
            )
        
        assert "slow_operation" in str(exc_info.value)
        assert "timed out after 0.5 seconds" in str(exc_info.value)
        assert manager.get_failure_count() == 1
        manager.shutdown()

    def test_failure_count_increments(self):
        """Test that failure count increments with each timeout."""
        manager = OperationManager(num_threads=2)
        
        def timeout_func():
            time.sleep(1.0)
        
        for i in range(3):
            with pytest.raises(OperationTimeoutError):
                manager.run_with_timeout(
                    timeout_func,
                    timeout_seconds=0.1,
                    label=f"op_{i}"
                )
            assert manager.get_failure_count() == i + 1
        
        manager.shutdown()

    def test_failure_count_does_not_increment_on_success(self):
        """Test that successful operations do not increment failure count."""
        manager = OperationManager(num_threads=2)
        
        def timeout_func():
            time.sleep(1.0)
        
        def success_func():
            return "ok"
        
        # First timeout
        with pytest.raises(OperationTimeoutError):
            manager.run_with_timeout(timeout_func, timeout_seconds=0.1)
        
        assert manager.get_failure_count() == 1
        
        # Successful operation
        result = manager.run_with_timeout(success_func, timeout_seconds=1.0)
        assert result == "ok"
        assert manager.get_failure_count() == 1  # Should not increment
        
        # Another timeout
        with pytest.raises(OperationTimeoutError):
            manager.run_with_timeout(timeout_func, timeout_seconds=0.1)
        
        assert manager.get_failure_count() == 2
        manager.shutdown()

    def test_reset_failure_count(self):
        """Test that reset_failure_count works correctly."""
        manager = OperationManager(num_threads=2)
        
        def timeout_func():
            time.sleep(1.0)
        
        # Generate some failures
        for _ in range(5):
            with pytest.raises(OperationTimeoutError):
                manager.run_with_timeout(timeout_func, timeout_seconds=0.1)
        
        assert manager.get_failure_count() == 5
        
        # Reset and check return value
        old_count = manager.reset_failure_count()
        assert old_count == 5
        assert manager.get_failure_count() == 0
        
        manager.shutdown()

    def test_concurrent_operations(self):
        """Test that multiple operations can run concurrently."""
        manager = OperationManager(num_threads=4)
        results = []
        
        def concurrent_func(value):
            time.sleep(0.1)
            return value * 2
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    manager.run_with_timeout,
                    lambda v=i: concurrent_func(v),
                    1.0,
                    f"concurrent_{i}"
                )
                for i in range(4)
            ]
            results = [f.result() for f in futures]
        
        assert sorted(results) == [0, 2, 4, 6]
        assert manager.get_failure_count() == 0
        manager.shutdown()

    def test_thread_safety_of_failure_count(self):
        """Test that failure count is thread-safe under concurrent updates."""
        manager = OperationManager(num_threads=10)
        
        def timeout_func():
            time.sleep(1.0)
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    manager.run_with_timeout,
                    timeout_func,
                    0.1,
                    f"concurrent_timeout_{i}"
                )
                for i in range(20)
            ]
            
            # All should timeout
            for f in futures:
                with pytest.raises(OperationTimeoutError):
                    f.result()
        
        # Verify all failures were counted
        assert manager.get_failure_count() == 20
        manager.shutdown()

    def test_operations_continue_after_timeouts(self):
        """Test that operations can continue even after multiple timeouts."""
        manager = OperationManager(num_threads=10)
        
        def timeout_func():
            time.sleep(0.5)
        
        def success_func():
            return "success"
        
        # Generate multiple timeouts
        for i in range(5):
            with pytest.raises(OperationTimeoutError):
                manager.run_with_timeout(
                    timeout_func,
                    timeout_seconds=0.1,
                    label=f"timeout_{i}"
                )
        
        assert manager.get_failure_count() == 5
        
        # Wait a bit for thread pool to clear
        time.sleep(0.6)
        
        # Operations should still work
        result = manager.run_with_timeout(success_func, timeout_seconds=1.0)
        assert result == "success"
        
        manager.shutdown()

    def test_metadata_in_exception(self):
        """Test that metadata is included in the timeout exception."""
        manager = OperationManager(num_threads=2)
        
        def timeout_func():
            time.sleep(1.0)
        
        metadata = {"key": "value", "number": 42}
        
        with pytest.raises(OperationTimeoutError) as exc_info:
            manager.run_with_timeout(
                timeout_func,
                timeout_seconds=0.1,
                label="test_metadata",
                metadata=metadata
            )
        
        # Check that metadata is in the exception args
        assert exc_info.value.args[1] == metadata
        manager.shutdown()

    def test_shutdown(self):
        """Test that shutdown properly cleans up the thread pool."""
        manager = OperationManager(num_threads=2)
        
        def quick_func():
            return "done"
        
        # Run some operations
        for _ in range(3):
            manager.run_with_timeout(quick_func, timeout_seconds=1.0)
        
        # Shutdown should complete without errors
        manager.shutdown()
        
        # Thread pool should be shutdown
        assert manager.timeout_pool._shutdown

    def test_custom_num_threads(self):
        """Test that custom number of threads is respected."""
        manager = OperationManager(num_threads=8)
        assert manager.timeout_pool._max_workers == 8
        manager.shutdown()

    def test_exception_propagation(self):
        """Test that exceptions from the function are propagated (not timeout)."""
        manager = OperationManager(num_threads=2)
        
        def raising_func():
            raise ValueError("Custom error")
        
        with pytest.raises(ValueError) as exc_info:
            manager.run_with_timeout(
                raising_func,
                timeout_seconds=1.0,
                label="raising_op"
            )
        
        assert "Custom error" in str(exc_info.value)
        # Should not increment failure count for non-timeout errors
        assert manager.get_failure_count() == 0
        manager.shutdown()

    def test_zero_timeout(self):
        """Test behavior with zero timeout."""
        manager = OperationManager(num_threads=2)
        
        def any_func():
            time.sleep(0.01)
            return "result"
        
        # Zero timeout should immediately timeout
        with pytest.raises(OperationTimeoutError):
            manager.run_with_timeout(any_func, timeout_seconds=0.0)
        
        assert manager.get_failure_count() == 1
        manager.shutdown()

    def test_very_short_successful_operation(self):
        """Test that very short operations complete successfully."""
        manager = OperationManager(num_threads=2)
        
        def instant_func():
            return 123
        
        result = manager.run_with_timeout(instant_func, timeout_seconds=0.001)
        assert result == 123
        assert manager.get_failure_count() == 0
        manager.shutdown()
