# SPDX-License-Identifier: Apache-2.0
# Standard
import multiprocessing as mp
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.accelerator import accelerator
from lmcache.v1.multiprocess.futures import GPUMessagingFuture, MessagingFuture

# ==============================================================================
# Helper Functions for GPUMessagingFuture Tests
# ==============================================================================


def _create_gpu_event_in_process(event_queue: mp.Queue, delay: float = 0.0):
    """Helper process that creates a GPU event and sends the IPC handle."""
    accelerator.init()
    if delay > 0:
        time.sleep(delay)

    # Create and record a GPU event with interprocess flag
    event = accelerator.Event(interprocess=True)
    event.record()
    event_bytes = event.ipc_handle()

    # Send the event handle to the main process
    event_queue.put(event_bytes)


def test_messaging_future_basic_usage():
    """Test basic usage of MessagingFuture: set result and retrieve it."""
    future = MessagingFuture[int]()

    # Initially, future should not be done
    assert not future.query(), "Future should not be done initially"

    # Set result
    future.set_result(42)

    # Future should now be done
    assert future.query(), "Future should be done after setting result"

    # Get result (should be immediate)
    result = future.result(timeout=1)
    assert result == 42, f"Expected result 42, got {result}"


def test_messaging_future_with_thread():
    """Test MessagingFuture with result set from another thread."""
    future = MessagingFuture[str]()

    def set_future_result():
        time.sleep(0.5)
        future.set_result("Hello from thread")

    # Start thread that will set the result
    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Initially should not be done
    assert not future.query(), "Future should not be done before thread sets result"

    # Wait for result
    result = future.result(timeout=2)
    assert result == "Hello from thread", f"Expected 'Hello from thread', got {result}"

    # Should be done now
    assert future.query(), "Future should be done after getting result"

    thread.join()


def test_messaging_future_wait_success():
    """Test wait method when result becomes available."""
    future = MessagingFuture[int]()

    def set_future_result():
        time.sleep(0.3)
        future.set_result(100)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait should return True when result is set
    success = future.wait(timeout=1)
    assert success, "Wait should return True when result is available"
    assert future.query(), "Future should be done after wait returns True"

    thread.join()


def test_messaging_future_wait_timeout():
    """Test wait method when timeout is reached."""
    future = MessagingFuture[int]()

    # Wait with short timeout (result never set)
    start_time = time.time()
    success = future.wait(timeout=0.2)
    elapsed = time.time() - start_time

    assert not success, "Wait should return False on timeout"
    assert not future.query(), "Future should not be done after timeout"
    assert 0.15 < elapsed < 0.3, f"Wait should respect timeout, elapsed: {elapsed}"


def test_messaging_future_result_timeout():
    """Test result method raises TimeoutError when timeout is reached."""
    future = MessagingFuture[int]()

    # Try to get result with timeout (result never set)
    with pytest.raises(
        TimeoutError, match="Future result not available within timeout"
    ):
        future.result(timeout=0.2)

    assert not future.query(), "Future should not be done after timeout"


def test_messaging_future_wait_no_timeout():
    """Test wait method without timeout (waits indefinitely until result is set)."""
    future = MessagingFuture[float]()

    def set_future_result():
        time.sleep(0.3)
        future.set_result(3.14)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait without timeout should wait until result is available
    success = future.wait()  # No timeout parameter
    assert success, "Wait should return True when result is set"
    assert future.result() == 3.14, "Result should be accessible after wait"

    thread.join()


def test_messaging_future_multiple_result_calls():
    """Test that result can be retrieved multiple times after being set."""
    future = MessagingFuture[str]()
    future.set_result("persistent value")

    # Get result multiple times
    result1 = future.result(timeout=0.1)
    result2 = future.result(timeout=0.1)
    result3 = future.result(timeout=0.1)

    assert result1 == result2 == result3 == "persistent value", (
        "Result should be retrievable multiple times"
    )


def test_messaging_future_complex_type():
    """Test MessagingFuture with complex types like lists and dicts."""
    future = MessagingFuture[dict]()

    complex_data = {"key1": [1, 2, 3], "key2": {"nested": "value"}, "key3": 42}

    def set_future_result():
        time.sleep(0.2)
        future.set_result(complex_data)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    result = future.result(timeout=1)
    assert result == complex_data, "Complex types should be preserved"

    thread.join()


# ==============================================================================
# GPUMessagingFuture Tests
# ==============================================================================


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_basic_usage():
    """Test basic usage of GPUMessagingFuture: create, wait, and get result."""
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    # Create the raw future that will return (event_bytes, result_value)
    raw_future = MessagingFuture[tuple[bytes, int]]()

    # Create GPUMessagingFuture from raw future
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    # Initially, future should not be done
    assert not gpu_future.query(), "GPUMessagingFuture should not be done initially"

    # Set result in raw future
    raw_future.set_result((event_bytes, 42))

    # Wait for GPU future to complete
    success = gpu_future.wait()
    assert success, "Wait should return True when result is available"

    # Get result
    result = gpu_future.result()
    assert result == 42, f"Expected result 42, got {result}"

    # Query should return True after completion
    assert gpu_future.query(), "GPUMessagingFuture should be done after wait"


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_with_thread():
    """Test GPUMessagingFuture with result set from another thread."""
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, str]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.5)
        raw_future.set_result((event_bytes, "Hello GPU"))

    # Start thread that will set the result
    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Initially should not be done
    assert not gpu_future.query(), "Future should not be done before thread sets result"

    # Wait for result
    result = gpu_future.result()
    assert result == "Hello GPU", f"Expected 'Hello GPU', got {result}"

    # Should be done now
    assert gpu_future.query(), "Future should be done after getting result"

    thread.join()


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_wait_no_timeout():
    """Test wait method without timeout (waits indefinitely
    until result is set).
    """
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, float]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 3.14))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait without timeout should wait until result is available
    success = gpu_future.wait()  # No timeout parameter
    assert success, "Wait should return True when result is set"
    assert gpu_future.result() == 3.14, "Result should be accessible after wait"

    thread.join()


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_wait_with_timeout_success():
    """Test that wait method works correctly with timeout when result is available."""
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 123))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait with timeout should return True when result is available
    success = gpu_future.wait(timeout=2.0)
    assert success, "Wait with timeout should return True when result is available"
    assert gpu_future.result() == 123, "Result should be accessible after wait"

    thread.join()


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_wait_timeout_reached():
    """Test that wait method returns False when timeout is reached."""
    accelerator.init()

    raw_future = MessagingFuture[tuple[bytes, int]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    # Wait with short timeout (result never set)
    start_time = time.time()
    success = gpu_future.wait(timeout=0.2)
    elapsed = time.time() - start_time

    assert not success, "Wait should return False on timeout"
    assert not gpu_future.query(), "Future should not be done after timeout"
    assert 0.15 < elapsed < 0.4, f"Wait should respect timeout, elapsed: {elapsed}"


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_result_with_timeout_success():
    """Test that result method works correctly with timeout when result is available."""
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 456))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Get result with timeout should succeed when result is available
    result = gpu_future.result(timeout=2.0)
    assert result == 456, f"Expected result 456, got {result}"

    thread.join()


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_result_timeout_reached():
    """Test that result method raises TimeoutError when timeout is reached."""
    accelerator.init()

    raw_future = MessagingFuture[tuple[bytes, int]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    # Try to get result with timeout (result never set)
    with pytest.raises(
        TimeoutError, match="GPUMessagingFuture result not available within timeout"
    ):
        gpu_future.result(timeout=0.2)


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_multiple_result_calls():
    """Test that result can be retrieved multiple times after being set."""
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, str]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    raw_future.set_result((event_bytes, "persistent gpu value"))

    # Get result multiple times
    result1 = gpu_future.result()
    result2 = gpu_future.result()
    result3 = gpu_future.result()

    assert result1 == result2 == result3 == "persistent gpu value", (
        "Result should be retrievable multiple times"
    )


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_query_before_and_after():
    """Test query method returns False before completion and True after."""
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    # Query before setting result
    assert not gpu_future.query(), "Query should return False before result is set"

    # Set result
    raw_future.set_result((event_bytes, 100))

    # Wait for completion
    gpu_future.wait()

    # Query after setting result
    assert gpu_future.query(), "Query should return True after result is set and waited"


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_complex_type():
    """Test GPUMessagingFuture with complex types like lists and dicts."""
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    complex_data = {"key1": [1, 2, 3], "key2": {"nested": "value"}, "key3": 42}

    raw_future = MessagingFuture[tuple[bytes, dict]]()
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.2)
        raw_future.set_result((event_bytes, complex_data))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    result = gpu_future.result()
    assert result == complex_data, "Complex types should be preserved"

    thread.join()


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_messaging_future_to_future():
    """Test converting MessagingFuture to GPUMessagingFuture
    using to_future method.
    """
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()

    # Convert to GPU future
    gpu_future = raw_future.to_future()

    # Verify it's a GPUMessagingFuture instance
    assert isinstance(gpu_future, GPUMessagingFuture), (
        "to_future should return GPUMessagingFuture instance"
    )

    # Set result and verify it works
    raw_future.set_result((event_bytes, 999))

    result = gpu_future.result()
    assert result == 999, f"Expected result 999, got {result}"


@pytest.mark.skipif(
    accelerator.name == "cpu",
    reason="GPU is required for GPUMessagingFuture tests",
)
def test_gpu_messaging_future_with_explicit_device():
    """Test GPUMessagingFuture with explicit device parameter."""
    accelerator.init()

    # Create GPU event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_gpu_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=5)
    process.join(timeout=2)

    device = accelerator.current_device()
    raw_future = MessagingFuture[tuple[bytes, str]]()

    # Create GPU future with explicit device
    gpu_future = GPUMessagingFuture.FromMessagingFuture(raw_future, device=device)

    # Set result
    raw_future.set_result((event_bytes, "explicit device"))

    # Get result
    result = gpu_future.result()
    assert result == "explicit device", f"Expected 'explicit device', got {result}"
