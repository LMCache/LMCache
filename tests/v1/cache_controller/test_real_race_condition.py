# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Callable, List
import asyncio
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.controllers import (
    KVController,
    RegistrationController,
)
from lmcache.v1.cache_controller.message import (
    DeRegisterMsg,
    KVAdmitMsg,
    RegisterMsg,
)


class MockClusterExecutor:
    """Mock cluster executor for testing."""

    async def execute(self, operation: str, msg):
        return None


@pytest.fixture
def kv_controller():
    """Create a KVController instance for testing."""
    controller = KVController()
    reg_controller = RegistrationController()
    mock_executor = MockClusterExecutor()
    controller.post_init(reg_controller=reg_controller, cluster_executor=mock_executor)
    reg_controller.post_init(kv_controller=controller, cluster_executor=mock_executor)
    return controller, reg_controller


def run_async_in_thread(async_func: Callable, errors: List[str]) -> None:
    """
    Helper function to run async function in a new thread with its own event loop.

    Args:
        async_func: Async function to execute
        errors: List to collect errors
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_func())
    except RuntimeError as e:
        if "dictionary changed size during iteration" in str(e):
            errors.append(f"RACE CONDITION DETECTED: {e}")
        else:
            errors.append(f"Other error: {e}")
    except ValueError as e:
        if "list.remove" in str(e):
            errors.append(f"RACE CONDITION: {e}")
        else:
            errors.append(f"Other error: {e}")
    except Exception as e:
        errors.append(f"Unexpected error: {e}")
    finally:
        loop.close()


def run_threads_and_collect_errors(
    thread_funcs: List[Callable], timeout: float = 5.0
) -> List[str]:
    """
    Helper function to run multiple threads and collect errors.

    Args:
        thread_funcs: List of functions to run in separate threads
        timeout: Timeout for thread join

    Returns:
        List of error messages
    """
    errors = []
    threads = [threading.Thread(target=func) for func in thread_funcs]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=timeout)
        if t.is_alive():
            errors.append("DEADLOCK: Thread did not finish in time")

    return errors


def check_race_condition_errors(errors: List[str]) -> None:
    """
    Helper function to check and fail test if race conditions detected.

    Args:
        errors: List of error messages
    """
    if not errors:
        print("SUCCESS: No race conditions detected, code is thread-safe!")
        return

    race_condition_found = any("RACE CONDITION" in err for err in errors)
    timeout_found = any("TIMEOUT" in err or "DEADLOCK" in err for err in errors)

    if race_condition_found:
        pytest.fail(
            f"Race condition detected! Code is NOT thread-safe. Errors: {errors}"
        )
    elif timeout_found:
        pytest.fail(f"DEADLOCK/TIMEOUT: Test infrastructure issue. Errors: {errors}")
    else:
        pytest.fail(f"Unexpected errors: {errors}")


@pytest.mark.asyncio
async def test_dict_iteration_race_in_deregister(kv_controller):
    """
    This test aggressively triggers the race condition in deregister.

    The bug is in kv_controller.deregister():
        for key in self.kv_pool:  # Iterating
            ...
            if not self.kv_pool[key]:
                del self.kv_pool[key]  # Modifying dict size during iteration!

    This WILL fail without proper locking.
    """
    controller, reg_controller = kv_controller

    # Register many workers
    num_workers = 50
    for i in range(num_workers):
        msg = RegisterMsg(
            instance_id=f"instance_{i}",
            worker_id=i,
            ip=f"192.168.1.{i}",
            port=8000 + i,
            peer_init_url=f"tcp://192.168.1.{i}:9000",
        )
        await reg_controller.register(msg)

    # Admit many keys for each worker
    for worker_id in range(num_workers):
        for key in range(100):
            msg = KVAdmitMsg(
                instance_id=f"instance_{worker_id}",
                worker_id=worker_id,
                key=key,
                location="cpu",
                seq_num=key,
            )
            await controller.admit(msg)

    print(f"Initial kv_pool size: {len(controller.kv_pool)}")

    errors = []

    async def deregister_range(start: int, end: int):
        """Deregister workers in given range"""
        for i in range(start, end):
            await controller.deregister(f"instance_{i}", i)

    # Create thread functions
    thread_funcs = [
        lambda: run_async_in_thread(lambda: deregister_range(0, 25), errors),
        lambda: run_async_in_thread(lambda: deregister_range(25, 50), errors),
    ]

    # Run threads and collect errors
    thread_errors = run_threads_and_collect_errors(thread_funcs)
    errors.extend(thread_errors)

    print(f"Final kv_pool size: {len(controller.kv_pool)}")
    print(f"Errors caught: {errors}")

    # If we caught race condition errors, the test should fail
    if errors:
        pytest.fail(f"Race condition detected! Errors: {errors}")


@pytest.mark.asyncio
async def test_check_then_act_race_in_worker_mapping(kv_controller):
    """
    Test check-then-act race condition in worker_mapping.

    The bug pattern:
        if instance_id in self.worker_mapping:  # Check
            self.worker_mapping[instance_id].remove(worker_id)  # Act
            # Another thread might have removed it between check and act!

    This test SHOULD fail without proper locking because asyncio.Lock
    does not work across different event loops in different threads.
    """
    controller, reg_controller = kv_controller
    instance_id = "shared_instance"

    # Register workers under same instance
    for i in range(20):
        msg = RegisterMsg(
            instance_id=instance_id,
            worker_id=i,
            ip=f"192.168.1.{i}",
            port=8000 + i,
            peer_init_url=f"tcp://192.168.1.{i}:9000",
        )
        await reg_controller.register(msg)

    errors = []

    async def deregister_workers_with_timeout(start: int, end: int):
        """Deregister workers with timeout"""
        for i in range(start, end):
            msg = DeRegisterMsg(
                instance_id=instance_id,
                worker_id=i,
                ip=f"192.168.1.{i}",
                port=8000 + i,
            )
            try:
                await asyncio.wait_for(reg_controller.deregister(msg), timeout=2.0)
            except asyncio.TimeoutError as e:
                errors.append(f"TIMEOUT: deregister operation timed out, {e}")
                break

    # Create thread functions that deregister same workers
    thread_funcs = [
        lambda: run_async_in_thread(
            lambda: deregister_workers_with_timeout(0, 10), errors
        ),
        lambda: run_async_in_thread(
            lambda: deregister_workers_with_timeout(0, 10), errors
        ),
    ]

    # Run threads and collect errors
    thread_errors = run_threads_and_collect_errors(thread_funcs)
    errors.extend(thread_errors)

    print(f"Errors caught: {errors}")

    # Check for race conditions
    check_race_condition_errors(errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
