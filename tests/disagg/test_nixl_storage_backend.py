# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple
import argparse
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.connector.nixl_connector import NixlConfig, NixlRole
from lmcache.v1.storage_backend.nixl_backend import NixlBackend

logger = init_logger(__name__)


def generate_test_data(
    num_objs: int, shape: torch.Size, dtype: torch.dtype = torch.bfloat16
) -> Tuple[List[CacheEngineKey], List[MemoryObj]]:
    keys = []
    objs = []
    allocator = AdHocMemoryAllocator(
        device="cuda",  # Assuming we are using CUDA for the test
    )
    for i in range(num_objs):
        keys.append(
            CacheEngineKey(
                fmt="test",
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
            )
        )
        obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
        obj.tensor.fill_(
            (i + 1) / num_objs
        )  # Fill with some test data, e.g., the index
        objs.append(obj)
    return keys, objs


def calculate_throughput(total_bytes: int, elapsed_time: float) -> float:
    """Calculate throughput in GB/s"""
    if elapsed_time == 0:
        return float("inf")
    gb = total_bytes / (1024 * 1024 * 1024)
    return gb / elapsed_time


def send_and_measure_throughput(
    backend: NixlBackend,
    keys: List[CacheEngineKey],
    objs: List[MemoryObj],
    wait_time: float = 2.0,
) -> float:
    """Send objects through the backend and measure throughput.

    Args:
        backend: The NixlBackend instance
        keys: List of cache engine keys
        objs: List of memory objects to send
        wait_time: Time to wait for receiver setup in seconds

    Returns:
        float: Throughput in GB/s
    """
    # Wait for the receiver to set up
    time.sleep(wait_time)

    total_size = sum(obj.get_size() for obj in objs)
    logger.info("Sending %d objects...", len(objs))

    backend.register_put_tasks(keys, [obj.metadata for obj in objs])
    start_time = time.time()
    backend.batched_submit_put_task(keys, objs)
    backend.flush_put_tasks()
    end_time = time.time()

    elapsed_time = end_time - start_time
    logger.info("Sent %d objects in %.6f seconds", len(objs), elapsed_time)
    throughput = calculate_throughput(total_size, elapsed_time)
    logger.info("Throughput: %.2f GB/s", throughput)

    return throughput


def receive_and_verify_data(
    backend: NixlBackend,
    keys: List[CacheEngineKey],
    num_objs: int,
    timeout: float = 60.0,
) -> bool:
    """Receive and verify data through the backend.

    Args:
        backend: The NixlBackend instance
        keys: List of cache engine keys to check
        num_objs: Number of objects expected
        timeout: Maximum time to wait for data in seconds

    Returns:
        bool: True if all data was received and verified correctly
    """
    logger.info("Waiting to receive data...")

    # Poll until we receive all objects or timeout
    received_count = 0
    start_time = time.time()

    while received_count < num_objs:
        received_count = sum(1 for key in keys if backend.contains(key))

        if received_count == num_objs:
            break

        if time.time() - start_time > timeout:
            logger.error(
                "Timed out waiting for data. Received only %d/%d objects.",
                received_count,
                num_objs,
            )
            return False

        time.sleep(0.1)  # Small sleep to avoid busy waiting

    passed_check = True
    if received_count == num_objs:
        logger.info("Received all %d objects", num_objs)

        # Verify the received data
        for i, key in enumerate(keys):
            received_obj = backend.get_blocking(key)
            if received_obj is None or received_obj.tensor is None:
                logger.error(f"Failed to retrieve object for key {key}")
                passed_check = False
                break

            # Check if the received object matches the original object
            expected_value = (i + 1) / num_objs
            actual_mean = received_obj.tensor.mean().item()

            # For bfloat16, we need some tolerance in the comparison
            if abs(actual_mean - expected_value) > 0.01:
                logger.error(
                    "Mismatch for key %s: received mean %f but expected %f",
                    key,
                    actual_mean,
                    expected_value,
                )
                passed_check = False
                break

        if passed_check:
            logger.info("All data verified successfully!")
        else:
            logger.error("Data verification failed!")

        for key in keys:
            backend.remove(key)

        return passed_check
    else:
        logger.error("Only received %d/%d objects", received_count, num_objs)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test NixlBackend with sender/receiver roles"
    )
    parser.add_argument(
        "--role",
        type=str,
        required=True,
        choices=["sender", "receiver"],
        help="Role of this instance (sender or receiver)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host name/IP for connection",
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Port number for connection"
    )
    parser.add_argument(
        "--num-objs", type=int, default=100, help="Number of objects to send"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of rounds to run the experiment",
    )
    args = parser.parse_args()

    # Generate test data
    keys, objs = generate_test_data(args.num_objs, torch.Size([32, 2, 256, 1024]))
    total_size = sum(obj.get_size() for obj in objs)
    logger.info(
        "Generated %d objects with total size %.2f MB",
        len(objs),
        total_size / (1024 * 1024),
    )

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        receiver_host=args.host,
        receiver_port=args.port,
        buffer_size=2**32,  # 4GB
        buffer_device="cuda",
    )

    # Create the NixlBackend
    backend = NixlBackend(config)

    if args.role == "sender":
        throughputs = []
        for i in range(args.num_rounds):
            logger.info("Round %d/%d", i + 1, args.num_rounds)
            throughput = send_and_measure_throughput(backend, keys, objs)
            throughputs.append(throughput)
        avg_throughput = sum(throughputs) / len(throughputs)
        logger.info("Average throughput: %.2f GB/s", avg_throughput)
    else:  # receiver
        for i in range(args.num_rounds):
            logger.info("Round %d/%d", i + 1, args.num_rounds)
            success = receive_and_verify_data(backend, keys, args.num_objs)

    # Wait a bit before closing
    time.sleep(2)
    backend.close()
    logger.info("Test completed")
