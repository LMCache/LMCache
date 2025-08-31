# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple
import argparse
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.connector.nixl_connector_v2 import (
    NixlChannel,
    NixlConfig,
    NixlObserverInterface,
    NixlRole,
)

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


class TestObserver(NixlObserverInterface):
    def __init__(self):
        self.key_to_tensors = {}  # Map keys to received tensors
        self.received_event = threading.Event()
        self.expected_count = None
        self.num_expected_senders = 1  # Default to 1 sender
        self.reset()

    def set_expected_count(self, count: int):
        self.expected_count = count

    def set_num_expected_senders(self, num_senders: int):
        self.num_expected_senders = num_senders

    def __call__(self, keys, objs, is_view=True):
        logger.info(f"Observer received {len(keys)} keys and {len(objs)} objects")

        # If these are views, we need to make copies
        if is_view:
            for i, obj in enumerate(objs):
                copied_tensor = obj.tensor.clone().detach()

                # Store tensor by key for verification
                key = keys[i]
                if key not in self.key_to_tensors:
                    self.key_to_tensors[key] = []
                self.key_to_tensors[key].append(copied_tensor)
        else:
            # For non-view objects, still store them by key
            for i, obj in enumerate(objs):
                key = keys[i]
                if key not in self.key_to_tensors:
                    self.key_to_tensors[key] = []
                self.key_to_tensors[key].append(obj.tensor)

        # Calculate total received tensors
        total_received = sum(len(tensors) for tensors in self.key_to_tensors.values())

        if (
            self.expected_count
            and total_received >= self.expected_count * self.num_expected_senders
        ):
            self.received_event.set()

    def summarize(self):
        total_tensors = sum(len(tensors) for tensors in self.key_to_tensors.values())
        logger.info(
            f"Received {len(self.key_to_tensors)} unique keys and "
            f"{total_tensors} total tensors"
        )

    def reset(self):
        # Explicitly free any existing tensors
        if hasattr(self, "key_to_tensors"):
            for tensors in self.key_to_tensors.values():
                for tensor in tensors:
                    del tensor
            del self.key_to_tensors

        self.key_to_tensors = {}
        self.received_event = threading.Event()
        self.expected_count = None
        torch.cuda.empty_cache()  # Force CUDA memory cleanup


def send_and_measure_throughput_v2(
    channel: NixlChannel,
    keys: List[CacheEngineKey],
    objs: List[MemoryObj],
    total_size: int,
    batch_size: Optional[int] = None,
    simulate_workload: bool = False,
) -> float:
    """Send data through the channel and measure throughput using V2 API.

    Args:
        channel: The NixlChannel to send data through
        keys: List of cache engine keys
        objs: List of memory objects to send
        total_size: Total size of objects in bytes
        batch_size: Size of batches to send (if None, send all at once)
        simulate_workload: If True, sleep 50ms between batches

    Returns:
        float: Throughput in GB/s
    """
    logger.info(f"Sending {len(objs)} objects using zero_copy_send_with_callback...")

    elapsed_time = 0.0

    if batch_size is None:
        # Original behavior - send all at once
        start_time = time.time()
        metadatas = [obj.metadata for obj in objs]
        channel.zero_copy_send_with_callback(
            keys=keys,
            metadatas=metadatas,
            callback=lambda dest_obj, idx=0: dest_obj.tensor.copy_(objs[idx].tensor),  # type: ignore
        )
        elapsed_time = time.time() - start_time
    else:
        # Send in batches
        elapsed_times: list[float] = []
        for i in range(0, len(objs), batch_size):
            start_time = time.time()
            batch_keys = keys[i : i + batch_size]
            batch_objs = objs[i : i + batch_size]
            batch_metadatas = [obj.metadata for obj in batch_objs]

            def callback(dest_obj, idx, batch_objs=batch_objs):
                dest_obj.tensor.copy_(batch_objs[idx].tensor)

            channel.zero_copy_send_with_callback(
                keys=batch_keys, metadatas=batch_metadatas, callback=callback
            )
            this_round = time.time() - start_time
            elapsed_times.append(this_round)
            logger.info(
                f"Sent batch {i // batch_size + 1}"
                f"/{len(objs) // batch_size}"
                f" in {this_round:.6f} seconds"
            )
            if simulate_workload:
                time.sleep(0.05)  # Sleep 50ms between batches
        elapsed_time = sum(elapsed_times)  # type: ignore
        logger.info(f"Elapsed times: {elapsed_times}")

    logger.info(f"Sent {len(objs)} objects in {elapsed_time:.6f} seconds")
    throughput = calculate_throughput(total_size, elapsed_time)
    logger.info(f"Throughput: {throughput:.2f} GB/s")
    time.sleep(2)
    return throughput


def receive_and_verify_data(
    observer: TestObserver,
    expected_keys: List[CacheEngineKey],
    expected_objs: List[MemoryObj],
    num_expected_senders: int = 1,
    timeout: int = 60,
) -> bool:
    """Receive data through the channel and verify it matches expected data.

    Args:
        observer: The TestObserver that receives data
        expected_keys: List of expected cache engine keys
        expected_objs: List of expected memory objects
        num_expected_senders: Number of senders expected to send the same data
        timeout: Maximum time to wait for data in seconds

    Returns:
        bool: True if all data was received and verified successfully
    """
    try:
        # Wait for all data to be received
        logger.info("Waiting to receive data...")
        start_time = time.time()
        expected_total = len(expected_keys) * num_expected_senders

        # Calculate total received tensors
        total_received = sum(
            len(tensors) for tensors in observer.key_to_tensors.values()
        )

        while total_received < expected_total:
            if time.time() - start_time > timeout:
                logger.error("Timed out waiting for data")
                return False
            logger.info(f"Received {total_received}/{expected_total} tensors so far...")
            time.sleep(1)
            # Update total received count
            total_received = sum(
                len(tensors) for tensors in observer.key_to_tensors.values()
            )

        if total_received >= expected_total:
            logger.info(
                f"Received all {len(observer.key_to_tensors)} unique keys and "
                f"{total_received} total tensors"
            )

            # Verify the received data
            success = True

            # Check that we received the expected number of tensors for each key
            for key in expected_keys:
                if key not in observer.key_to_tensors:
                    logger.error(f"Missing key: {key}")
                    success = False
                    continue

                if len(observer.key_to_tensors[key]) != num_expected_senders:
                    logger.error(
                        f"Expected {num_expected_senders} objs for key {key}, "
                        f"but got {len(observer.key_to_tensors[key])}"
                    )
                    success = False
                    continue

                chunk_hash = key.chunk_hash
                try:
                    idx = chunk_hash
                    expected_value = (idx + 1) / len(
                        expected_keys
                    )  # Match the value in generate_test_data

                    # Verify the data for this key
                    for tensor in observer.key_to_tensors[key]:
                        # Check if tensor values match expected value
                        if not torch.allclose(
                            tensor, torch.full_like(tensor, expected_value)
                        ):
                            logger.error(
                                f"Data mismatch for key {key}. "
                                f"Received value: {tensor.flatten()[0]}. "
                                f"Expected value: {expected_value}"
                            )
                            success = False
                except (IndexError, ValueError) as e:
                    logger.error(f"Error parsing chunk_hash {chunk_hash}: {e}")
                    success = False

            return success
        else:
            logger.error(
                f"Only received {total_received}/{expected_total} "
                "tensors before timeout"
            )
            return False
    finally:
        # Always cleanup, even if verification fails
        observer.summarize()
        observer.reset()
        torch.cuda.empty_cache()


@pytest.mark.skip(reason="test needs to be parameterized")
def test_allocate_for_send(
    channel: NixlChannel, shape: torch.Size, dtype: torch.dtype
) -> None:
    """Test the allocate_for_send API"""
    logger.info("Testing allocate_for_send API...")

    # Create test keys
    keys = [
        CacheEngineKey(
            fmt="test",
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=i,
        )
        for i in range(3)
    ]

    # Create test metadatas
    allocator = AdHocMemoryAllocator(device="cuda")
    temp_objs = [allocator.allocate(shape, dtype) for _ in range(3)]
    metadatas = [obj.metadata for obj in temp_objs]

    # Prepare send
    channel.prepare_send(keys, metadatas)

    # Allocate and fill objects
    for i in range(3):
        obj = channel.allocate_for_send(shape, dtype)
        assert obj is not None, "Failed to allocate memory for send"
        assert obj.tensor is not None
        obj.tensor.fill_(i + 10)  # Fill with test data

    # Finish send
    channel.finish_send()
    logger.info("allocate_for_send test completed")


def main():
    parser = argparse.ArgumentParser(
        description="Test NixlChannel V2 with sender/receiver roles"
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
        "--batch-size",
        type=int,
        help="Size of batches to send (default: send all at once)",
    )
    parser.add_argument(
        "--simulate-workload",
        action="store_true",
        help="Simulate workload by sleeping 50ms between batches",
    )
    parser.add_argument(
        "--num-expected-senders",
        type=int,
        default=1,
        help="Number of senders expected to connect (receiver only)",
    )
    args = parser.parse_args()

    # Generate test data
    keys, objs = generate_test_data(args.num_objs, torch.Size([32, 2, 256, 1024]))
    total_size = sum(obj.get_size() for obj in objs)
    logger.info(
        f"Generated {len(objs)} objects with total size "
        f"{total_size / (1024 * 1024):.2f} MB"
    )

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        receiver_host=args.host,
        receiver_port=args.port,
        buffer_size=2**32,  # 4GB
        buffer_device="cuda:0",
        enable_gc=False,
    )

    # Create the NixlChannel
    channel = NixlChannel(config)

    if args.role == "sender":
        throughput = send_and_measure_throughput_v2(
            channel,
            keys,
            objs,
            total_size,
            batch_size=args.batch_size,
            simulate_workload=args.simulate_workload,
        )
        logger.info(f"Throughput: {throughput:.2f} GB/s")
    else:  # receiver
        observer = TestObserver()
        observer.set_expected_count(len(keys))
        observer.set_num_expected_senders(args.num_expected_senders)
        channel.register_receive_observer(observer)
        success = receive_and_verify_data(
            observer, keys, objs, args.num_expected_senders
        )
        if not success:
            logger.error("Data verification failed")

    # Wait a bit before closing
    time.sleep(2)
    channel.close()
    logger.info("Test completed")


if __name__ == "__main__":
    main()
