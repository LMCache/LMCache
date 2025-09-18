# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple
import argparse
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj

# from lmcache.v1.storage_backend.connector.nixl_connector import (
#    NixlChannel, NixlConfig, NixlObserverInterface, NixlRole)
from lmcache.v1.storage_backend.connector.nixl_connector import (
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
        obj.tensor.fill_(i + 1)  # Fill with some test data, e.g., the index
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
        self.received_keys = []
        self.received_tensors = []
        self.received_objs = []
        self.received_event = threading.Event()
        self.expected_count = None
        self.reset()

    def set_expected_count(self, count: int):
        self.expected_count = count

    def __call__(self, keys, objs, is_view=True):
        logger.info(f"Observer received {len(keys)} keys and {len(objs)} objects")

        # Clear previous data if we're starting a new batch
        if len(self.received_keys) == 0:
            self.reset()

        self.received_keys.extend(keys)

        # If these are views, we need to make copies
        if is_view:
            for obj in objs:
                copied_tensor = (
                    obj.tensor.clone().detach()
                )  # Detach to ensure no gradient history
                self.received_tensors.append(copied_tensor)
                # copied_obj = TensorMemoryObj(copied_tensor, obj.metadata)
        else:
            self.received_objs.extend(objs)

        if self.expected_count and len(self.received_objs) >= self.expected_count:
            self.received_event.set()

    def summarize(self):
        logger.info(
            f"Received {len(self.received_keys)} keys and "
            f"{len(self.received_tensors)} tensors"
        )

    def reset(self):
        # Explicitly free any existing tensors
        if hasattr(self, "received_objs"):
            for obj in self.received_objs:
                del obj.raw_data
            del self.received_objs

        if hasattr(self, "received_keys"):
            del self.received_keys

        if hasattr(self, "received_tensors"):
            del self.received_tensors

        self.received_keys = []
        self.received_tensors = []
        self.received_event = threading.Event()
        self.expected_count = None
        torch.cuda.empty_cache()  # Force CUDA memory cleanup


def send_and_measure_throughput(
    channel: NixlChannel,
    keys: List[CacheEngineKey],
    objs: List[MemoryObj],
    total_size: int,
) -> float:
    """Send data through the channel and measure throughput.

    Args:
        channel: The NixlChannel to send data through
        keys: List of cache engine keys
        objs: List of memory objects to send
        total_size: Total size of objects in bytes

    Returns:
        float: Throughput in GB/s
    """
    # Wait a bit for the receiver to set up
    time.sleep(2)

    # Send the data
    logger.info(f"Sending {len(objs)} objects...")
    start_time = time.time()
    channel.send(keys, objs)
    end_time = time.time()

    elapsed_time = end_time - start_time
    logger.info(f"Sent {len(objs)} objects in {elapsed_time:.6f} seconds")
    throughput = calculate_throughput(total_size, elapsed_time)
    logger.info(f"Throughput: {throughput:.2f} GB/s")
    return throughput


def receive_and_verify_data(
    observer: TestObserver,
    channel: NixlChannel,
    expected_keys: List[CacheEngineKey],
    expected_objs: List[MemoryObj],
    timeout: int = 60,
) -> bool:
    """Receive data through the channel and verify it matches expected data.

    Args:
        channel: The NixlChannel to receive data through
        expected_keys: List of expected cache engine keys
        expected_objs: List of expected memory objects
        timeout: Maximum time to wait for data in seconds

    Returns:
        bool: True if all data was received and verified successfully
    """
    # Create and register an observer

    try:
        # Wait for all data to be received
        logger.info("Waiting to receive data...")
        start_time = time.time()

        while len(observer.received_tensors) < len(expected_keys):
            if time.time() - start_time > timeout:
                logger.error("Timed out waiting for data")
                return False
            logger.info(
                f"Received {len(observer.received_tensors)}/"
                f"{len(expected_keys)} tensors so far..."
            )
            time.sleep(1)

        if len(observer.received_tensors) == len(expected_keys):
            logger.info(
                f"Received all {len(observer.received_keys)} keys and "
                f"{len(observer.received_tensors)} tensors"
            )

            # Verify the received data
            success = True
            for i, (received_tensor, original_tensor) in enumerate(
                zip(observer.received_tensors, expected_objs, strict=False)
            ):
                if not torch.allclose(received_tensor, original_tensor.tensor):
                    logger.error(f"Data mismatch at index {i}")
                    success = False
                    break

            for i, (received_key, original_key) in enumerate(
                zip(observer.received_keys, expected_keys, strict=False)
            ):
                if received_key != original_key:
                    logger.error(f"Key mismatch at index {i}")
                    success = False
                    break

            return success
        else:
            logger.error(
                f"Only received {len(observer.received_objs)}/"
                f"{len(expected_keys)} objects before timeout"
            )
            return False
    finally:
        # Always cleanup, even if verification fails
        observer.summarize()
        observer.reset()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test NixlChannel with sender/receiver roles"
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
        f"Generated {len(objs)} objects with total size "
        f"{total_size / (1024 * 1024):.2f} MB"
    )

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        receiver_host=args.host,
        receiver_port=args.port,
        buffer_size=2**32,  # 4GB
        buffer_device="cuda",
        enable_gc=False,
    )

    # Create the NixlChannel
    channel = NixlChannel(config)

    if args.role == "sender":
        throughputs = []
        for i in range(args.num_rounds):
            logger.info(f"Round {i + 1}/{args.num_rounds}")
            throughput = send_and_measure_throughput(channel, keys, objs, total_size)
            throughputs.append(throughput)
        avg_throughput = sum(throughputs) / len(throughputs)
        logger.info(f"Average throughput: {avg_throughput:.2f} GB/s")
    else:  # receiver
        observer = TestObserver()
        observer.set_expected_count(len(keys))
        channel.register_receive_observer(observer)
        for i in range(args.num_rounds):
            logger.info(f"Round {i + 1}/{args.num_rounds}")
            success = receive_and_verify_data(observer, channel, keys, objs)

    # Wait a bit before closing
    time.sleep(2)
    channel.close()
    logger.info("Test completed")
