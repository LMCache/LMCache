import argparse
import threading
import time
from typing import List, Optional, Tuple

import torch

from lmcache.experimental.memory_management import (AdHocMemoryAllocator,
                                                    MemoryFormat, MemoryObj)
from lmcache.experimental.storage_backend.connector.nixl_connector_v2 import (
    NixlChannel, NixlConfig, NixlObserverInterface, NixlRole)
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


def generate_test_data(
    num_objs: int,
    shape: torch.Size,
    dtype: torch.dtype = torch.bfloat16
) -> Tuple[List[CacheEngineKey], List[MemoryObj]]:
    keys = []
    objs = []
    allocator = AdHocMemoryAllocator(
        device='cuda',  # Assuming we are using CUDA for the test
    )
    for i in range(num_objs):
        keys.append(
            CacheEngineKey(fmt="test",
                           model_name="test_model",
                           world_size=1,
                           worker_id=0,
                           chunk_hash=f"test_{i}"))
        obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_BLOB)
        obj.tensor.fill_(i + 1)  # Fill with some test data, e.g., the index
        objs.append(obj)
    return keys, objs


def calculate_throughput(total_bytes: int, elapsed_time: float) -> float:
    """Calculate throughput in GB/s"""
    if elapsed_time == 0:
        return float('inf')
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
        logger.info(
            f"Observer received {len(keys)} keys and {len(objs)} objects")

        # Clear previous data if we're starting a new batch
        if len(self.received_keys) == 0:
            self.reset()

        self.received_keys.extend(keys)

        # If these are views, we need to make copies
        if is_view:
            for obj in objs:
                copied_tensor = obj.tensor.clone().detach()
                self.received_tensors.append(copied_tensor)
        else:
            self.received_objs.extend(objs)

        if self.expected_count and len(
                self.received_keys) >= self.expected_count:
            self.received_event.set()

    def summarize(self):
        logger.info(f"Received {len(self.received_keys)} keys and "
                    f"{len(self.received_tensors)} tensors")

    def reset(self):
        # Explicitly free any existing tensors
        if hasattr(self, 'received_objs'):
            for obj in self.received_objs:
                del obj
            del self.received_objs

        if hasattr(self, 'received_keys'):
            del self.received_keys

        if hasattr(self, 'received_tensors'):
            del self.received_tensors

        self.received_keys = []
        self.received_tensors = []
        self.received_objs = []
        self.received_event = threading.Event()
        self.expected_count = None
        torch.cuda.empty_cache()  # Force CUDA memory cleanup


def send_and_measure_throughput_v2(channel: NixlChannel,
                                   keys: List[CacheEngineKey],
                                   objs: List[MemoryObj],
                                   total_size: int,
                                   batch_size: Optional[int] = None,
                                   simulate_workload: bool = False) -> float:
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
    logger.info(f"Sending {len(objs)} objects using "
                "zero_copy_send_with_callback...")

    elapsed_time = 0.

    if batch_size is None:
        # Original behavior - send all at once
        start_time = time.time()
        metadatas = [obj.metadata for obj in objs]
        channel.zero_copy_send_with_callback(
            keys=keys,
            metadatas=metadatas,
            callback=lambda dest_obj, idx=0: \
                dest_obj.tensor.copy_(objs[idx].tensor)
        )
        elapsed_time = time.time() - start_time
    else:
        # Send in batches
        elapsed_times: list[float] = []
        for i in range(0, len(objs), batch_size):
            start_time = time.time()
            batch_keys = keys[i:i + batch_size]
            batch_objs = objs[i:i + batch_size]
            batch_metadatas = [obj.metadata for obj in batch_objs]

            def callback(dest_obj, idx, batch_objs=batch_objs):
                dest_obj.tensor.copy_(batch_objs[idx].tensor)

            channel.zero_copy_send_with_callback(keys=batch_keys,
                                                 metadatas=batch_metadatas,
                                                 callback=callback)
            this_round = time.time() - start_time
            elapsed_times.append(this_round)
            logger.info(f"Sent batch {i//batch_size + 1}"
                        f"/{len(objs)//batch_size}"
                        f" in {this_round:.6f} seconds")
            if simulate_workload:
                time.sleep(0.05)  # Sleep 50ms between batches
        elapsed_time = sum(elapsed_times)  # type: ignore
        logger.info(f"Elapsed times: {elapsed_times}")

    logger.info(f"Sent {len(objs)} objects in {elapsed_time:.6f} seconds")
    throughput = calculate_throughput(total_size, elapsed_time)
    logger.info(f"Throughput: {throughput:.2f} GB/s")
    time.sleep(2)
    return throughput


def receive_and_verify_data(observer: TestObserver,
                            expected_keys: List[CacheEngineKey],
                            expected_objs: List[MemoryObj],
                            timeout: int = 60) -> bool:
    """Receive data through the channel and verify it matches expected data.
    
    Args:
        observer: The TestObserver that receives data
        expected_keys: List of expected cache engine keys
        expected_objs: List of expected memory objects
        timeout: Maximum time to wait for data in seconds
    
    Returns:
        bool: True if all data was received and verified successfully
    """
    try:
        # Wait for all data to be received
        logger.info("Waiting to receive data...")
        start_time = time.time()

        while len(observer.received_tensors) < len(expected_keys):
            if time.time() - start_time > timeout:
                logger.error("Timed out waiting for data")
                return False
            logger.info(f"Received {len(observer.received_tensors)}/"
                        f"{len(expected_keys)} tensors so far...")
            time.sleep(1)

        if len(observer.received_tensors) == len(expected_keys):
            logger.info(f"Received all {len(observer.received_keys)} keys and "
                        f"{len(observer.received_tensors)} tensors")

            # Verify the received data
            success = True
            for i, (received_tensor, original_tensor) in enumerate(
                    zip(observer.received_tensors, expected_objs)):
                if not torch.allclose(received_tensor, original_tensor.tensor):
                    logger.error(f"Data mismatch at index {i}")
                    success = False
                    break

            for i, (received_key, original_key) in enumerate(
                    zip(observer.received_keys, expected_keys)):
                if received_key != original_key:
                    logger.error(f"Key mismatch at index {i}")
                    success = False
                    break

            return success
        else:
            logger.error(f"Only received {len(observer.received_tensors)}/"
                         f"{len(expected_keys)} tensors before timeout")
            return False
    finally:
        # Always cleanup, even if verification fails
        observer.summarize()
        observer.reset()
        torch.cuda.empty_cache()


def test_allocate_for_send(channel: NixlChannel, shape: torch.Size,
                           dtype: torch.dtype) -> None:
    """Test the allocate_for_send API"""
    logger.info("Testing allocate_for_send API...")

    # Create test keys
    keys = [
        CacheEngineKey(fmt="test",
                       model_name="test_model",
                       world_size=1,
                       worker_id=0,
                       chunk_hash=f"test_alloc_{i}") for i in range(3)
    ]

    # Create test metadatas
    allocator = AdHocMemoryAllocator(device='cuda')
    temp_objs = [allocator.allocate(shape, dtype) for _ in range(3)]
    metadatas = [obj.metadata for obj in temp_objs]

    # Prepare send
    channel.prepare_send(keys, metadatas)

    # Allocate and fill objects
    for i in range(3):
        obj = channel.allocate_for_send(shape, dtype)
        assert obj is not None, "Failed to allocate memory for send"
        obj.tensor.fill_(i + 10)  # Fill with test data

    # Finish send
    channel.finish_send()
    logger.info("allocate_for_send test completed")


def main():
    parser = argparse.ArgumentParser(
        description='Test NixlChannel V2 with sender/receiver roles')
    parser.add_argument('--role',
                        type=str,
                        required=True,
                        choices=['sender', 'receiver'],
                        help='Role of this instance (sender or receiver)')
    parser.add_argument('--host',
                        type=str,
                        default='localhost',
                        help='Host name/IP for connection')
    parser.add_argument('--port',
                        type=int,
                        default=5555,
                        help='Port number for connection')
    parser.add_argument('--num-objs',
                        type=int,
                        default=100,
                        help='Number of objects to send')
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Size of batches to send (default: send all at once)')
    parser.add_argument(
        '--simulate-workload',
        action='store_true',
        help='Simulate workload by sleeping 50ms between batches')
    args = parser.parse_args()

    # Generate test data
    keys, objs = generate_test_data(args.num_objs,
                                    torch.Size([32, 2, 256, 1024]))
    total_size = sum(obj.get_size() for obj in objs)
    logger.info(f"Generated {len(objs)} objects with total size "
                f"{total_size / (1024*1024):.2f} MB")

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        peer_host_name=args.host,
        peer_port=args.port,
        buffer_size=2**32,  # 4GB
        buffer_device='cuda',
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
            simulate_workload=args.simulate_workload)
        logger.info(f"Throughput: {throughput:.2f} GB/s")
    else:  # receiver
        observer = TestObserver()
        observer.set_expected_count(len(keys))
        channel.register_receive_observer(observer)
        success = receive_and_verify_data(observer, keys, objs)
        if not success:
            logger.error("Data verification failed")

    # Wait a bit before closing
    time.sleep(2)
    channel.close()
    logger.info("Test completed")


if __name__ == "__main__":
    main()
