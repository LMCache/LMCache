import torch
import threading
import zmq
import time
import argparse
import sys
import threading
from typing import List, Tuple

from lmcache.experimental.storage_backend.connector.nixl_connector import (
    NixlConfig, NixlChannel, NixlRole, NixlObserverInterface
)
from lmcache.experimental.memory_management import (
    MemoryObj, TensorMemoryObj, AdHocMemoryAllocator, MemoryFormat
)
from lmcache.experimental.storage_backend.nixl_backend import NixlBackend
from lmcache.utils import CacheEngineKey
from lmcache.logging import init_logger

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
        keys.append(CacheEngineKey(
            fmt="test",
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=f"test_{i}"
        ))
        obj = allocator.allocate(shape, dtype, 
            fmt=MemoryFormat.KV_BLOB)
        obj.tensor.fill_((i+1) / num_objs) # Fill with some test data, e.g., the index
        objs.append(obj)
    return keys, objs

def calculate_throughput(total_bytes: int, elapsed_time: float) -> float:
    """Calculate throughput in GB/s"""
    if elapsed_time == 0:
        return float('inf')
    gb = total_bytes / (1024 * 1024 * 1024)
    return gb / elapsed_time

def send_and_measure_throughput(backend: NixlBackend, keys: List[CacheEngineKey], 
                              objs: List[MemoryObj], wait_time: float = 2.0) -> float:
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
    logger.info(f"Sending {len(objs)} objects...")
    
    backend.register_put_tasks(keys, [obj.metadata for obj in objs])
    start_time = time.time()
    for key, obj in zip(keys, objs):
        backend.submit_put_task(key, obj)
    backend.flush_put_tasks()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    logger.info(f"Sent {len(objs)} objects in {elapsed_time:.6f} seconds")
    throughput = calculate_throughput(total_size, elapsed_time)
    logger.info(f"Throughput: {throughput:.2f} GB/s")
    
    return throughput

def receive_and_verify_data(backend: NixlBackend, keys: List[CacheEngineKey], 
                          num_objs: int, timeout: float = 60.0) -> bool:
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
            logger.error(f"Timed out waiting for data. Received only {received_count}/{num_objs} objects.")
            return False
            
        time.sleep(0.1)  # Small sleep to avoid busy waiting
    
    passed_check = True
    if received_count == num_objs:
        logger.info(f"Received all {num_objs} objects")
        
        # Verify the received data
        for i, key in enumerate(keys):
            received_obj = backend.get_blocking(key)
            if received_obj is None:
                logger.error(f"Failed to retrieve object for key {key}")
                passed_check = False
                break
                
            # Check if the received object matches the original object
            expected_value = (i + 1) / num_objs
            actual_mean = received_obj.tensor.mean().item()
            
            # For bfloat16, we need some tolerance in the comparison
            if abs(actual_mean - expected_value) > 0.01:
                logger.error(f"Data mismatch for key {key}: received mean {actual_mean} "
                            f"but expected {expected_value}")
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
        logger.error(f"Only received {received_count}/{num_objs} objects")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test NixlBackend with sender/receiver roles')
    parser.add_argument('--role', type=str, required=True, choices=['sender', 'receiver'],
                       help='Role of this instance (sender or receiver)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host name/IP for connection')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port number for connection')
    parser.add_argument('--num-objs', type=int, default=100,
                       help='Number of objects to send')
    parser.add_argument('--num-rounds', type=int, default=1,
                       help='Number of rounds to run the experiment')
    args = parser.parse_args()

    # Generate test data
    keys, objs = generate_test_data(args.num_objs, torch.Size([32, 2, 256, 1024]))
    total_size = sum(obj.get_size() for obj in objs)
    logger.info(f"Generated {len(objs)} objects with total size {total_size / (1024*1024):.2f} MB")

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        peer_host_name=args.host,
        peer_port=args.port,
        buffer_size=2**32,  # 4GB
        buffer_device='cuda',
    )
    
    # Create the NixlBackend
    backend = NixlBackend(config)
    
    if args.role == "sender":
        throughputs = []
        for i in range(args.num_rounds):
            logger.info(f"Round {i+1}/{args.num_rounds}")
            throughput = send_and_measure_throughput(backend, keys, objs)
            throughputs.append(throughput)
        avg_throughput = sum(throughputs) / len(throughputs)
        logger.info(f"Average throughput: {avg_throughput:.2f} GB/s")
    else:  # receiver
        for i in range(args.num_rounds):
            logger.info(f"Round {i+1}/{args.num_rounds}")
            success = receive_and_verify_data(backend, keys, args.num_objs)
    
    # Wait a bit before closing
    time.sleep(2)
    backend.close()
    logger.info("Test completed")

