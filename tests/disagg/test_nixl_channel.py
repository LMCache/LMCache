import torch
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
        obj.tensor.fill_(i+1)  # Fill with some test data, e.g., the index
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
        self.received_objs = []
        self.received_event = threading.Event()
        
    def __call__(self, keys, objs, is_view=True):
        logger.info(f"Observer received {len(keys)} keys and {len(objs)} objects")
        self.received_keys.extend(keys)
        
        # If these are views, we need to make copies
        if is_view:
            copied_objs = []
            for obj in objs:
                # Create a copy of the tensor
                copied_tensor = obj.tensor.clone()
                # Create a new memory object with the copied tensor
                copied_obj = TensorMemoryObj(copied_tensor, obj.metadata)
                copied_objs.append(copied_obj)
            self.received_objs.extend(copied_objs)
        else:
            self.received_objs.extend(objs)
            
        self.received_event.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test NixlChannel with sender/receiver roles')
    parser.add_argument('--role', type=str, required=True, choices=['sender', 'receiver'],
                       help='Role of this instance (sender or receiver)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host name/IP for connection')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port number for connection')
    parser.add_argument('--num-objs', type=int, default=100,
                       help='Number of objects to send')
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
    
    # Create the NixlChannel
    channel = NixlChannel(config)
    
    if args.role == "sender":
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
        
    else:  # receiver
        # Create and register an observer
        observer = TestObserver()
        channel.register_receive_observer(observer)
        
        # Wait for data to be received
        logger.info("Waiting to receive data...")
        if observer.received_event.wait(timeout=60):
            logger.info(f"Received {len(observer.received_keys)} keys and {len(observer.received_objs)} objects")
            
            # Verify the received data
            if len(observer.received_keys) != len(keys):
                logger.error(f"Expected {len(keys)} keys but received {len(observer.received_keys)}")
            
            # Check if the received objects match the original objects
            for i, (received_obj, original_obj) in enumerate(zip(observer.received_objs, objs)):
                if not torch.allclose(received_obj.tensor, original_obj.tensor):
                    logger.error(f"Data mismatch at index {i}: received {received_obj.tensor.mean()} "
                                f"but expected {original_obj.tensor.mean()}")
                    break
            else:
                logger.info("All data verified successfully!")
        else:
            logger.error("Timed out waiting for data")
    
    # Wait a bit before closing
    time.sleep(2)
    channel.close()
    logger.info("Test completed") 
