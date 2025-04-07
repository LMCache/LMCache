import torch
import zmq
import time
import argparse
import sys
from typing import List, Tuple

from lmcache.experimental.storage_backend.connector.nixl_connector import (
    NixlConfig, NIXLPipe, NixlRole, uuid_to_message, message_to_uuid
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test NixlChannel with sender/receiver roles')
    parser.add_argument('--role', type=str, required=True, choices=['sender', 'receiver'],
                       help='Role of this instance (sender or receiver)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host name/IP for connection')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port number for connection')
    
    args = parser.parse_args()

    keys, objs = generate_test_data(100, torch.Size([32, 2, 256, 1024]))

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        peer_host_name=args.host,
        peer_port=args.port,
        buffer_size=2**32,  # 4GB
        buffer_device='cuda',
    )
    
    context = zmq.Context()
    side_channel = context.socket(zmq.PAIR)
    if args.role == "sender":
        side_channel.bind(f"tcp://{args.host}:{args.port}")
    else:
        side_channel.connect(f"tcp://{args.host}:{args.port}")

    # Test the NIXLPipe
    pipe = NIXLPipe(config, side_channel)
    initial_uuid = "test"
    next_uuid = "new_test"
    
    if args.role == "sender":
        # Write data to buffer (not timed)
        num_objs, total_size = pipe.write_buffer(objs)
        logger.info(f"Wrote {num_objs} objects to the buffer")
        
        # Measure commit time (actual transfer)
        commit_start = time.time()
        new_uuid = pipe.commit_write(total_size, initial_uuid)
        commit_end = time.time()
        commit_time = commit_end - commit_start
        
        logger.info(f"New UUID: {new_uuid}")
        logger.info(f"Transfer time: {commit_time:.6f} seconds")
        transfer_throughput = calculate_throughput(total_size, commit_time)
        logger.info(f"Transfer throughput: {transfer_throughput:.2f} GB/s")
        
        assert new_uuid == next_uuid, \
            f"Expected new UUID '{next_uuid}', but got '{new_uuid}'"
    else:
        # Measure wait time (actual transfer)
        wait_start = time.time()
        pipe.wait_read(initial_uuid)
        wait_end = time.time()
        wait_time = wait_end - wait_start
        
        logger.info(f"Transfer wait time: {wait_time:.6f} seconds")
        
        # Read data from buffer (not timed)
        metadatas = [obj.metadata for obj in objs]
        received_objs = pipe.read_buffer(metadatas)
        total_size = sum(obj.get_size() for obj in received_objs)
        
        logger.info(f"Received {len(received_objs)} objects")
        transfer_throughput = calculate_throughput(total_size, wait_time)
        logger.info(f"Transfer throughput: {transfer_throughput:.2f} GB/s")
        
        # Check if the received objects are the same as the original objects
        for received_obj, original_obj in zip(received_objs, objs):
            assert torch.allclose(received_obj.tensor, original_obj.tensor), \
                    f"Data mismatch: received {received_obj.tensor.mean()} "\
                    f"but expected {original_obj.tensor.mean()}"
        
        # Send acknowledgment
        pipe.ack_receive(next_uuid)

    # Wait a bit before closing
    time.sleep(2)
    pipe.close()
    logger.info("Test completed successfully") 