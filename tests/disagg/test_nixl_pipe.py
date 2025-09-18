# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple
import argparse
import time

# Third Party
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.connector.nixl_connector import (
    NixlConfig,
    NixlPipe,
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
        "--num-rounds",
        type=int,
        default=1,
        help="Number of rounds to run the experiment",
    )

    args = parser.parse_args()

    keys, objs = generate_test_data(100, torch.Size([32, 2, 256, 1024]))

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        receiver_host=args.host,
        receiver_port=args.port,
        buffer_size=2**32,  # 4GB
        buffer_device="cuda",
        enable_gc=False,
    )

    context = zmq.Context()  # type: ignore
    side_channel = context.socket(zmq.PAIR)  # type: ignore
    if args.role == "sender":
        side_channel.bind(f"tcp://{args.host}:{args.port}")
    else:
        side_channel.connect(f"tcp://{args.host}:{args.port}")

    # Test the NIXLPipe
    pipe = NixlPipe(config, side_channel)

    total_commit_time = 0.0
    total_wait_time = 0.0
    total_bytes_transferred = 0

    for round_num in range(args.num_rounds):
        logger.info(f"Starting round {round_num + 1}/{args.num_rounds}")

        initial_uuid = f"test_{round_num}"
        next_uuid = f"new_test_{round_num}"

        if args.role == "sender":
            # Write data to buffer (not timed)
            num_objs, total_size = pipe.write_buffer(objs)
            logger.info(f"Wrote {num_objs} objects to the buffer")

            # Measure commit time (actual transfer)
            commit_start = time.time()
            new_uuid = pipe.commit_write(total_size, initial_uuid)
            commit_end = time.time()
            commit_time = commit_end - commit_start

            total_commit_time += commit_time
            total_bytes_transferred += total_size

            logger.info(f"New UUID: {new_uuid}")
            logger.info(f"Transfer time: {commit_time:.6f} seconds")
            transfer_throughput = calculate_throughput(total_size, commit_time)
            logger.info(f"Transfer throughput: {transfer_throughput:.2f} GB/s")

            assert new_uuid == next_uuid, (
                f"Expected new UUID '{next_uuid}', but got '{new_uuid}'"
            )
        else:
            # Measure wait time (actual transfer)
            wait_start = time.time()
            pipe.wait_read(initial_uuid)
            wait_end = time.time()
            wait_time = wait_end - wait_start

            total_wait_time += wait_time

            logger.info(f"Transfer wait time: {wait_time:.6f} seconds")

            # Read data from buffer (not timed)
            metadatas = [obj.metadata for obj in objs]
            received_objs = pipe.read_buffer(metadatas)
            total_size = sum(obj.get_size() for obj in received_objs)

            total_bytes_transferred += total_size

            logger.info(f"Received {len(received_objs)} objects")
            transfer_throughput = calculate_throughput(total_size, wait_time)
            logger.info(f"Transfer throughput: {transfer_throughput:.2f} GB/s")

            # Check if the received objects are the same as the original objects
            for received_obj, original_obj in zip(received_objs, objs, strict=False):
                assert received_obj.tensor is not None
                assert original_obj.tensor is not None
                assert torch.allclose(received_obj.tensor, original_obj.tensor), (
                    f"Data mismatch: received {received_obj.tensor.mean()}"
                    f" but expected {original_obj.tensor.mean()}"
                )

            # Send acknowledgment
            pipe.ack_receive(next_uuid)

    # Print aggregate statistics
    if args.num_rounds > 1:
        if args.role == "sender":
            avg_time = total_commit_time / args.num_rounds
            logger.info(f"Average transfer time: {avg_time:.6f} seconds")
        else:
            avg_time = total_wait_time / args.num_rounds
            logger.info(f"Average wait time: {avg_time:.6f} seconds")

        avg_throughput = calculate_throughput(
            total_bytes_transferred,
            total_commit_time if args.role == "sender" else total_wait_time,
        )
        logger.info(
            f"Average throughput over {args.num_rounds} rounds: "
            f"{avg_throughput:.2f} GB/s"
        )

    # Wait a bit before closing
    time.sleep(2)
    pipe.close()
    logger.info("Test completed successfully")
