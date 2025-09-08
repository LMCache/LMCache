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
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.connector.nixl_connector_v2 import (
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
        description="Test NixlPipe V2 with sender/receiver roles"
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
    parser.add_argument(
        "--num-objs",
        type=int,
        default=100,
        help="Number of objects to transfer",
    )
    parser.add_argument(
        "--simulate-work",
        action="store_true",
        help="Simulate some work on both sides",
    )

    args = parser.parse_args()

    keys, objs = generate_test_data(args.num_objs, torch.Size([32, 2, 256, 1024]))

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        receiver_host=args.host,
        receiver_port=args.port,
        buffer_size=2**32,  # 4GB
        buffer_device="cuda:0",
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

    total_transfer_time = 0.0
    total_bytes_transferred = 0

    for round_num in range(args.num_rounds):
        logger.info(f"Starting round {round_num + 1}/{args.num_rounds}")

        if args.role == "sender":
            # Sender side
            total_size = 0

            # Allocate and write data to buffer
            transfer_time = 0.0
            for idx, obj in enumerate(objs):
                if args.simulate_work and idx % 10 == 0:
                    time.sleep(0.05)  # Simulate some work

                # Use the new allocate_for_write method
                transfer_start = time.time()
                new_obj = pipe.allocate_for_write(
                    obj.get_shape(), obj.get_dtype(), obj.metadata.fmt
                )
                if new_obj is not None and new_obj.tensor is not None:
                    # Copy data from original object to the new one
                    new_obj.tensor.copy_(obj.tensor)
                    total_size += new_obj.get_size()
                transfer_time += time.time() - transfer_start
            # Measure transfer time
            flush_start = time.time()
            pipe.flush()  # This will wait for receiver's ack
            flush_end = time.time()
            transfer_time += flush_end - flush_start

            total_transfer_time += transfer_time
            total_bytes_transferred += total_size

            logger.info(f"Transfer time: {transfer_time:.6f} seconds")
            transfer_throughput = calculate_throughput(total_size, transfer_time)
            logger.info(f"Transfer throughput: {transfer_throughput:.2f} GB/s")

        else:
            # Receiver side
            # Read data from buffer
            transfer_start = time.time()
            metadatas = [obj.metadata for obj in objs]
            received_objs: list[MemoryObj] = []
            while len(received_objs) < len(metadatas):
                pipe.wait_read()
                new_objs = pipe.read_buffer(metadatas[len(received_objs) :])
                nobj_before = len(received_objs)
                for idx, obj in enumerate(new_objs):
                    cloned_tensor = obj.tensor.detach().clone()
                    received_objs.append(
                        TensorMemoryObj(cloned_tensor, obj.metadata, None)
                    )

                    # Simulate some work: 20ms per 10 objects
                    if args.simulate_work and len(received_objs) % 10 == 0:
                        time.sleep(0.02)

                pipe.ack_receive()
            transfer_end = time.time()
            transfer_time = transfer_end - transfer_start
            total_size = sum(obj.get_size() for obj in received_objs)

            total_bytes_transferred += total_size
            total_transfer_time += transfer_time

            logger.info(f"Received {len(received_objs)} objects")
            transfer_throughput = calculate_throughput(total_size, transfer_time)
            logger.info(f"Transfer throughput: {transfer_throughput:.2f} GB/s")

            # Check if the received objects are the same as the original objects
            assert len(received_objs) == len(objs), (
                "Number of received objects does not match the number of "
                "original objects"
            )
            for i, (received_obj, original_obj) in enumerate(
                zip(received_objs, objs, strict=False)
            ):
                assert received_obj.tensor is not None
                assert original_obj.tensor is not None
                assert torch.allclose(received_obj.tensor, original_obj.tensor), (
                    f"Data mismatch at index {i}: received "
                    f"{received_obj.tensor.mean()} "
                    f"but expected {original_obj.tensor.mean()}"
                )
            logger.info("Round passed")

    # Print aggregate statistics
    if args.num_rounds > 1:
        avg_time = total_transfer_time / args.num_rounds
        logger.info(f"Average transfer time: {avg_time:.6f} seconds")

        avg_throughput = calculate_throughput(
            total_bytes_transferred, total_transfer_time
        )
        logger.info(f"Total bytes transferred: {total_bytes_transferred}")
        logger.info(
            f"Average throughput over {args.num_rounds} rounds: "
            f"{avg_throughput:.2f} GB/s"
        )

    # Wait a bit before closing
    time.sleep(5)
    pipe.close()
    logger.info("Test completed successfully")
