# SPDX-License-Identifier: Apache-2.0
"""
Benchmark and example usage of the RESPClient.

This script demonstrates how to use the RESPClient for high-throughput
batch operations with Redis using the RESP protocol.
"""

# Standard
import asyncio
import time

# First Party
from lmcache.v1.storage_backend.resp_client import RESPClient


async def run_benchmark():
    host = "127.0.0.1"
    port = 6379
    chunk_bytes = 4 * 1024 * 1024  # 4MB chunks
    num_workers = 8
    num_keys = 500

    client = RESPClient(host, port, num_workers)

    try:
        print("Redis RESP Client Benchmark")
        print(f"Server: {host}:{port}, Workers: {num_workers}")
        print(f"Chunk size: {chunk_bytes / 1024:.0f}KB, Keys: {num_keys}")
        print("-" * 60)

        # Prepare test data
        print("starting buffer initialization (this might take a while)")
        keys = [f"bench:key:{i}" for i in range(num_keys)]
        buffers = [bytearray(chunk_bytes) for _ in range(num_keys)]
        for i, buf in enumerate(buffers):
            for j in range(chunk_bytes):
                buf[j] = (i + j) % 256

        print("buffer initialization complete")
        print("starting throughput benchmarks")

        # Batch SET
        t0 = time.perf_counter()
        await client.batch_set(keys, [memoryview(b) for b in buffers])
        t1 = time.perf_counter()
        elapsed_set = t1 - t0
        total_bytes_set = num_keys * chunk_bytes
        throughput_set = total_bytes_set / elapsed_set / (1024**3)
        print(
            f"Batch SET:    {throughput_set:6.2f} GB/s  "
            f"({total_bytes_set / (1024**3):.2f} GB written)"
        )

        # Batch GET
        read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
        t0 = time.perf_counter()
        await client.batch_get(keys, [memoryview(b) for b in read_bufs])
        t1 = time.perf_counter()
        elapsed_get = t1 - t0
        total_bytes_get = num_keys * chunk_bytes
        throughput_get = total_bytes_get / elapsed_get / (1024**3)
        print(
            f"Batch GET:    {throughput_get:6.2f} GB/s  "
            f"({total_bytes_get / (1024**3):.2f} GB read)"
        )

        # Verify data
        assert all(read_bufs[i] == buffers[i] for i in range(num_keys)), "Data mismatch"

        # Batch EXISTS
        t0 = time.perf_counter()
        exists_results = await client.batch_exists(keys)
        t1 = time.perf_counter()
        elapsed_exists = t1 - t0
        ops_per_sec = num_keys / elapsed_exists
        hits = sum(exists_results)
        print(f"Batch EXISTS: {ops_per_sec:6.0f} ops/s  ({hits}/{num_keys} hits)")

        # Test batched_exists alias
        results = await client.batched_exists(keys[:10])
        assert results == exists_results[:10], "batched_exists mismatch"

        print("-" * 60)
        print("All tests passed")

    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
