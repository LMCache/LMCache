# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, Optional, Tuple, Union
import asyncio
import concurrent.futures

# Third Party
# we need to import torch since LMCacheRedisClient is built
# with torch.utils.cpp_extension.*Extension
# so we are linking against libc10.so and libtorch.so
# this line needs to be before the LMCacheRedisClient import
import torch  # noqa: F401

# First Party
from lmcache.lmcache_redis import LMCacheRedisClient


# sync and asyncio wrapper around the LMCacheRedisClient
# the pybinding interface allows us to work with both sync and async code
class RESPClient:
    def __init__(
        self,
        host: str,
        port: int,
        chunk_bytes: int,
        num_workers: int,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.loop = loop or asyncio.get_running_loop()
        self._client = LMCacheRedisClient(host, port, chunk_bytes, num_workers)
        self._fd = int(self._client.event_fd())
        self._closed = False

        # future_id -> (Future, op_name)
        # we support both types of futures since we only their basic interface
        self._pending: Dict[
            int, Tuple[Union[asyncio.Future, concurrent.futures.Future], str]
        ] = {}

        self.loop.add_reader(self._fd, self._on_ready)

    def _on_ready(self) -> None:
        if self._closed:
            return

        try:
            # drain until empty; completions can race in while processing
            while True:
                items = self._client.drain_completions()
                if not items:
                    return

                for future_id, ok, result_bool, error, result_bools in items:
                    fid = int(future_id)
                    entry = self._pending.pop(fid, None)
                    if entry is None:
                        continue

                    fut, op = entry
                    # fut can be asyncio.Future OR concurrent.future.Future
                    # the .done() and .set_result() and .set_exception()
                    # interface is the same
                    if fut.done():
                        continue

                    if ok:
                        if op == "exists":
                            fut.set_result(bool(result_bool))
                        elif op == "batch_exists":
                            # result_bools is a list of booleans (or None if empty)
                            if result_bools is not None:
                                fut.set_result(list(result_bools))
                            else:
                                fut.set_result([])
                        else:
                            fut.set_result(None)
                    else:
                        fut.set_exception(RuntimeError(str(error)))

        except Exception as e:
            # Native layer is likely broken; fail everything and tear down.
            self._fail_all(RuntimeError(f"native drain_completions failed: {e}"))
            self._shutdown_native(best_effort=True)

    def _fail_all(self, exc: Exception) -> None:
        """Fail all pending futures with the given exception."""
        for fid, (fut, _) in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    def _shutdown_native(self, best_effort: bool = False) -> None:
        """Shutdown the native client and cleanup resources."""
        try:
            self._closed = True
            self.loop.remove_reader(self._fd)
        except Exception:
            if not best_effort:
                raise

    def _register_future_async(self, op: str, future_id: int) -> asyncio.Future:
        fut = self.loop.create_future()
        self._pending[int(future_id)] = (fut, op)
        return fut

    def _register_future_sync(
        self, op: str, future_id: int
    ) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending[int(future_id)] = (fut, op)
        return fut

    async def get(self, key: str, buf: memoryview) -> None:
        future_id = int(self._client.submit_get(key, buf))
        fut = self._register_future_async("get", future_id)
        return await fut

    def get_sync(self, key: str, buf: memoryview) -> None:
        future_id = int(self._client.submit_get(key, buf))
        fut = self._register_future_sync("get", future_id)
        return fut.result()

    async def set(self, key: str, buf: memoryview) -> None:
        future_id = int(self._client.submit_set(key, buf))
        fut = self._register_future_async("set", future_id)
        return await fut

    def set_sync(self, key: str, buf: memoryview) -> None:
        future_id = int(self._client.submit_set(key, buf))
        fut = self._register_future_sync("set", future_id)
        return fut.result()

    async def exists(self, key: str) -> bool:
        future_id = int(self._client.submit_exists(key))
        fut = self._register_future_async("exists", future_id)
        return await fut

    def exists_sync(self, key: str) -> bool:
        future_id = int(self._client.submit_exists(key))
        fut = self._register_future_sync("exists", future_id)
        return fut.result()

    async def batch_get(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_get(keys, bufs))
        fut = self._register_future_async("batch_get", future_id)
        return await fut

    def batch_get_sync(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_get(keys, bufs))
        fut = self._register_future_sync("batch_get", future_id)
        return fut.result()

    async def batch_set(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_set(keys, bufs))
        fut = self._register_future_async("batch_set", future_id)
        return await fut

    def batch_set_sync(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_set(keys, bufs))
        fut = self._register_future_sync("batch_set", future_id)
        return fut.result()

    async def batch_exists(self, keys: list[str]) -> list[bool]:
        """Check existence of multiple keys in a single batch operation."""
        future_id = int(self._client.submit_batch_exists(keys))
        fut = self._register_future_async("batch_exists", future_id)
        return await fut

    def batch_exists_sync(self, keys: list[str]) -> list[bool]:
        """Check existence of multiple keys in a single
        batch operation (sync version)."""
        future_id = int(self._client.submit_batch_exists(keys))
        fut = self._register_future_sync("batch_exists", future_id)
        return fut.result()

    async def batched_exists(self, keys: list[str]) -> list[bool]:
        """Alias for batch_exists."""
        return await self.batch_exists(keys)

    def batched_exists_sync(self, keys: list[str]) -> list[bool]:
        """Alias for batch_exists_sync."""
        return self.batch_exists_sync(keys)

    def close(self) -> None:
        """Close the client and cleanup resources."""
        if not self._closed:
            self._shutdown_native(best_effort=True)
            self._fail_all(RuntimeError("Client closed"))
            self._client.close()


if __name__ == "__main__":
    # Standard
    import time

    async def run_benchmark():
        host = "127.0.0.1"
        port = 6379
        chunk_bytes = 4 * 1024 * 1024  # 4MB chunks
        num_workers = 8
        num_keys = 500

        client = RESPClient(host, port, chunk_bytes, num_workers)

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
            assert all(read_bufs[i] == buffers[i] for i in range(num_keys)), (
                "Data mismatch"
            )

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

    asyncio.run(run_benchmark())
