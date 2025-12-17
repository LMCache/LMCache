# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from typing import List, Optional, Tuple, no_type_check
from urllib.parse import urlparse
import asyncio
import inspect
import os
import socket

# Third Party
from redis.asyncio.cluster import ClusterNode, RedisCluster
import redis.asyncio as redis

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class AsyncRESPClient:
    """
    Fully optimized deterministic-size RESP client (async version of RESP.py).

    Assumptions:
      - Every SET stores exactly buffer_size bytes.
      - Every GET returns exactly buffer_size bytes.
      - Keys always exist (no $-1).
      - RESP header is fixed-length: $<size>\r\n
    """

    def __init__(self, host: str, port: int, buffer_size: int):
        self.buffer_size = buffer_size

        # Pre-compute the deterministic RESP bulk header
        self.header = f"${buffer_size}\r\n".encode()
        self.header_len = len(self.header)
        self.trailer_len = 2  # b"\r\n"

        self._get_prefix = [
            memoryview(b"*2\r\n"),
            memoryview(b"$3\r\nGET\r\n"),
        ]

        self._set_prefix = [
            memoryview(b"*3\r\n"),
            memoryview(b"$3\r\nSET\r\n"),
        ]

        self._ok = memoryview(b"+OK\r\n")

        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.lock = asyncio.Lock()

    async def _connect(self):
        """Connect to Redis server"""
        if self.sock is None:
            self.sock = await asyncio.to_thread(
                socket.create_connection, (self.host, self.port)
            )

    async def _recv_exactly(self, n: int, into: memoryview | None = None):
        """
        Reads exactly n bytes.
        If 'into' is provided, fills that buffer; else returns a new buffer.
        """
        if into is not None:
            view = into
        else:
            buf = bytearray(n)
            view = memoryview(buf)

        total = 0
        while total < n:
            if self.sock is None:
                raise ConnectionError("Socket not connected")
            m = await asyncio.to_thread(
                self.sock.recv_into, view[total : total + (n - total)]
            )
            if m == 0:
                raise ConnectionError("Socket closed during recv_exactly")
            total += m

        return view if into is None else None

    async def _send_all_via_sendmsg(self, parts: list[memoryview]):
        """
        Zero-copy scatter/gather write with correct partial-write handling.
        """
        assert self.sock is not None, "Socket not connected"
        while parts:
            n_sent = await asyncio.to_thread(self.sock.sendmsg, parts)
            if n_sent == 0:
                raise ConnectionError("Broken connection during sendmsg")

            sent = 0
            while parts and sent < n_sent:
                p = parts[0]
                p_len = len(p)
                remain = n_sent - sent

                if remain >= p_len:
                    parts.pop(0)
                    sent += p_len
                else:
                    parts[0] = p[remain:]
                    break

    async def get(self, key: str, recv_buf: memoryview) -> Optional[int]:
        """
        Retrieves exactly buffer_size bytes into recv_buf.
        Uses deterministic header length (no parsing).
        Returns None if key doesn't exist.
        """
        async with self.lock:
            await self._connect()

            key_b = key.encode()
            key_len_hdr = f"${len(key_b)}\r\n".encode()

            # Build scatter-gather message
            parts = [
                *self._get_prefix,
                memoryview(key_len_hdr),
                memoryview(key_b),
                memoryview(b"\r\n"),
            ]

            # Send GET command
            await self._send_all_via_sendmsg(parts)

            # 1. Read header byte-by-byte until \r\n
            header_bytes = bytearray()
            cr_seen = False

            # Read $ byte
            dollar = bytearray(1)
            await self._recv_exactly(1, memoryview(dollar))
            if dollar[0] != ord("$"):
                raise ValueError(f"Unexpected GET response: {dollar!r}")
            header_bytes.extend(dollar)

            # Read until \r\n
            while True:
                ch = bytearray(1)
                await self._recv_exactly(1, memoryview(ch))
                header_bytes.extend(ch)

                if ch == b"\r":
                    cr_seen = True
                elif ch == b"\n" and cr_seen:
                    break
                else:
                    cr_seen = False

            # 2. Check if it's $-1\r\n (key not found)
            if header_bytes == b"$-1\r\n":
                return None

            # 3. Verify it matches our expected header
            if header_bytes != self.header:
                raise ValueError(
                    f"Unexpected GET header: {header_bytes!r} vs "
                    f"expected {self.header!r}"
                )

            # 4. Read VALUE directly into caller buffer (zero-copy)
            if len(recv_buf) < self.buffer_size:
                raise ValueError("recv_buf too small")

            await self._recv_exactly(self.buffer_size, recv_buf)

            # 5. Read trailing CRLF
            tmp = bytearray(2)
            await self._recv_exactly(2, memoryview(tmp))

            if tmp != b"\r\n":
                raise ValueError("Missing final CRLF after GET body")

            return self.buffer_size

    async def set(self, key: str, send_buf: memoryview):
        """
        Sends exactly buffer_size bytes as the value.
        """
        async with self.lock:
            await self._connect()

            if len(send_buf) != self.buffer_size:
                raise ValueError("send_buf must be exactly buffer_size bytes.")

            key_b = key.encode()
            key_len_hdr = f"${len(key_b)}\r\n".encode()

            parts = [
                *self._set_prefix,
                memoryview(key_len_hdr),
                memoryview(key_b),
                memoryview(b"\r\n"),
                memoryview(self.header),  # $<buffer_size>\r\n
                send_buf,  # actual payload (zero-copy)
                memoryview(b"\r\n"),
            ]

            await self._send_all_via_sendmsg(parts)

            # Expect "+OK\r\n"
            tmp = bytearray(5)
            await self._recv_exactly(5, memoryview(tmp))

            if tmp != b"+OK\r\n":
                raise ValueError(f"Unexpected SET reply: {tmp!r}")

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        async with self.lock:
            await self._connect()

            key_b = key.encode()
            key_len_hdr = f"${len(key_b)}\r\n".encode()

            # Build EXISTS command
            parts = [
                memoryview(b"*2\r\n"),
                memoryview(b"$6\r\nEXISTS\r\n"),
                memoryview(key_len_hdr),
                memoryview(key_b),
                memoryview(b"\r\n"),
            ]

            await self._send_all_via_sendmsg(parts)

            # Read response: :1\r\n or :0\r\n
            resp = bytearray(4)
            await self._recv_exactly(4, memoryview(resp))
            if resp[:1] != b":":
                raise ValueError(f"Unexpected EXISTS response: {resp!r}")
            return resp[1:2] == b"1"

    async def close(self) -> None:
        """Close connection"""
        async with self.lock:
            if self.sock is not None:
                await asyncio.to_thread(self.sock.close)
                self.sock = None


class RedisConnector(RemoteConnector):
    """
    The remote url should start with "redis://" and only have one host-port pair
    Uses RESP protocol directly for zero-copy operations.
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # Parse URL to get host and port
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379

        self.host = host
        self.port = port
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        # RESP client will be initialized lazily when we discover buffer_size
        self.resp_client: Optional[AsyncRESPClient] = None

        # Fallback redis-py client for discovery (STRLEN command)
        self.max_connections = 150
        self.sem = asyncio.Semaphore(self.max_connections)
        self.pool = redis.ConnectionPool.from_url(
            url, max_connections=self.max_connections
        )
        self.connection = redis.Redis.from_pool(self.pool)

        self.pq_executor = AsyncPQExecutor(loop)

        # Existence cache (all keys are same size, so we just track existence)
        self.existence_cache: dict[str, bool] = {}

    def post_init(self):
        """Post-initialization after chunk metadata is set"""
        # If full_chunk_size is available, initialize RESP client
        if self.full_chunk_size is not None:
            self.resp_client = AsyncRESPClient(
                self.host, self.port, self.full_chunk_size
            )
            logger.info(
                f"Redis connector initialized with "
                f"full_chunk_size={self.full_chunk_size}"
            )
        else:
            logger.info(
                "Redis connector initialized without full_chunk_size, "
                "will discover on first access"
            )

    async def _check_key_exists(self, key_str: str) -> bool:
        """Check if key exists using EXISTS command"""
        async with self.sem:
            exists = await self.connection.exists(key_str)
            return bool(exists)

    async def _exists(self, key: CacheEngineKey) -> bool:
        key_str = key.to_string()
        # Use existence cache
        if key_str in self.existence_cache:
            return self.existence_cache[key_str]

        # Check existence
        exists = await self._check_key_exists(key_str)
        self.existence_cache[key_str] = exists

        # If key exists and we don't have full_chunk_size yet, discover it
        if exists and self.full_chunk_size is None:
            # Use STRLEN to discover the size
            async with self.sem:
                size = await self.connection.strlen(key_str)
                if size is not None and size > 0:
                    self.full_chunk_size = size
                    # Initialize RESP client now that we know the size
                    if self.resp_client is None:
                        self.resp_client = AsyncRESPClient(
                            self.host, self.port, self.full_chunk_size
                        )
                        logger.info(
                            f"Discovered full_chunk_size={self.full_chunk_size} "
                            f"and initialized RESP client"
                        )

        return exists

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(future.result())

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        # Check existence cache first
        if key_str in self.existence_cache and not self.existence_cache[key_str]:
            return None

        # If we don't have full_chunk_size yet, discover it from first key
        if self.full_chunk_size is None:
            # Check if key exists and get its size
            exists = await self._check_key_exists(key_str)
            self.existence_cache[key_str] = exists
            if not exists:
                return None

            # Discover size using STRLEN
            async with self.sem:
                size = await self.connection.strlen(key_str)
                if size is not None and size > 0:
                    self.full_chunk_size = size
                    # Initialize RESP client now that we know the size
                    if self.resp_client is None:
                        self.resp_client = AsyncRESPClient(
                            self.host, self.port, self.full_chunk_size
                        )
                        logger.info(
                            f"Discovered full_chunk_size={self.full_chunk_size} "
                            f"and initialized RESP client"
                        )
                else:
                    self.existence_cache[key_str] = False
                    return None

        # Allocate memory - meta_shape should be set via init_chunk_meta
        if self.meta_shape is None:
            logger.error(
                f"meta_shape not set for RedisConnector. "
                f"This should be set via init_chunk_meta(). "
                f"Key: {key_str}"
            )
            return None

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shape,
            self.meta_dtype,
            self.meta_fmt,
        )

        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # Check if stored size matches expected size
        if (
            self.full_chunk_size is not None
            and memory_obj.get_size() != self.full_chunk_size
        ):
            logger.error(
                f"Size mismatch for {key_str}: Redis has {self.full_chunk_size} bytes, "
                f"but current config expects {memory_obj.get_size()} bytes. "
                f"This usually means the data was stored with different chunk_size "
                f"or model configuration. Please use matching config or clear Redis."
            )
            memory_obj.ref_count_down()
            self.existence_cache[key_str] = False
            return None

        # Get data directly using RESP client (zero-copy) if available
        if self.resp_client is not None:
            try:
                if isinstance(memory_obj.byte_array, memoryview):
                    view = memory_obj.byte_array
                    if view.format == "<B":
                        view = view.cast("B")
                else:
                    view = memoryview(memory_obj.byte_array)

                bytes_read = await self.resp_client.get(key_str, view)

                # RESP client returns buffer_size on success, None if key doesn't exist
                if bytes_read is None:
                    # Key doesn't exist
                    memory_obj.ref_count_down()
                    self.existence_cache[key_str] = False
                    return None

                if bytes_read != self.full_chunk_size:
                    logger.error(
                        f"Size mismatch for {key_str}: RESP returned "
                        f"{bytes_read} bytes, but expected "
                        f"{self.full_chunk_size} bytes."
                    )
                    memory_obj.ref_count_down()
                    self.existence_cache[key_str] = False
                    return None

                self.existence_cache[key_str] = True
                return memory_obj
            except ValueError as e:
                # Key doesn't exist or size mismatch
                logger.debug(f"Key not found or error: {e}")
                memory_obj.ref_count_down()
                self.existence_cache[key_str] = False
                return None
            except Exception as e:
                logger.debug(f"Error getting via RESP: {e}, falling back to redis-py")
                # Fall through to redis-py fallback

        # Fallback to redis-py if RESP client not available
        async with self.sem:
            kv_bytes = await self.connection.get(key_str)
            if kv_bytes is None:
                self.existence_cache[key_str] = False
                memory_obj.ref_count_down()
                return None

            if isinstance(memory_obj.byte_array, memoryview):
                view = memory_obj.byte_array
                if view.format == "<B":
                    view = view.cast("B")
            else:
                view = memoryview(memory_obj.byte_array)

            if isinstance(kv_bytes, (bytes, bytearray)):
                view[: len(kv_bytes)] = kv_bytes
            elif isinstance(kv_bytes, str):
                converted = kv_bytes.encode("utf-8")
                view[: len(converted)] = converted
            else:
                converted = bytes(kv_bytes)
                view[: len(converted)] = converted

            self.existence_cache[key_str] = True
            return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        # calling self.put will create a circular dependency
        await asyncio.gather(
            *(
                self._put(key, memory_obj)
                for key, memory_obj in zip(keys, memory_objs, strict=False)
            )
        )

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()

        # Store raw data only (no metadata)
        kv_bytes = memory_obj.byte_array
        kv_size = (
            len(kv_bytes)
            if isinstance(kv_bytes, (bytes, bytearray))
            else memory_obj.get_size()
        )

        # If we don't have full_chunk_size yet, discover it from this put
        if self.full_chunk_size is None:
            self.full_chunk_size = kv_size
            # Initialize RESP client now that we know the size
            if self.resp_client is None:
                self.resp_client = AsyncRESPClient(
                    self.host, self.port, self.full_chunk_size
                )
                logger.info(
                    f"Discovered full_chunk_size={self.full_chunk_size} "
                    f"from put and initialized RESP client"
                )

        if isinstance(kv_bytes, memoryview):
            view = kv_bytes
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(kv_bytes)

        # Use RESP client if available and size matches
        if self.resp_client is not None and kv_size == self.full_chunk_size:
            try:
                await self.resp_client.set(key_str, view)
                self.existence_cache[key_str] = True
                return
            except Exception as e:
                logger.debug(f"Error setting via RESP: {e}, falling back to redis-py")
                # Fall through to redis-py fallback

        # Fallback to redis-py
        async with self.sem:
            await self.connection.set(key_str, bytes(view))
            self.existence_cache[key_str] = True

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        if self.resp_client is not None:
            await self.resp_client.close()
        await self.connection.close()
        logger.info("Closed the redis connection")

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            key_str = key.to_string()
            # Use existence cache
            if key_str in self.existence_cache:
                if self.existence_cache[key_str]:
                    num_hit_counts += 1
                    continue
                else:
                    return num_hit_counts

            # Check existence
            exists = await self._check_key_exists(key_str)
            self.existence_cache[key_str] = exists

            # If key exists and we don't have full_chunk_size yet, discover it
            if exists and self.full_chunk_size is None:
                # Use STRLEN to discover the size
                async with self.sem:
                    size = await self.connection.strlen(key_str)
                    if size is not None and size > 0:
                        self.full_chunk_size = size
                        # Initialize RESP client now that we know the size
                        if self.resp_client is None:
                            self.resp_client = AsyncRESPClient(
                                self.host, self.port, self.full_chunk_size
                            )
                            logger.info(
                                f"Discovered full_chunk_size={self.full_chunk_size} "
                                f"and initialized RESP client"
                            )

            if exists:
                num_hit_counts += 1
                continue
            else:
                return num_hit_counts
        return num_hit_counts

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # calling self.get will create a circular dependency
        results = await asyncio.gather(*(self._get(key) for key in keys))
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )


class RedisSentinelConnector(RemoteConnector):
    """
    Uses redis.Sentinel to connect to a Redis cluster.
    The hosts are specified in the config file, started with "redis-sentinel://"
    and separated by commas.

    Example:
        remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"

    Extra environment variables:
    - REDIS_SERVICE_NAME (required) -- service name for redis.
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    ENV_REDIS_TIMEOUT = "REDIS_TIMEOUT"
    ENV_REDIS_SERVICE_NAME = "REDIS_SERVICE_NAME"

    def __init__(
        self,
        hosts_and_ports: List[Tuple[str, int]],
        username: str,
        password: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # Get service name
        match os.environ.get(self.ENV_REDIS_SERVICE_NAME):
            case None:
                logger.warning(
                    f"Environment variable {self.ENV_REDIS_SERVICE_NAME} is "
                    f"not found, using default value 'redismaster'"
                )
                service_name = "redismaster"
            case value:
                service_name = value

        timeout: float = -1000.0

        # Get timeout
        match os.environ.get(self.ENV_REDIS_TIMEOUT):
            case None:
                timeout = 1
            case value:
                timeout = float(value)

        logger.info(f"Host and ports: {hosts_and_ports}")
        self.sentinel = redis.Sentinel(hosts_and_ports, socket_timeout=timeout)
        self.master = self.sentinel.master_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )
        self.slave = self.sentinel.slave_for(
            service_name, socket_timeout=timeout, username=username, password=password
        )

        self.local_cpu_backend = local_cpu_backend

    async def exists(self, key: CacheEngineKey) -> bool:
        return bool(self.slave.exists(key.to_string() + "metadata"))

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return bool(self.slave.exists(key.to_string() + "metadata"))

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        metadata_bytes = self.slave.get(key_str + "metadata")

        if metadata_bytes is None:
            return None

        assert not inspect.isawaitable(metadata_bytes)

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # TODO(Jiayi): Find a way to do `get` inplace
        kv_bytes = self.slave.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A background sweeper might be better
            # for the sake of performance.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            self.master.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[0 : metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[0 : metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[0 : metadata.length] = converted

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shape, kv_dtype, memory_format
        ).serialize()

        key_str = key.to_string()
        # kv bytes needs to be set first to avoid race condition
        self.master.set(key_str + "kv_bytes", kv_bytes)
        self.master.set(key_str + "metadata", metadata_bytes)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.master.close()
        self.slave.close()


class RedisClusterConnector(RemoteConnector):
    """
    The remote url starts with "redis-cluster:// and can include one or
    multiple hosts:ports, separated by commas.

    Example:
        remote_url: "redis-cluster://host1:7000,host2:7000,host3:7000"

    Extra environment variables:
    - REDIS_TIMEOUT (optional) -- Timeout in seconds, default is 1 if not set
    """

    def __init__(
        self,
        hosts_and_ports: List[Tuple[str, int]],
        username: str,
        password: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # Convert hosts_and_ports to startup_nodes format expected by RedisCluster
        startup_nodes = [ClusterNode(h, p) for (h, p) in hosts_and_ports]

        # set a large max
        self.max_connections = 150
        # redis will crash if we have more than max_connections connections
        self.sem = asyncio.Semaphore(self.max_connections)

        # Initialize cluster connection
        self.cluster = RedisCluster(
            startup_nodes=startup_nodes,
            username=username,
            password=password,
            max_connections=self.max_connections,
            decode_responses=False,
        )
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.pq_executor = AsyncPQExecutor(loop)

    async def _exists(self, key: CacheEngineKey) -> bool:
        async with self.sem:
            return bool(await self.cluster.exists(key.to_string() + "metadata"))

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(future.result())

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        async with self.sem:
            metadata_bytes = await self.cluster.get(key_str + "metadata")

            if metadata_bytes is None:
                return None

            assert not inspect.isawaitable(metadata_bytes)

            metadata = RemoteMetadata.deserialize(memoryview(metadata_bytes))

            memory_obj = self.local_cpu_backend.allocate(
                metadata.shape,
                metadata.dtype,
                metadata.fmt,
            )
            if memory_obj is None:
                logger.warning("Failed to allocate memory during remote receive")
                return None

            # TODO(Jiayi): Find a way to do `get` inplace
            kv_bytes = await self.cluster.get(key_str + "kv_bytes")

        assert not inspect.isawaitable(kv_bytes)

        if kv_bytes is None:
            # TODO (Jiayi): We might need a way to better handle
            # consistency issues.
            # TODO (Jiayi): A better way is to aggregate metadata
            # and kv cache in one key.
            logger.warning(
                "Key exists but KV cache does not exist."
                "Might happen when the cache is evicted by redis."
            )
            async with self.sem:
                await self.cluster.delete(key_str + "metadata")
            return None

        if isinstance(memory_obj.byte_array, memoryview):
            view = memory_obj.byte_array
            if view.format == "<B":
                view = view.cast("B")
        else:
            view = memoryview(memory_obj.byte_array)

        if isinstance(kv_bytes, (bytes, bytearray)):
            view[: metadata.length] = kv_bytes
        elif isinstance(kv_bytes, str):
            converted = kv_bytes.encode("utf-8")
            view[: metadata.length] = converted
        else:
            converted = bytes(kv_bytes)
            view[: metadata.length] = converted

        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def support_batched_put(self) -> bool:
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        # calling self.put will create a circular dependency
        await asyncio.gather(
            *(
                self._put(key, memory_obj)
                for key, memory_obj in zip(keys, memory_objs, strict=False)
            )
        )

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shape, kv_dtype, memory_format
        ).serialize()

        key_str = key.to_string()
        # kv bytes needs to be set first to avoid race condition
        async with self.sem:
            await self.cluster.set(key_str + "kv_bytes", kv_bytes)
            await self.cluster.set(key_str + "metadata", metadata_bytes)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        await self.cluster.close()
        logger.info("Closed the redis cluster connection")

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            async with self.sem:
                if not await self.cluster.exists(key.to_string() + "metadata"):
                    return num_hit_counts
            num_hit_counts += 1
        return num_hit_counts

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # calling self.get will create a circular dependency
        results = await asyncio.gather(*(self._get(key) for key in keys))
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )
