# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List, Optional
import asyncio
import os
import subprocess

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


@dataclass
class ShmFileConfig:
    """Configuration for the shared memory file connector."""

    storage_dir: str
    shm_name: str
    worker_binary: str

    @staticmethod
    def from_path_and_shm(
        storage_dir: str,
        shm_name: str,
        worker_binary: Optional[str] = None,
    ) -> "ShmFileConfig":
        if worker_binary is None:
            worker_binary = os.environ.get("SHM_FILE_WORKER_BIN", "shm_file_worker")
        return ShmFileConfig(
            storage_dir=storage_dir,
            shm_name=shm_name,
            worker_binary=worker_binary,
        )


class ShmFileConnector(RemoteConnector):
    """Connector that uses a C++ subprocess to read/write
    files via POSIX shared memory.

    The connector allocates a shm-backed pinned buffer via
    MixedMemoryAllocator(shm_name=...) and communicates with
    a child process (shm_file_worker) that opens the same shm
    region and performs file I/O directly into it -- avoiding
    an extra memory copy.
    """

    def __init__(
        self,
        storage_dir: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig] = None,
        shm_name: Optional[str] = None,
        worker_binary: Optional[str] = None,
    ):
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        # Resolve shm_name from allocator
        allocator = self.local_cpu_backend.memory_allocator
        self.shm_name = shm_name or getattr(allocator, "shm_name", None)
        if not self.shm_name:
            raise ValueError(
                "ShmFileConnector requires a shm_name. "
                "Pass shm_name= or use "
                "MixedMemoryAllocator(shm_name=...)."
            )

        self.shm_size = getattr(allocator, "size", 0)
        if self.shm_size <= 0:
            raise ValueError("ShmFileConnector requires allocator.size > 0")

        # Cache base address for offset calculation in worker
        base_ptr = getattr(allocator, "buffer", None)
        if base_ptr is None:
            raise ValueError("ShmFileConnector requires allocator.buffer")
        self.base_addr = base_ptr.data_ptr()

        # Resolve worker binary
        if worker_binary is None:
            worker_binary = os.environ.get("SHM_FILE_WORKER_BIN", "shm_file_worker")
        self.worker_binary = worker_binary

        # Start subprocess
        self._proc = subprocess.Popen(
            [self.worker_binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Attach to the shared memory with base_addr
        resp = self._send_cmd(
            "ATTACH %s %d %d",
            self.shm_name,
            self.shm_size,
            self.base_addr,
        )
        if not resp.startswith("OK"):
            raise RuntimeError("shm_file_worker ATTACH failed: %s" % resp)

        logger.info(
            "ShmFileConnector: worker attached to shm=%s size=%d base_addr=%d",
            self.shm_name,
            self.shm_size,
            self.base_addr,
        )

    # -- Low-level subprocess communication -----------------------

    def _send_cmd(self, fmt: str, *args) -> str:
        """Send a command to the worker and return the response."""
        cmd = (fmt % args) + "\n"
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(cmd)
        self._proc.stdin.flush()
        return self._proc.stdout.readline().strip()

    def _file_path(self, key: CacheEngineKey) -> str:
        return os.path.join(self.storage_dir, key.to_string() + ".data")

    # -- Connector interface --------------------------------------

    async def exists(self, key: CacheEngineKey) -> bool:
        path = self._file_path(key)
        return await asyncio.to_thread(os.path.isfile, path)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return os.path.isfile(self._file_path(key))

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Write memory_obj data to a file via the worker."""
        tensor = memory_obj.tensor
        assert tensor is not None
        buf_ptr = tensor.data_ptr()
        buf_size = tensor.numel() * tensor.element_size()

        path = self._file_path(key)
        resp = await asyncio.to_thread(
            self._send_cmd,
            "WRITE %s %d %d",
            path,
            buf_ptr,
            buf_size,
        )
        if not resp.startswith("OK"):
            raise RuntimeError("shm_file_worker WRITE failed: %s" % resp)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Read a file into a newly allocated memory_obj
        via the worker subprocess."""
        path = self._file_path(key)
        if not os.path.isfile(path):
            return None

        if not self.meta_shapes or not self.meta_dtypes or not self.meta_fmt:
            logger.error("Metadata not available for get")
            return None

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes, self.meta_dtypes, self.meta_fmt
        )
        if memory_obj is None:
            return None

        tensor = memory_obj.tensor
        if tensor is None:
            memory_obj.ref_count_down()
            return None

        buf_ptr = tensor.data_ptr()
        buf_size = tensor.numel() * tensor.element_size()

        resp = await asyncio.to_thread(
            self._send_cmd,
            "READ %s %d %d",
            path,
            buf_ptr,
            buf_size,
        )
        if not resp.startswith("OK"):
            memory_obj.ref_count_down()
            logger.error("shm_file_worker READ failed: %s", resp)
            return None

        parts = resp.split()
        bytes_read = int(parts[1]) if len(parts) > 1 else 0
        if bytes_read <= 0:
            memory_obj.ref_count_down()
            return None

        try:
            return self.reshape_partial_chunk(memory_obj, bytes_read)
        except Exception:
            logger.error(
                "reshape_partial_chunk failed for key %s",
                key,
            )
            memory_obj.ref_count_down()
            return None

    async def list(self) -> List[str]:
        files = await asyncio.to_thread(os.listdir, self.storage_dir)
        return [f.replace(".data", "") for f in files if f.endswith(".data")]

    async def close(self):
        try:
            self._send_cmd("QUIT")
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        logger.info("ShmFileConnector closed")
