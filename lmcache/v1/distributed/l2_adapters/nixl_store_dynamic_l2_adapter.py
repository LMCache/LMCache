# SPDX-License-Identifier: Apache-2.0
"""
Dynamic-file-mode Nixl L2 adapter.

Unlike the static ``NixlStoreL2Adapter`` which pre-allocates all storage
files at init time, this adapter opens/registers files per operation.

Atomic publish:
- Stores DMA-write to a per-operation ``<final_path>.tmp.<uuid>`` and
  atomically ``rename()`` to the final deterministic path on completion.
  This guarantees that readers (including other processes sharing the
  same directory) never observe a partially-written file.

Persist (enabled by default via ``persist_enabled``, can be opted out):
- Keeps data files on disk at shutdown (no metadata dump).

Secondary lookup (always on):
- Lookup always checks secondary storage (disk) on miss and lazily
  populates the in-memory index when a file is found. File names are
  derived deterministically from ObjectKey.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional
from abc import ABC, abstractmethod
import asyncio
import os
import threading
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig
from nixl._api import (
    nixlBind,
)

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_l2_adapter import (
    NixlStoreObj,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


class _DynamicNixlStorageBackend(ABC):
    """Abstract base class for storage backend implementations."""

    @abstractmethod
    def get_mem_type(self) -> str:
        """Return the memory type for nixl registration (e.g., "FILE" or "OBJ")."""
        pass

    @abstractmethod
    def get_new_object_context(
        self, key: ObjectKey, obj_size: int, create: bool
    ) -> dict[str, any]:
        """Get or create a device ID for the given key.
        
        Args:
            key: The object key.
            create: If True, create a new entry if it doesn't exist.
        
        Returns:
            The device ID for the key.
        """
        pass

    @abstractmethod
    def close_failed_transfer(self, key: ObjectKey, context: dict[str,str]) -> None:
        """Clean up after a failed transfer.
        
        Args:
            key: The object key that failed to transfer.
        """
        pass

    @abstractmethod
    def close_transfer(self, key: ObjectKey, context: dict[str,str]) -> None:
        """Clean up after a completed transfer.
        
        Args:
            key: The object key that was transferred.
        """
        pass

    @abstractmethod
    def delete_object(self, key: ObjectKey) -> None:
        """Delete a stored object from backend storage.
        
        Args:
            key: The object key to delete.
        """
        pass

    @abstractmethod
    def get_object_size(self, key: ObjectKey) -> int | None:
        """Get the size of a stored object.
        
        Args:
            key: The object key.
        
        Returns:
            The size in bytes, or None if the object doesn't exist.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any temporary or orphaned files in the backend."""
        pass

    def get_file_path_for_key(self, key: ObjectKey) -> str:
        """Return the full file path for a given ObjectKey."""
        return os.path.join(self.file_path, _object_key_to_filename(key))


class _DynamicNixlFileStorage(_DynamicNixlStorageBackend):
    def __init__(self, file_path: str, use_direct_io: bool):
        super().__init__()
        self.file_path = file_path
        self.use_direct_io = use_direct_io

    def _open_flags(self, create: bool) -> int:
        """Return os.open flags for storage files."""
        flags = os.O_RDWR
        if create:
            # O_TRUNC ensures any orphaned file from a previous crash
            # is truncated, avoiding stale trailing bytes on disk.
            flags |= os.O_CREAT | os.O_TRUNC
        if self.use_direct_io and hasattr(os, "O_DIRECT"):
            flags |= os.O_DIRECT
        return flags
    
    def get_mem_type(self) -> str:
        return "FILE"

    def get_new_object_context(
        self, key: ObjectKey, obj_size: int, create: bool
    ) -> dict[str, any]:
        # Implementation for file-based storage backend

        result = {}
        file_path = self.get_file_path_for_key(key)
        if create:
            tmp_path = f"{file_path}.tmp.{uuid.uuid4().hex}"
            fd = os.open(tmp_path, self._open_flags(create=True))
            result["dev_id"] = fd
            result["tmp_path"] = tmp_path
        else:
            fd = os.open(file_path, self._open_flags(create=False))
            result["dev_id"] = fd

        result["file_path"] = file_path
        result["md"] = ""

        return result
    
    def close_failed_transfer(self, key: ObjectKey, context: dict[str,str]) -> None:
        """Clean up after a failed transfer.
        
        Args:
            key: The object key that failed to transfer.
        """
        if "tmp_path" in context:
            os.unlink(context["tmp_path"])
            context["failed"] = True

    def close_transfer(self, key: ObjectKey, context: dict[str,str]) -> None:
        """Clean up after a completed transfer.
        
        Args:
            key: The object key that was transferred.
        """
        os.close(context["dev_id"])
        if "tmp_path" in context and not context.get("failed", False):
            tmp_path = context["tmp_path"]
            file_path = context["file_path"]
            os.rename(tmp_path, file_path)
    
    def delete_object(self, key: ObjectKey) -> None:
        """Delete a stored object from backend storage.
        
        Args:
            key: The object key to delete.
        """
        file_path = self.get_file_path_for_key(key)
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            logger.warning("File already deleted: %s", file_path)

    def get_object_size(self, key: ObjectKey) -> int | None:
        """Get the size of a stored object.
        
        Args:
            key: The object key.
        
        Returns:
            The size in bytes, or None if the object doesn't exist.
        """
        file_path = self.get_file_path_for_key(key)
        try:
            return os.stat(file_path).st_size
        except FileNotFoundError:
            return None

    def cleanup(self) -> None:
        """Clean up any temporary or orphaned files in the backend."""
        try:
            entries = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        for name in entries:
            # Temp suffix format: "<final_name>.tmp.<hex>"
            if ".tmp." in name:
                try:
                    os.unlink(os.path.join(self.file_path, name))
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning(
                        "Failed to remove leftover temp file %s: %s", name, e
                    )

class _DynamicNixlObjectStorage(_DynamicNixlStorageBackend):
    def __init__(self, nixl_agent: NixlAgent):
        super().__init__()
        self.obj_size = 0
        self.dev_id = 0
        self.nixl_agent = nixl_agent

    def get_mem_type(self) -> str:
        return "OBJ"

    def get_new_object_context(
        self, key: ObjectKey, obj_size: int, create: bool
    ) -> dict[str, any]:
        if self.obj_size < obj_size:
            self.obj_size = obj_size
        self.dev_id = self.dev_id + 1
        md = _object_key_to_filename(key)
        return {"dev_id": self.dev_id, "md": md}

    def close_failed_transfer(self, key: ObjectKey, context: dict[str,str]) -> None:
        """Clean up after a failed transfer."""

    def close_transfer(self, key: ObjectKey, context: dict[str,str]) -> None:
        """Clean up after a completed transfer."""
        # No additional cleanup needed for successful transfers in OBJ backend

    def delete_object(self, key: ObjectKey) -> None:
        """Delete a stored object from backend storage."""
        logger.warning("Delete operation is not supported for OBJ backend.")

    def get_object_size(self, key: ObjectKey) -> int | None:
        """Get the size of a stored object."""
        object_name = _object_key_to_filename(key)
        reg_list = [(0, 0, 0, object_name)]
        resp = self.nixl_agent.query_memory(reg_list, "OBJ", mem_type="OBJ")

        if resp and resp[0] is not None:
            return self.obj_size
        return None

    def cleanup(self) -> None:
        """Clean up any temporary or orphaned files in the backend."""
        # No cleanup needed for OBJ backend
        pass


# ---------------------------------------------------------------
# ObjectKey <-> file path helpers
# ---------------------------------------------------------------


def _object_key_to_filename(key: ObjectKey) -> str:
    """Derive a deterministic file name from an ObjectKey.

    Replaces ``/`` in model names with ``--`` to avoid creating
    subdirectories (e.g. ``meta-llama/Llama-3-8B`` becomes
    ``meta-llama--Llama-3-8B``).
    """
    safe_model_name = key.model_name.replace("/", "--")
    chunk_hex = key.chunk_hash.hex()
    return (
        f"{safe_model_name}_{key.kv_rank:08x}_{key.object_group_id:x}_{chunk_hex}.bin"
    )


# ---------------------------------------------------------------
# Dynamic Nixl storage agent
# ---------------------------------------------------------------


class DynamicNixlStorageAgent:
    """Nixl storage agent that opens/registers files per operation.

    The L1 memory handler is registered once at init (same as the static
    agent).  Storage files are registered on-demand for each store/load
    and deregistered immediately after the transfer completes.
    """

    def __init__(
        self,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        l1_memory_desc: L1MemoryDesc,
    ):
        self.backend = backend
        self.device = device
        self.backend_params = backend_params
        self.l1_align_bytes = l1_memory_desc.align_bytes
        if self.backend != "OBJ":
            self.file_path = backend_params["file_path"]
            os.makedirs(self.file_path, exist_ok=True)
            self.use_direct_io = (
                str(backend_params.get("use_direct_io", "false")).lower() == "true"
            )

        self.agent_name = "DynNixlAgent_" + str(uuid.uuid4())
        nixl_conf = NixlAgentConfig(backends=[])
        self.nixl_agent = NixlAgent(self.agent_name, nixl_conf)
        self.nixl_agent.create_backend(backend, backend_params)

        # Register L1 memory (same as static agent)
        self._init_mem_handlers(
            device,
            l1_memory_desc.ptr,
            l1_memory_desc.size,
            l1_memory_desc.align_bytes,
            device_id=0,
        )

        if backend == "OBJ":
            self.storage_backend = _DynamicNixlObjectStorage(
                self.nixl_agent
            )
        else:
            self.storage_backend = _DynamicNixlFileStorage(
                self.file_path, self.use_direct_io
            )


    # ---- L1 memory registration (one-time) ----

    def _init_mem_handlers(self, device, buffer_ptr, buffer_size, page_size, device_id):
        reg_list = [(buffer_ptr, buffer_size, device_id, "")]
        xfer_desc = [
            (base_addr, page_size, device_id)
            for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size)
        ]

        mem_type = "DRAM" if device == "cpu" else "VRAM"

        self.mem_reg_descs = self.nixl_agent.register_memory(
            reg_list, mem_type=mem_type
        )
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        self.mem_xfer_handler = self.nixl_agent.prep_xfer_dlist(
            "", xfer_descs, mem_type=mem_type
        )

    def _register_single_object(self, dev_id: int, meta_data: str, file_size: int, page_size: int):
        """Register a single file with nixl and return (reg_descs, xfer_handler).

        Returns:
            Tuple of (reg_descs, xfer_handler) for later cleanup.
        """
        num_pages = file_size // page_size

        reg_list = [(0, file_size, dev_id, meta_data)]
        xfer_desc = [
            (offset * page_size, page_size, dev_id) for offset in range(num_pages)
        ]

        mem_type = self.storage_backend.get_mem_type()
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type=mem_type)
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type=mem_type
        )
        return reg_descs, xfer_handler

    def _deregister_object(self, reg_descs, xfer_handler):
        """Deregister a file from nixl."""
        self.nixl_agent.release_dlist_handle(xfer_handler)
        self.nixl_agent.deregister_memory(reg_descs)

    async def dynamic_store_object(
        self,
        mem_indices: list[int],
        key: ObjectKey,
        page_size: int,
    ) -> None:
        """Write-to-temp-then-rename to publish the final file atomically.

        The DMA write goes to ``<file_path>.tmp.<uuid>`` in the same
        directory. Only after the transfer completes successfully is the
        temp file atomically renamed to the final path, ensuring that
        concurrent readers (including other processes sharing the same
        directory) never observe a partially-written file.
        """
        file_size = len(mem_indices) * page_size

        obj_context = self.storage_backend.get_new_object_context(key, file_size, create=True)
        # tmp_path = f"{file_path}.tmp.{uuid.uuid4().hex}"
        # fd = os.open(tmp_path, self._open_flags(create=True))
        try:
            reg_descs, xfer_handler = self._register_single_object(
                obj_context["dev_id"], obj_context["md"], file_size, page_size
            )
            try:
                storage_indices = list(range(len(mem_indices)))
                handle = self.nixl_agent.make_prepped_xfer(
                    "WRITE",
                    self.mem_xfer_handler,
                    mem_indices,
                    xfer_handler,
                    storage_indices,
                )
                await self._post_non_blocking(handle)
                self.nixl_agent.release_xfer_handle(handle)
            finally:
                self._deregister_object(reg_descs, xfer_handler)
        except BaseException:
            # Best-effort cleanup of the temp file on failure.
            try:
                self.storage_backend.close_failed_transfer(key, obj_context)
                # os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        finally:
            self.storage_backend.close_transfer(key, obj_context)

        # Atomic publish: readers only ever see a complete file at file_path.
        # TODO(Jiayi): Only guaranteed to be atomic within the local posix filesystems.
        # os.rename(tmp_path, file_path)

    async def dynamic_load_object(
        self,
        mem_indices: list[int],
        key: ObjectKey,
        page_size: int,
    ) -> None:
        """Open an existing file, DMA read into L1 memory, then clean up."""
        file_size = len(mem_indices) * page_size
        obj_context = self.storage_backend.get_new_object_context(key, file_size, create=False)
        # fd = os.open(file_path, self._open_flags(create=False))
        try:
            reg_descs, xfer_handler = self._register_single_object(
                obj_context["dev_id"], obj_context["md"], file_size, page_size
            )
            try:
                storage_indices = list(range(len(mem_indices)))
                handle = self.nixl_agent.make_prepped_xfer(
                    "READ",
                    self.mem_xfer_handler,
                    mem_indices,
                    xfer_handler,
                    storage_indices,
                )
                await self._post_non_blocking(handle)
                self.nixl_agent.release_xfer_handle(handle)
            finally:
                self._deregister_object(reg_descs, xfer_handler)
        finally:
            self.storage_backend.close_transfer(key, obj_context)

    def dynamic_delete_object(self, key: ObjectKey) -> None:
        """Delete a storage file from disk."""
        # try:
        self.storage_backend.delete_object(key)
            # os.unlink(file_path)
        # except FileNotFoundError:
        #     logger.warning("File already deleted: %s", file_path)

    # ---- Shared helpers ----

    def get_memory_indices(self, raw_addr: int, mem_size: int) -> list[int]:
        """Get L1 memory page indices for the given address and size."""
        if raw_addr % self.l1_align_bytes != 0:
            raise ValueError(
                f"Raw address {raw_addr} is not aligned to "
                f"page size {self.l1_align_bytes}"
            )
        if mem_size % self.l1_align_bytes != 0:
            raise ValueError(
                f"Memory size {mem_size} is not a multiple of "
                f"page size {self.l1_align_bytes}"
            )
        num_pages = mem_size // self.l1_align_bytes
        return [(raw_addr // self.l1_align_bytes + i) for i in range(num_pages)]

    async def _post_non_blocking(self, handle):
        """Await a nixl transfer until done."""
        state = self.nixl_agent.transfer(handle)
        while state != "DONE" and state != "ERR":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError:
                raise
            await asyncio.sleep(0.01)
        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")

    def cleanup_temp_files(self) -> None:
        """Remove leftover ``*.tmp.*`` files in the storage directory.

        These can be left behind if a store crashed between opening the
        temp file and the atomic rename. Called at shutdown as a best-effort
        GC; orphans don't affect correctness because they're never matched
        by the deterministic ``ObjectKey → filename`` mapping.
        """
        # try:
        #     entries = os.listdir(self.file_path)
        # except FileNotFoundError:
        #     return
        # for name in entries:
        #     # Temp suffix format: "<final_name>.tmp.<hex>"
        #     if ".tmp." in name:
        #         try:
        #             os.unlink(os.path.join(self.file_path, name))
        #         except FileNotFoundError:
        #             pass
        #         except OSError as e:
        #             logger.warning(
        #                 "Failed to remove leftover temp file %s: %s", name, e
        #             )
        self.storage_backend.cleanup()

    def close(self):
        """Release L1 memory handlers."""
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)


# ---------------------------------------------------------------
# Dynamic L2 adapter
# ---------------------------------------------------------------


class DynamicNixlStoreL2Adapter(L2AdapterInterface):
    """Nixl L2 adapter using dynamic per-operation file registration.

    Each store creates a new file on disk; each load re-opens the file.

    When ``persist_enabled`` is True (the default), data files are kept
    on disk at shutdown.  Lookup always checks secondary storage (disk)
    for keys not in the in-memory index and populates the index lazily.
    """

    def __init__(
        self,
        config: DynamicNixlStoreL2AdapterConfig,
        l1_memory_desc: L1MemoryDesc,
    ):
        max_capacity_gb = float(config.backend_params.get("max_capacity_gb", 0))
        if max_capacity_gb <= 0:
            raise ValueError("backend_params must include a positive 'max_capacity_gb'")
        super().__init__(max_capacity_bytes=int(max_capacity_gb * (1024**3)))
        self._config = config

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Cache data structures
        self._memory_objects: dict[ObjectKey, NixlStoreObj] = {}
        self._inflight_stores: set[ObjectKey] = set()
        self._total_bytes: int = 0

        # Task ID management
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()

        # Asyncio event loop running in a background thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # Initialize dynamic Nixl agent (L1 memory only, no pre-allocated files)
        self.nixl_agent = DynamicNixlStorageAgent(
            device="cpu",
            backend=config.backend,
            backend_params=config.backend_params,
            l1_memory_desc=l1_memory_desc,
        )

        self._persist_enabled = config.persist_config.persist_enabled

    # --------------------
    # Event Fd Interface
    # --------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    #####################
    # Store Interface
    #####################

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_store_in_the_loop(keys, objects, task_id), self._loop
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    #####################
    # Lookup and Lock Interface
    #####################

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        self._loop.call_soon_threadsafe(self._execute_lookup_in_the_loop, keys, task_id)
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        def _unlock_keys(keys: list[ObjectKey]) -> None:
            for key in keys:
                if (obj := self._memory_objects.get(key)) is not None:
                    obj.decrease_pin_count()

        self._loop.call_soon_threadsafe(_unlock_keys, keys)

    #####################
    # Load Interface
    #####################

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_load_in_loop(keys, objects, task_id), self._loop
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    #####################
    # Eviction Interface
    #####################

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete objects from storage, removing their files from disk."""
        to_delete: list[tuple[ObjectKey, int]] = []
        with self._lock:
            for key in keys:
                obj = self._memory_objects.get(key)
                if obj is None:
                    continue
                if obj.pin_count > 0:
                    logger.debug(
                        "Skipping eviction of pinned key %s (pin_count=%d)",
                        key,
                        obj.pin_count,
                    )
                    continue
                self._total_bytes -= obj.size
                del self._memory_objects[key]
                to_delete.append(
                    (key, obj.size)
                )
        # Filesystem I/O outside the lock to avoid blocking concurrent
        # store/lookup/load operations.
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for key, size in to_delete:
            self.nixl_agent.dynamic_delete_object(key)
            deleted_keys.append(key)
            deleted_sizes.append(size)
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage`` is inherited from L2AdapterInterface; byte accounting
    # is driven by ``_notify_keys_*`` through the base class now.

    #####################
    # Status Interface
    #####################

    def report_status(self) -> dict:
        with self._lock:
            stored_object_count = len(self._memory_objects)
            pinned_object_count = sum(
                1 for obj in self._memory_objects.values() if obj.pin_count > 0
            )
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "DynamicNixlStoreL2Adapter",
            "backend": self._config.backend,
            "stored_object_count": stored_object_count,
            "pinned_object_count": pinned_object_count,
            "event_loop_alive": self._loop_thread.is_alive(),
        }

    #####################
    # Cleanup Interface
    #####################

    def close(self):
        # Stop the event loop and wait for all in-flight tasks to finish
        async def _stop_tasks():
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop)
            future.result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join()
        self._loop.close()

        # If persist is enabled, keep data files on disk; otherwise clean up.
        if self._persist_enabled:
            logger.info("persist_enabled=True, keeping data files on disk")
        else:
            logger.info("persist_enabled=False, deleting all data files")
            with self._lock:
                for key in list(self._memory_objects.keys()):
                    self.nixl_agent.dynamic_delete_object(key)

        # Best-effort cleanup of orphaned temp files from crashed stores.
        self.nixl_agent.cleanup_temp_files()

        self.nixl_agent.close()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    ##################
    # Helper functions
    ##################

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_next_task_id(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _signal_store_event(self) -> None:
        self._store_efd.notify()

    def _signal_lookup_event(self) -> None:
        self._lookup_efd.notify()

    def _signal_load_event(self) -> None:
        self._load_efd.notify()

    async def _execute_store_in_the_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Store each key-object pair to its own file via dynamic DMA write."""
        success = True
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        try:
            for key, obj in zip(keys, objects, strict=False):
                mem_addr = obj.meta.address
                mem_size = obj.meta.phy_size

                # Reserve the key and capacity under the lock *before*
                # the DMA write so that concurrent coroutines (other
                # stores, secondary lookups) see the reservation.
                with self._lock:
                    if key in self._memory_objects or key in self._inflight_stores:
                        continue
                    if self._total_bytes + mem_size > self._max_capacity_bytes:
                        logger.warning(
                            "Storage capacity exceeded, skipping store for key %s",
                            key,
                        )
                        break
                    self._inflight_stores.add(key)
                    self._total_bytes += mem_size

                try:
                    mem_indices = self.nixl_agent.get_memory_indices(mem_addr, mem_size)

                    await self.nixl_agent.dynamic_store_object(
                        mem_indices, key, self.nixl_agent.l1_align_bytes
                    )

                    store_obj = NixlStoreObj(
                        page_indices=[],  # not used in dynamic mode
                        size=mem_size,
                        layout=MemoryLayoutDesc(
                            [obj.meta.shape],
                            [obj.meta.dtype],
                        ),
                        pin_count=1,
                    )
                    with self._lock:
                        self._inflight_stores.discard(key)
                        self._memory_objects[key] = store_obj
                        store_obj.decrease_pin_count()
                    stored_keys.append(key)
                    stored_sizes.append(mem_size)
                except Exception:
                    # Un-reserve on failure so capacity accounting
                    # stays correct.
                    with self._lock:
                        self._inflight_stores.discard(key)
                        self._total_bytes -= mem_size
                    raise

        except Exception:
            logger.exception("Dynamic NIXL store task %d failed", task_id)
            success = False

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)

        bytes_transferred = sum(stored_sizes)
        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(
                success, bytes_transferred
            )
        self._signal_store_event()

    def _execute_lookup_in_the_loop(
        self, keys: list[ObjectKey], task_id: L2TaskId
    ) -> None:
        """Look up keys and pin found objects.

        Also checks secondary storage (disk) for keys not in the
        in-memory index and lazily populates ``_memory_objects`` for any
        data files found on disk.
        """
        bitmap = Bitmap(len(keys))
        # Keys populated by secondary lookup need a ``_notify_keys_stored``
        # so the base class accounting stays in sync with disk state.
        recovered_keys: list[ObjectKey] = []
        recovered_sizes: list[int] = []
        with self._lock:
            for i, key in enumerate(keys):
                obj = self._memory_objects.get(key)
                if obj is None:
                    obj = self._secondary_lookup_locked(key)
                    if obj is not None:
                        recovered_keys.append(key)
                        recovered_sizes.append(obj.size)
                if obj is None:
                    continue
                bitmap.set(i)
                obj.increase_pin_count()
            self._completed_lookup_tasks[task_id] = bitmap
        if recovered_keys:
            self._notify_keys_stored(recovered_keys, recovered_sizes)
        self._signal_lookup_event()

    def _secondary_lookup_locked(self, key: ObjectKey) -> NixlStoreObj | None:
        """Check if a data file for ``key`` exists on disk; if so, populate
        ``_memory_objects`` and return the entry. Caller must hold ``_lock``.

        The file size is read via ``os.stat``. Layout is left as ``None`` and
        will be supplied by the caller's MemoryObj at load time.
        """
        # Skip keys with an in-flight store to avoid double-counting
        # in _total_bytes.
        if key in self._inflight_stores:
            return None
        obj_size = self.nixl_agent.storage_backend.get_object_size(key)

        if obj_size is None:
            return None
        
        # file_path = self.nixl_agent.get_file_path_for_key(key)
        # try:
        #     obj_size = os.stat(file_path).st_size
        # except FileNotFoundError:
        #     return None

        # Enforce capacity when populating lazily too.
        if self._total_bytes + obj_size > self._max_capacity_bytes:
            logger.debug(
                "Secondary lookup hit for %s but capacity exceeded, skipping",
                key,
            )
            return None

        obj = NixlStoreObj(
            page_indices=[],  # not used in dynamic mode
            size=obj_size,
            layout=None,
        )
        self._memory_objects[key] = obj
        self._total_bytes += obj_size
        return obj

    async def _execute_load_in_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Execute a queued load task for ``task_id``.

        For each requested key present in this adapter, read its stored
        data into the caller-provided ``objects[i]`` and set bit ``i`` of
        the result bitmap; keys that are missing or fail to load are left
        unset. The bitmap is recorded under ``task_id`` (retrieve via
        ``query_load_result``) and the load event fd is signaled.
        """
        bitmap = Bitmap(len(keys))
        accessed_keys: list[ObjectKey] = []
        try:
            # Read every present key's file concurrently (asyncio.gather
            # below) so a multi-chunk request does not pay per-file NIXL
            # latency serially -- the dominant cost of a single-request load.
            #
            # TODO(perf): batch a request's files into one NIXL
            # ``register_memory`` (a single transfer + single deregister),
            # like the static adapter's pre-registered flat-index pool,
            # instead of the current per-file register/transfer/deregister.
            # Trade-off: one large per-request registration list vs. many
            # small ones.
            coros = []
            found_positions: list[int] = []
            for i, key in enumerate(keys):
                with self._lock:
                    storage_obj = self._memory_objects.get(key)
                if storage_obj is None:
                    continue

                mem_addr = objects[i].meta.address
                mem_size = objects[i].meta.phy_size
                mem_indices = self.nixl_agent.get_memory_indices(mem_addr, mem_size)

                coros.append(
                    self.nixl_agent.dynamic_load_object(
                        mem_indices, key, self.nixl_agent.l1_align_bytes
                    )
                )
                found_positions.append(i)

            if coros:
                # return_exceptions=True so one chunk's failure doesn't
                # discard the chunks that loaded successfully: mark each
                # key by its own result rather than all-or-nothing.
                results = await asyncio.gather(*coros, return_exceptions=True)
                for pos, result in zip(found_positions, results, strict=True):
                    if isinstance(result, BaseException):
                        logger.error(
                            "Dynamic NIXL load failed for key %s: %r",
                            keys[pos],
                            result,
                        )
                        continue
                    bitmap.set(pos)
                    accessed_keys.append(keys[pos])

        except Exception:
            logger.exception("Dynamic NIXL load task %d failed", task_id)

        if accessed_keys:
            self._notify_keys_accessed(accessed_keys)
        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._signal_load_event()


# ---------------------------------------------------------------------
# Config and self-registration
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Config and self-registration
# ---------------------------------------------------------------------

# TODO: should detect which backend is selected and allow based on memory types, not backend names
_FILE_DYNAMIC_BACKENDS = ("GDS", "GDS_MT", "POSIX", "HF3FS")
_VALID_DYNAMIC_BACKENDS = (*_FILE_DYNAMIC_BACKENDS, "OBJ")


class DynamicNixlStoreL2AdapterConfig(L2AdapterConfigBase):
    """Configure dynamic NIXL FILE_SEG or OBJ_SEG storage.

    Fields:
    - backend: NIXL backend (GDS, GDS_MT, POSIX, HF3FS, or OBJ).
    - backend_params: Backend-specific string key-value pairs. FILE backends
      require ``file_path`` and ``use_direct_io``.
    - persist_enabled: Must be true for OBJ. NIXL OBJ deletion is unsupported,
      so objects require external bucket lifecycle policies or object-store
      tooling.
    """

    def __init__(
        self,
        backend: str,
        backend_params: dict[str, str],
    ) -> None:
        if backend not in _VALID_DYNAMIC_BACKENDS:
            raise ValueError(
                "backend must be one of %s, got %r" % (_VALID_DYNAMIC_BACKENDS, backend)
            )
        if backend in _FILE_DYNAMIC_BACKENDS and "file_path" not in backend_params:
            raise ValueError(
                "backend_params must include 'file_path' for backend %r" % backend
            )
        if backend in _FILE_DYNAMIC_BACKENDS and "use_direct_io" not in backend_params:
            raise ValueError(
                "backend_params must include 'use_direct_io' for backend %r" % backend
            )
        self.backend = backend
        self.backend_params = backend_params

    @classmethod
    def from_dict(cls, d: dict) -> DynamicNixlStoreL2AdapterConfig:
        backend = d.get("backend")
        if backend not in _VALID_DYNAMIC_BACKENDS:
            raise ValueError(
                "backend must be one of %s, got %r" % (_VALID_DYNAMIC_BACKENDS, backend)
            )

        backend_params = d.get("backend_params", {})
        if not isinstance(backend_params, dict):
            raise ValueError("backend_params must be a dict of string key-value pairs")
        if backend == "OBJ" and not bool(d.get("persist_enabled", True)):
            raise ValueError(
                "Dynamic NIXL OBJ requires persist_enabled=True"
            )

        return cls(backend=backend, backend_params=backend_params)

    @classmethod
    def help(cls) -> str:
        return (
            "Dynamic Nixl store L2 adapter config fields:\n"
            "- backend (str): Nixl storage backend, "
            "one of %s (required)\n"
            "- backend_params (dict): backend-specific "
            "string key-value pairs. FILE backends require "
            "'file_path' and 'use_direct_io'.\n"
            "- persist_enabled (bool): keep data in storage at shutdown "
            "(optional, default True). Must be True for OBJ backend\n"
            "Lookup always checks the selected secondary storage on miss."
            % (_VALID_DYNAMIC_BACKENDS,)
        )


# Self-register config type and adapter factory
register_l2_adapter_type("nixl_store_dynamic", DynamicNixlStoreL2AdapterConfig)


def _create_dynamic_nixl_store_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create a DynamicNixlStoreL2Adapter from config."""
    if l1_memory_desc is None:
        raise ValueError(
            "l1_memory_desc is required to create a DynamicNixlStoreL2Adapter."
        )
    return DynamicNixlStoreL2Adapter(config, l1_memory_desc)  # type: ignore[arg-type]


register_l2_adapter_factory("nixl_store_dynamic", _create_dynamic_nixl_store_adapter)
