# SPDX-License-Identifier: Apache-2.0
"""
Dynamic-file-mode Nixl L2 adapter.

Unlike the static ``NixlStoreL2Adapter`` which pre-allocates all storage
files at init time, this adapter opens/registers files per operation and
supports persist/recover of cached KV metadata across restarts.

Another alternative is that we can still pre-allocates and register
all files at start time.
"""

# Future
from __future__ import annotations

# Standard
from typing import Optional
import asyncio
import json
import os
import threading
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig
from nixl._api import (
    nixlBind,
)
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    PersistConfig,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_l2_adapter import (
    NixlStoreObj,
)
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


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
    return f"{safe_model_name}_{key.kv_rank:08x}_{chunk_hex}.bin"


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
        self.file_path = backend_params["file_path"]
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

    # ---- Per-operation file helpers ----

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

    def _register_single_file(self, fd: int, file_size: int, page_size: int):
        """Register a single file with nixl and return (reg_descs, xfer_handler).

        Returns:
            Tuple of (reg_descs, xfer_handler) for later cleanup.
        """
        num_pages = file_size // page_size

        reg_list = [(0, file_size, fd, "")]
        xfer_desc = [(offset * page_size, page_size, fd) for offset in range(num_pages)]

        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="FILE")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="FILE")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type="FILE"
        )
        return reg_descs, xfer_handler

    def _deregister_file(self, reg_descs, xfer_handler):
        """Deregister a file from nixl."""
        self.nixl_agent.release_dlist_handle(xfer_handler)
        self.nixl_agent.deregister_memory(reg_descs)

    async def dynamic_store_file(
        self,
        mem_indices: list[int],
        file_path: str,
        page_size: int,
    ) -> None:
        """Create a file, DMA write from L1 memory, then clean up nixl state."""
        file_size = len(mem_indices) * page_size
        fd = os.open(file_path, self._open_flags(create=True))
        try:
            reg_descs, xfer_handler = self._register_single_file(
                fd, file_size, page_size
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
                self._deregister_file(reg_descs, xfer_handler)
        finally:
            os.close(fd)

    async def dynamic_load_file(
        self,
        mem_indices: list[int],
        file_path: str,
        page_size: int,
    ) -> None:
        """Open an existing file, DMA read into L1 memory, then clean up."""
        file_size = len(mem_indices) * page_size
        fd = os.open(file_path, self._open_flags(create=False))
        try:
            reg_descs, xfer_handler = self._register_single_file(
                fd, file_size, page_size
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
                self._deregister_file(reg_descs, xfer_handler)
        finally:
            os.close(fd)

    def dynamic_delete_file(self, file_path: str) -> None:
        """Delete a storage file from disk."""
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            logger.warning("File already deleted: %s", file_path)

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

    def get_file_path_for_key(self, key: ObjectKey) -> str:
        """Return the full file path for a given ObjectKey."""
        return os.path.join(self.file_path, _object_key_to_filename(key))

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
    Supports persist/recover to preserve cached KV metadata across restarts.
    """

    def __init__(
        self,
        config: DynamicNixlStoreL2AdapterConfig,
        l1_memory_desc: L1MemoryDesc,
    ):
        super().__init__()
        self._config = config

        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # Cache data structures
        self._memory_objects: dict[ObjectKey, NixlStoreObj] = {}
        self._total_bytes: int = 0
        max_capacity_gb = float(config.backend_params.get("max_capacity_gb", 0))
        if max_capacity_gb <= 0:
            raise ValueError("backend_params must include a positive 'max_capacity_gb'")
        self._max_capacity_bytes: int = int(max_capacity_gb * (1024**3))

        # Task ID management
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, bool] = {}
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

        # Recover if configured
        if config.persist_config and config.persist_config.recover_path:
            self.recover(config.persist_config)

    # --------------------
    # Event Fd Interface
    # --------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        return self._load_efd

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

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    #####################
    # Lookup and Lock Interface
    #####################

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
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
        deleted_keys: list[ObjectKey] = []
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
                file_path = self.nixl_agent.get_file_path_for_key(key)
                self.nixl_agent.dynamic_delete_file(file_path)
                deleted_keys.append(key)
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys)

    def get_usage(self) -> tuple[float, float]:
        """Return (current_usage, usage_after_ongoing_eviction) in [0, 1]."""
        with self._lock:
            usage = self._total_bytes / self._max_capacity_bytes
        return (usage, usage)

    #####################
    # Persist/Recover Interface
    #####################

    def persist(self, config: PersistConfig) -> bool:
        """Persist the ObjectKey -> metadata mapping to disk as JSON."""
        if config.persist_path is None:
            logger.info("persist_path is None, skipping persist")
            return False

        entries = []
        with self._lock:
            for key, obj in self._memory_objects.items():
                entry = {
                    "chunk_hash": key.chunk_hash.hex(),
                    "model_name": key.model_name,
                    "kv_rank": key.kv_rank,
                    "size": obj.size,
                }
                if obj.layout is not None:
                    entry["layout"] = {
                        "shapes": [list(s) for s in obj.layout.shapes],
                        "dtypes": [str(d) for d in obj.layout.dtypes],
                    }
                entries.append(entry)

        persist_dir = os.path.dirname(config.persist_path)
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
        with open(config.persist_path, "w") as f:
            json.dump(entries, f)

        logger.info(
            "Persisted %d object metadata entries to %s",
            len(entries),
            config.persist_path,
        )
        return True

    def recover(self, config: PersistConfig) -> bool:
        """Recover the ObjectKey -> metadata mapping from a persisted JSON file."""
        if config.recover_path is None:
            logger.warning("recover_path is None, skipping recover")
            return False

        if not os.path.exists(config.recover_path):
            logger.warning(
                "Recover path %s does not exist, skipping recover",
                config.recover_path,
            )
            return False

        with open(config.recover_path, "r") as f:
            entries = json.load(f)

        recovered = 0
        skipped = 0
        for entry in entries:
            key = ObjectKey(
                chunk_hash=bytes.fromhex(entry["chunk_hash"]),
                model_name=entry["model_name"],
                kv_rank=entry["kv_rank"],
            )
            # Verify the data file still exists on disk
            file_path = self.nixl_agent.get_file_path_for_key(key)
            if not os.path.exists(file_path):
                logger.warning(
                    "Data file missing for key %s: %s, skipping", key, file_path
                )
                skipped += 1
                continue

            layout = None
            if "layout" in entry:
                shapes = [torch.Size(s) for s in entry["layout"]["shapes"]]
                dtypes_str = entry["layout"]["dtypes"]
                dtypes = [getattr(torch, d.replace("torch.", "")) for d in dtypes_str]
                layout = MemoryLayoutDesc(shapes, dtypes)

            obj_size = entry["size"]
            with self._lock:
                self._memory_objects[key] = NixlStoreObj(
                    page_indices=[],  # not used in dynamic mode
                    size=obj_size,
                    layout=layout,
                )
                self._total_bytes += obj_size
            recovered += 1

        logger.info(
            "Recovered %d objects (%d skipped) from %s",
            recovered,
            skipped,
            config.recover_path,
        )
        return True

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

        # Persist or clean up data files after all tasks are done
        if self._config.persist_config and self._config.persist_config.persist_path:
            logger.info(
                "Persisting metadata to %s before shutdown",
                self._config.persist_config.persist_path,
            )
            self.persist(self._config.persist_config)
        else:
            logger.info("No persist_path configured, deleting all data files")
            with self._lock:
                for key in list(self._memory_objects.keys()):
                    file_path = self.nixl_agent.get_file_path_for_key(key)
                    self.nixl_agent.dynamic_delete_file(file_path)

        self.nixl_agent.close()

        os.close(self._store_efd)
        os.close(self._lookup_efd)
        os.close(self._load_efd)

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
        os.eventfd_write(self._store_efd, 1)

    def _signal_lookup_event(self) -> None:
        os.eventfd_write(self._lookup_efd, 1)

    def _signal_load_event(self) -> None:
        os.eventfd_write(self._load_efd, 1)

    async def _execute_store_in_the_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Store each key-object pair to its own file via dynamic DMA write."""
        success = True
        stored_keys: list[ObjectKey] = []
        try:
            for key, obj in zip(keys, objects, strict=False):
                mem_addr = obj.meta.address
                mem_size = obj.meta.phy_size

                # Skip if key already exists or capacity exceeded
                with self._lock:
                    if key in self._memory_objects:
                        continue
                    if self._total_bytes + mem_size > self._max_capacity_bytes:
                        logger.warning(
                            "Storage capacity exceeded, skipping store for key %s",
                            key,
                        )
                        break

                mem_indices = self.nixl_agent.get_memory_indices(mem_addr, mem_size)
                file_path = self.nixl_agent.get_file_path_for_key(key)

                await self.nixl_agent.dynamic_store_file(
                    mem_indices, file_path, self.nixl_agent.l1_align_bytes
                )

                store_obj = NixlStoreObj(
                    page_indices=[],  # not used in dynamic mode
                    size=obj.meta.phy_size,
                    layout=MemoryLayoutDesc(
                        [obj.meta.shape],
                        [obj.meta.dtype],
                    ),
                    pin_count=1,
                )
                with self._lock:
                    self._memory_objects[key] = store_obj
                    self._total_bytes += store_obj.size
                    store_obj.decrease_pin_count()
                stored_keys.append(key)

        except Exception:
            logger.exception("Dynamic NIXL store task %d failed", task_id)
            success = False

        if stored_keys:
            self._notify_keys_stored(stored_keys)

        with self._lock:
            self._completed_store_tasks[task_id] = success
        self._signal_store_event()

    def _execute_lookup_in_the_loop(
        self, keys: list[ObjectKey], task_id: L2TaskId
    ) -> None:
        """Look up keys and pin found objects."""
        bitmap = Bitmap(len(keys))
        with self._lock:
            for i, key in enumerate(keys):
                if (obj := self._memory_objects.get(key)) is None:
                    continue
                bitmap.set(i)
                obj.increase_pin_count()
            self._completed_lookup_tasks[task_id] = bitmap
        self._signal_lookup_event()

    async def _execute_load_in_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Load each found key from its file via dynamic DMA read."""
        bitmap = Bitmap(len(keys))
        accessed_keys: list[ObjectKey] = []
        try:
            for i, key in enumerate(keys):
                with self._lock:
                    storage_obj = self._memory_objects.get(key)
                if storage_obj is None:
                    continue

                mem_addr = objects[i].meta.address
                mem_size = objects[i].meta.phy_size
                mem_indices = self.nixl_agent.get_memory_indices(mem_addr, mem_size)
                file_path = self.nixl_agent.get_file_path_for_key(key)

                await self.nixl_agent.dynamic_load_file(
                    mem_indices, file_path, self.nixl_agent.l1_align_bytes
                )

                bitmap.set(i)
                accessed_keys.append(key)

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

# TODO(Jiayi): OBJ backend is not supported in the dynamic adapter yet.
# Only file-based backends are supported.
_VALID_DYNAMIC_BACKENDS = ("GDS", "GDS_MT", "POSIX", "HF3FS")


class DynamicNixlStoreL2AdapterConfig(L2AdapterConfigBase):
    """Config for the dynamic-file Nixl L2 adapter.

    Fields:
    - backend: Nixl storage backend (GDS, GDS_MT, POSIX, HF3FS).
    - backend_params: Backend-specific parameters as a dict of string
      key-value pairs. Must include ``file_path`` and ``use_direct_io``.
    """

    def __init__(
        self,
        backend: str,
        backend_params: dict[str, str],
    ):
        if backend not in _VALID_DYNAMIC_BACKENDS:
            raise ValueError(
                "backend must be one of %s, got %r" % (_VALID_DYNAMIC_BACKENDS, backend)
            )
        if "file_path" not in backend_params:
            raise ValueError(
                "backend_params must include 'file_path' for backend %r" % backend
            )
        if "use_direct_io" not in backend_params:
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

        return cls(backend=backend, backend_params=backend_params)

    @classmethod
    def help(cls) -> str:
        return (
            "Dynamic Nixl store L2 adapter config fields:\n"
            "- backend (str): Nixl storage backend, "
            "one of %s (required)\n"
            "- backend_params (dict): backend-specific "
            "string key-value pairs. Must include "
            "'file_path' and 'use_direct_io'.\n"
            "- persist_path (str): path to persist metadata "
            "at shutdown (optional)\n"
            "- recover_path (str): path to recover metadata "
            "at startup (optional)" % (_VALID_DYNAMIC_BACKENDS,)
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
