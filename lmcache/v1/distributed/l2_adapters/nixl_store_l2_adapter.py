# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import os
import threading
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig
from nixl._api import nixl_prepped_dlist_handle as NixlDlistHandle
from nixl._api import nixl_xfer_handle as NixlXferHandle
from nixl._api import (
    nixlBind,
)
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import NixlStoreL2AdapterConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)

logger = init_logger(__name__)

# Main class


@dataclass
class NixlStoreObj:
    """
    The object stored in Nixl L2 cache.
    Can be used for both file and object.
    """

    page_index: int

    size: int  # in bytes

    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None

    fmt: Optional[MemoryFormat] = None
    pin_count: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def increase_pin_count(self):
        with self._lock:
            self.pin_count += 1

    def decrease_pin_count(self):
        with self._lock:
            if self.pin_count > 0:
                self.pin_count -= 1
            else:
                logger.warning(
                    "Trying to decrease pin count of object at page index %d below 0",
                    self.page_index,
                )


class NixlObjPool:
    def __init__(self, num_total_objs: int):
        self.indices = list(range(num_total_objs))
        self._lock = threading.Lock()

    def batched_allocate(self, num_objs: int) -> list[int]:
        with self._lock:
            if num_objs > len(self.indices):
                raise RuntimeError("Not enough objects in the pool")
            allocated = self.indices[:num_objs]
            self.indices = self.indices[num_objs:]
            return allocated

    def batched_free(self, obj_indices: list[int]) -> None:
        with self._lock:
            self.indices.extend(obj_indices)


class NixlStorageAgent:
    agent_name: str
    nixl_agent: NixlAgent
    mem_reg_descs: nixlBind.nixlRegDList
    mem_xfer_handler: NixlDlistHandle

    def __init__(
        self,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        pool_size: int,
    ):
        self.backend = backend
        self.pool_size = pool_size
        self.device = device
        self.backend_params = backend_params

        self.agent_name = "NixlAgent_" + str(uuid.uuid4())
        nixl_conf = NixlAgentConfig(backends=[])
        self.nixl_agent = NixlAgent(self.agent_name, nixl_conf)
        self.nixl_agent.create_backend(backend, backend_params)

    def lazy_init_memory(
        self,
        **kwargs,
    ):
        """
        Lazy initialize memory handlers when the memory buffer is ready.

        """

        assert "buffer_ptr" in kwargs, (
            "buffer_ptr is required for lazy memory initialization"
        )
        assert "buffer_size" in kwargs, (
            "buffer_size is required for lazy memory initialization"
        )
        assert "page_size" in kwargs, (
            "page_size is required for lazy memory initialization"
        )
        buffer_ptr = kwargs["buffer_ptr"]
        buffer_size = kwargs["buffer_size"]
        page_size = kwargs["page_size"]

        device_id = kwargs.get("device_id", 0)
        self.init_mem_handlers(
            self.device, buffer_ptr, buffer_size, page_size, device_id
        )

        self.pool = NixlObjPool(num_total_objs=self.pool_size)
        if self.backend in ["GDS", "GDS_MT", "POSIX", "HF3FS"]:
            assert "file_path" in self.backend_params, (
                f"file_path is required for backend {self.backend}"
            )
            assert "use_direct_io" in self.backend_params, (
                f"use_direct_io is required for backend {self.backend}"
            )

            self.init_storage_handlers_file(
                num_pages=self.pool_size,
                page_size=page_size,
                file_path=self.backend_params["file_path"],
                use_direct_io=self.backend_params["use_direct_io"],
            )
        elif self.backend in ["OBJ"]:
            self.init_storage_handlers_object(
                page_size=page_size,
                num_pages=self.pool_size,
            )
        else:
            raise TypeError(f"Unsupported backend type: {self.backend}")

    def init_mem_handlers(self, device, buffer_ptr, buffer_size, page_size, device_id):
        """
        Initialize memory handlers for the given device and buffer.
        """
        reg_list = [(buffer_ptr, buffer_size, device_id, "")]
        xfer_desc = [
            (base_addr, page_size, device_id)
            for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size)
        ]

        if device == "cpu":
            mem_type = "DRAM"
        else:
            mem_type = "VRAM"

        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type=mem_type)
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            "", xfer_descs, mem_type=mem_type
        )

        self.mem_reg_descs = reg_descs
        self.mem_xfer_handler = xfer_handler

    def init_storage_handlers_file(
        self,
        num_pages: int,
        page_size: int,
        file_path: str,
        use_direct_io: bool,
    ):
        """Initialize storage handlers for file-based backends."""

        # Create file descriptors for Nixl to register
        fds: list[int] = []
        flags = os.O_CREAT | os.O_RDWR
        if use_direct_io:
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            else:
                logger.warning(
                    "use_direct_io is True, but O_DIRECT is not available on "
                    "this system. Falling back to buffered I/O."
                )
        for i in range(num_pages):
            filename = f"obj_{i}_{uuid.uuid4().hex[0:4]}.bin"
            tmp_path = os.path.join(file_path, filename)
            fd = os.open(tmp_path, flags)
            fds.append(fd)

        # Register and prepare xfer handler
        reg_list = []
        xfer_desc = []
        for fd in fds:
            reg_list.append((0, page_size, fd, ""))
            xfer_desc.append((0, page_size, fd))
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="FILE")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="FILE")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="FILE"
        )

        self.storage_reg_descs = reg_descs
        self.storage_xfer_descs = xfer_descs
        self.storage_xfer_handler = xfer_handler

    def init_storage_handlers_object(
        self,
        page_size: int,
        num_pages: int,
    ):
        """Initialize storage handlers for object-based backends."""

        # Create object keys for Nixl to register
        keys = []

        for i in range(num_pages):
            key = f"obj_{i}_{uuid.uuid4().hex[0:4]}"
            keys.append(key)

        # Register and prepare xfer handler
        reg_list = []
        xfer_desc = []
        for i, key in enumerate(keys):
            reg_list.append((0, page_size, i, key))
            xfer_desc.append((0, page_size, i))
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_desc, mem_type="OBJ"
        )

        self.storage_reg_descs = reg_descs
        self.storage_xfer_descs = xfer_descs
        self.storage_xfer_handler = xfer_handler

    def get_mem_to_storage_handle(self, mem_indices, storage_indices) -> NixlXferHandle:
        """Get a Nixl transfer handle for transferring data from memory to storage."""

        return self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.mem_xfer_handler,
            mem_indices,
            self.storage_xfer_handler,
            storage_indices,
        )

    def get_storage_to_mem_handle(self, mem_indices, storage_indices) -> NixlXferHandle:
        """Get a Nixl transfer handle for transferring data from storage to memory."""
        return self.nixl_agent.make_prepped_xfer(
            "READ",
            self.mem_xfer_handler,
            mem_indices,
            self.storage_xfer_handler,
            storage_indices,
        )

    def post_blocking(self, handle: NixlXferHandle):
        """Post a Nixl transfer handle and block until the transfer is done."""

        state = self.nixl_agent.transfer(handle)

        while state != "DONE" and state != "ERR":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError:
                raise

        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")

    async def post_non_blocking(self, handle: NixlXferHandle):
        """Post a Nixl transfer handle and await until the transfer is done."""

        state = self.nixl_agent.transfer(handle)

        while state != "DONE" and state != "ERR":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError:
                raise

            # TODO(Jiayi): Tune this for better perf
            await asyncio.sleep(0.01)

        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")

    def get_storage_indices(self, num_objs: int) -> list[int]:
        return self.pool.batched_allocate(num_objs)

    def release_handle(self, handle):
        self.nixl_agent.release_xfer_handle(handle)

    def close(self):
        self.nixl_agent.release_dlist_handle(self.storage_xfer_handler)
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.storage_reg_descs)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)


class NixlStoreL2Adapter(L2AdapterInterface):
    """
    A Nixl-based L2 adapter
    """

    def __init__(self, config: NixlStoreL2AdapterConfig):
        self._config = config

        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # Cache data structures
        self._memory_objects: dict[ObjectKey, NixlStoreObj] = {}

        # Task ID management
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, bool] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()  # lock for all shared state

        # Asyncio event loop running in a background thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # Initialize Nixl agent
        self.nixl_agent = NixlStorageAgent(
            device="cpu",
            backend=config.backend,
            backend_params=config.backend_params,
            pool_size=config.pool_size,
        )

    #####################
    # Memory Registration Interface
    #####################
    def lazy_init_memory(self, **kwargs) -> None:
        self.nixl_agent.lazy_init_memory(**kwargs)

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
        """
        Submit a store task to store a batch of memory objects associated with
        a batch of keys.

        Args:
            keys (list[ObjectKey]): the list of keys to be stored.
            objects (list[MemoryObj]): the list of memory objects to be stored.
                The length of the objects list should be the same as the length of
                the keys list.

        Returns:
            L2TaskId: the task id of the submitted store task.
        """
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_store_in_the_loop(keys, objects, task_id), self._loop
        )

        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """
        Pop all the completed store tasks with a flag indicating
        whether the task is successful or not.

        Returns:
            dict[L2TaskId, bool]: a dictionary mapping the task id to a boolean flag
            indicating whether the task is successful or not. True means
            successful, and False means failed.
        """
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

        # Schedule the lookup operation in the event loop thread
        self._loop.call_soon_threadsafe(self._execute_lookup_in_the_loop, keys, task_id)
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        def _unlock_keys(keys: list[ObjectKey]) -> None:
            """
            Unlock keys in the event loop thread.
            """
            for key in keys:
                if (obj := self._memory_objects.get(key)) is not None:
                    obj.decrease_pin_count()

        # Schedule the unlock operation in the event loop thread
        self._loop.call_soon_threadsafe(_unlock_keys, keys)

    #####################
    # Load Interface
    ######################

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        # Schedule the load operation in the event loop thread
        asyncio.run_coroutine_threadsafe(
            self._execute_load_in_loop(keys, objects, task_id), self._loop
        )

        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    def close(self):
        # Stop the event loop and wait for the thread to finish
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
            try:
                future.result(timeout=5)  # Wait for tasks to be cancelled
            except Exception:
                pass  # Ignore exceptions during shutdown
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join()

        os.close(self._store_efd)
        os.close(self._lookup_efd)
        os.close(self._load_efd)

    ##################
    # Helper functions
    ##################

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_next_task_id(self) -> L2TaskId:
        """Get the next task ID and increment the counter."""
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _evict_if_needed(
        self,
    ) -> None:
        """
        Evict objects from the cache using desired caching policy.
        """

        # TODO(Jiayi): Support caching policy

        pass

    def _signal_store_event(self) -> None:
        """Signal the store event fd to notify completion."""
        os.eventfd_write(self._store_efd, 1)

    async def _execute_store_in_the_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        success = True
        try:
            # Get memory page indices and storage slot indices
            mem_indices = [obj.meta.address for obj in objects]
            storage_indices = self.nixl_agent.get_storage_indices(num_objs=len(keys))
            storage_objs = [
                NixlStoreObj(
                    page_index=storage_idx,
                    size=obj.meta.phy_size,
                    shape=obj.meta.shape,
                    dtype=obj.meta.dtype,
                    fmt=obj.meta.fmt,
                    pin_count=1,
                )
                for storage_idx, obj in zip(storage_indices, objects, strict=False)
            ]

            handle = self.nixl_agent.get_mem_to_storage_handle(
                mem_indices,
                storage_indices,
            )

            await self.nixl_agent.post_non_blocking(handle)
            self.nixl_agent.release_handle(handle)

        except Exception:
            success = False

        with self._lock:
            for key, storage_obj in zip(keys, storage_objs, strict=False):
                self._memory_objects[key] = storage_obj

            self._completed_store_tasks[task_id] = success

        for storage_obj in storage_objs:
            storage_obj.decrease_pin_count()

        self._signal_store_event()

    def _signal_lookup_event(self) -> None:
        """Signal the lookup event fd to notify completion."""
        os.eventfd_write(self._lookup_efd, 1)

    def _execute_lookup_in_the_loop(
        self, keys: list[ObjectKey], task_id: L2TaskId
    ) -> None:
        bitmap = Bitmap(len(keys))
        with self._lock:
            for i, key in enumerate(keys):
                if (obj := self._memory_objects.get(key)) is None:
                    continue
                bitmap.set(i)
                obj.increase_pin_count()
            self._completed_lookup_tasks[task_id] = bitmap
        self._signal_lookup_event()

    def _signal_load_event(self) -> None:
        """Signal the load event fd to notify completion."""
        os.eventfd_write(self._load_efd, 1)

    async def _execute_load_in_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        try:
            mem_indices = []
            storage_indices = []

            with self._lock:
                for i, key in enumerate(keys):
                    if (storage_obj := self._memory_objects.get(key)) is None:
                        continue
                    mem_indices.append(objects[i].meta.address)
                    storage_indices.append(storage_obj.page_index)

                    bitmap.set(i)

            if mem_indices:
                handle = self.nixl_agent.get_storage_to_mem_handle(
                    mem_indices,
                    storage_indices,
                )
                await self.nixl_agent.post_non_blocking(handle)
                self.nixl_agent.release_handle(handle)

        except Exception:
            pass

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._signal_load_event()
