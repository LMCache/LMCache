# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral dynamic NIXL storage agent interfaces and lifecycle."""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from typing import Any
import asyncio
import os
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig
from nixl._api import nixl_xfer_handle as NixlXferHandle
from nixl._api import nixlBind

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc


def _object_key_to_filename(key: ObjectKey) -> str:
    """Derive a deterministic storage object name from an object key.

    Args:
        key: Key identifying the stored cache object.

    Returns:
        A deterministic object name that is safe to use as a file name.
    """
    safe_model_name = key.model_name.replace("/", "--")
    chunk_hex = key.chunk_hash.hex()
    return (
        f"{safe_model_name}_{key.kv_rank:08x}_{key.object_group_id:x}_{chunk_hex}.bin"
    )


def _object_key_to_relpath(key: ObjectKey) -> str:
    """Relative path ``<hex[:2]>/<hex[2:4]>/filename`` — a 2-level hash-prefix
    subdir tree (GDS-style) keyed on the chunk-hash hex.

    ``hex`` is ``chunk_hash.hex()``, the same value embedded in the filename, so
    the two subdir levels are the first four hex chars of the hash and match the
    filename's hash prefix (e.g. ``834e...`` -> ``83/4e/``). Spreads files
    across up to 256*256 subdirectories instead of one flat directory.
    """
    h = key.chunk_hash.hex()
    return os.path.join(h[:2], h[2:4], _object_key_to_filename(key))


class DynamicNixlStorageAgent(ABC):
    """Base class for dynamic NIXL storage agents.

    This class owns the NIXL agent, the L1 memory registration, and the common
    transfer lifecycle. Concrete agents map :class:`ObjectKey` values to their
    storage backend without exposing backend-specific paths to callers.
    """

    def __init__(
        self,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        self.backend = backend
        self.device = device
        self.backend_params = backend_params
        self.l1_align_bytes = l1_memory_desc.align_bytes

        self.agent_name = "DynNixlAgent_" + str(uuid.uuid4())
        nixl_conf = NixlAgentConfig(backends=[])
        self.nixl_agent = NixlAgent(self.agent_name, nixl_conf)
        self.nixl_agent.create_backend(backend, backend_params)

        self._init_mem_handlers(
            device,
            l1_memory_desc.ptr,
            l1_memory_desc.size,
            l1_memory_desc.align_bytes,
            device_id=0,
        )

    def get_memory_indices(self, raw_addr: int, mem_size: int) -> list[int]:
        """Return the registered L1 page indices for a memory range.

        Args:
            raw_addr: Starting address of the memory range.
            mem_size: Size of the memory range in bytes.

        Returns:
            Registered NIXL page indices covering the memory range.

        Raises:
            ValueError: If the address or size is not L1-page aligned.
        """
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

    def close(self) -> None:
        """Release NIXL resources owned by this storage agent."""
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)

    @abstractmethod
    async def dynamic_store(self, mem_indices: list[int], key: ObjectKey) -> None:
        """Store L1 memory pages for ``key`` in the backing storage."""

    @abstractmethod
    async def dynamic_load(self, mem_indices: list[int], key: ObjectKey) -> None:
        """Load the stored pages for ``key`` into L1 memory."""

    @abstractmethod
    def dynamic_delete(self, key: ObjectKey) -> None:
        """Delete the backing-storage object associated with ``key``."""

    @abstractmethod
    def get_stored_size(self, key: ObjectKey) -> int | None:
        """Return the stored size for ``key``, or ``None`` if it is absent."""

    @abstractmethod
    def cleanup(self) -> None:
        """Perform best-effort cleanup of backend-specific transient data."""

    def _init_mem_handlers(
        self,
        device: str,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        device_id: int,
    ) -> None:
        """Register the L1 memory used by all dynamic storage transfers."""
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

    async def _transfer(
        self,
        direction: str,
        mem_indices: list[int],
        storage_xfer_handler: Any,
    ) -> None:
        """Transfer pages between L1 memory and a prepared storage handler."""
        storage_indices = list(range(len(mem_indices)))
        handle = self.nixl_agent.make_prepped_xfer(
            direction,
            self.mem_xfer_handler,
            mem_indices,
            storage_xfer_handler,
            storage_indices,
        )
        try:
            await self._post_non_blocking(handle)
        finally:
            self.nixl_agent.release_xfer_handle(handle)

    async def _post_non_blocking(self, handle: NixlXferHandle) -> None:
        """Await a nixl transfer until done."""
        state = self.nixl_agent.transfer(handle)
        while state != "DONE" and state != "ERR":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError:
                raise
            if state != "DONE" and state != "ERR":
                await asyncio.sleep(0.01)
        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")
