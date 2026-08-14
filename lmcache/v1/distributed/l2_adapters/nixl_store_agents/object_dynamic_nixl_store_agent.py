# SPDX-License-Identifier: Apache-2.0
"""Object-backed implementation of the dynamic NIXL storage agent."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.nixl_store_agents.dynamic_nixl_store_agent import (  # noqa: E501
    DynamicNixlStorageAgent,
    _object_key_to_filename,
)

logger = init_logger(__name__)

OBJECT_DYNAMIC_BACKENDS = ("OBJ", "AZURE_BLOB")


class ObjectDynamicNixlStorageAgent(DynamicNixlStorageAgent):
    """Dynamic NIXL storage agent backed by deterministic object keys.

    Object storage does not provide backend-neutral size or deletion APIs.
    The agent therefore uses NIXL's object-presence query for recovery and
    leaves retention to the configured object-storage backend.
    """

    def __init__(
        self,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        self._device_id_counter = 0
        self._device_id_lock = threading.Lock()
        super().__init__(device, backend, backend_params, l1_memory_desc)

    async def dynamic_store(self, mem_indices: list[int], key: ObjectKey) -> None:
        """Write L1 memory pages to the deterministic object for ``key``.

        Args:
            mem_indices: Registered L1 page indices to write.
            key: Cache-object key that determines the object name.
        """
        object_size = len(mem_indices) * self.l1_align_bytes
        reg_descs, xfer_handler = self._register_single_object(
            self._get_object_key_for_key(key), object_size
        )
        try:
            await self._transfer("WRITE", mem_indices, xfer_handler)
        finally:
            self._deregister_object(reg_descs, xfer_handler)

    async def dynamic_load(self, mem_indices: list[int], key: ObjectKey) -> None:
        """Read the deterministic object for ``key`` into L1 memory.

        Args:
            mem_indices: Registered L1 page indices to populate.
            key: Cache-object key that determines the object name.
        """
        object_size = len(mem_indices) * self.l1_align_bytes
        reg_descs, xfer_handler = self._register_single_object(
            self._get_object_key_for_key(key), object_size
        )
        try:
            await self._transfer("READ", mem_indices, xfer_handler)
        finally:
            self._deregister_object(reg_descs, xfer_handler)

    def dynamic_delete(self, key: ObjectKey) -> None:
        """Leave object deletion to the object storage backend's lifecycle.

        Args:
            key: Cache-object key whose object would otherwise be deleted.
        """

    def get_stored_size(self, key: ObjectKey) -> int | None:
        """Return zero when the deterministic object exists, otherwise ``None``.

        Object backends do not expose a backend-neutral object-size query, so
        a successful presence query is represented by zero bytes.

        Args:
            key: Cache-object key whose presence should be checked.

        Returns:
            Zero for a present object, otherwise ``None``.
        """
        if self.object_exists(self._get_object_key_for_key(key)):
            return 0
        return None

    def cleanup(self) -> None:
        """Leave object cleanup to the object storage backend's lifecycle."""

    def object_exists(self, object_key: str) -> bool:
        """Return whether NIXL reports a matching object descriptor.

        Args:
            object_key: Deterministic object key to query.

        Returns:
            ``True`` if NIXL reports an object for the key; otherwise ``False``.
        """
        reg_list = [(0, 0, 0, object_key)]
        try:
            response = self.nixl_agent.query_memory(
                reg_list, self.backend, mem_type="OBJ"
            )
        except Exception as error:
            logger.warning("NIXL object query failed for %s: %s", object_key, error)
            return False
        return bool(response and response[0] is not None)

    def _register_single_object(
        self, object_key: str, object_size: int
    ) -> tuple[Any, Any]:
        """Register one object and prepare its NIXL transfer handler."""
        num_pages = object_size // self.l1_align_bytes
        device_id = self._allocate_device_id()
        reg_list = [(0, object_size, device_id, object_key)]
        xfer_desc = [
            (offset * self.l1_align_bytes, self.l1_align_bytes, device_id)
            for offset in range(num_pages)
        ]
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type="OBJ"
        )
        return reg_descs, xfer_handler

    def _deregister_object(self, reg_descs: Any, xfer_handler: Any) -> None:
        """Release NIXL resources registered for one object."""
        self.nixl_agent.release_dlist_handle(xfer_handler)
        self.nixl_agent.deregister_memory(reg_descs)

    def _get_object_key_for_key(self, key: ObjectKey) -> str:
        """Return the deterministic object-storage key for ``key``."""
        return _object_key_to_filename(key)

    def _allocate_device_id(self) -> int:
        """Allocate a unique OBJ device ID for a registration cycle."""
        with self._device_id_lock:
            device_id = self._device_id_counter
            self._device_id_counter += 1
        return device_id
