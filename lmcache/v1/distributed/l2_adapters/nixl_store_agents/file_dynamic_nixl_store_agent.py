# SPDX-License-Identifier: Apache-2.0
"""File-backed implementation of the dynamic NIXL storage agent."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import os
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.nixl_store_agents.dynamic_nixl_store_agent import (  # noqa: E501
    DynamicNixlStorageAgent,
    _object_key_to_filename,
    _object_key_to_relpath,
)

logger = init_logger(__name__)

FILE_DYNAMIC_BACKENDS = ("GDS", "GDS_MT", "POSIX", "HF3FS")


class FileDynamicNixlStorageAgent(DynamicNixlStorageAgent):
    """Dynamic NIXL storage agent backed by one file per object.

    The L1 memory handler is registered once at initialization. Each file is
    registered for the duration of an individual store or load transfer and
    is deregistered immediately afterward.
    """

    def __init__(
        self,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        if "file_path" not in backend_params:
            raise ValueError(
                "backend_params must include 'file_path' for backend %r" % backend
            )
        if "use_direct_io" not in backend_params:
            raise ValueError(
                "backend_params must include 'use_direct_io' for backend %r" % backend
            )

        self.file_path = backend_params["file_path"]
        # Validate and create the storage directory before registering any NIXL
        # resources so a filesystem error cannot leak a partially initialized
        # base agent.
        os.makedirs(self.file_path, exist_ok=True)

        super().__init__(device, backend, backend_params, l1_memory_desc)
        self.use_direct_io = (
            str(backend_params.get("use_direct_io", "false")).lower() == "true"
        )
        # Opt-in: spread per-key files across a fixed 2-level subdir tree
        # instead of one flat directory. Default false = the original flat
        # layout (unchanged). See _object_key_to_relpath / get_file_path_for_key.
        self.shard_dirs = (
            str(backend_params.get("shard_dirs", "false")).lower() == "true"
        )
        # Subdirs already created (shard_dirs only), to skip redundant makedirs
        # on the store hot path. Bounded by the fanout (<= 256*256 entries).
        self._created_subdirs: set[str] = set()

    async def dynamic_store(self, mem_indices: list[int], key: ObjectKey) -> None:
        """Write-to-temp-then-rename to publish the final file atomically.

        The DMA write goes to ``<file_path>.tmp.<uuid>`` in the same
        directory. Only after the transfer completes successfully is the
        temp file atomically renamed to the final path, ensuring that
        concurrent readers (including other processes sharing the same
        directory) never observe a partially-written file.
        """
        file_path = self._get_file_path_for_key(key)
        page_size = self.l1_align_bytes
        file_size = len(mem_indices) * page_size
        tmp_path = f"{file_path}.tmp.{uuid.uuid4().hex}"
        # With sharding, create the subdir once per bucket (cached). Flat layout
        # needs nothing here — the base dir is created at init.
        if self.shard_dirs:
            subdir = os.path.dirname(tmp_path)
            if subdir not in self._created_subdirs:
                os.makedirs(subdir, exist_ok=True)
                self._created_subdirs.add(subdir)
        fd = os.open(tmp_path, self._open_flags(create=True))
        try:
            reg_descs, xfer_handler = self._register_single_file(
                fd, file_size, page_size
            )
            try:
                await self._transfer("WRITE", mem_indices, xfer_handler)
            finally:
                self._deregister_file(reg_descs, xfer_handler)
        except BaseException:
            # Best-effort cleanup of the temp file on failure.
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        finally:
            os.close(fd)

        # Atomic publish: readers only ever see a complete file at file_path.
        # TODO(Jiayi): Only guaranteed to be atomic within the local posix filesystems.
        os.rename(tmp_path, file_path)

    async def dynamic_load(self, mem_indices: list[int], key: ObjectKey) -> None:
        """Open an existing file, DMA read into L1 memory, then clean up."""
        file_path = self._get_file_path_for_key(key)
        page_size = self.l1_align_bytes
        file_size = len(mem_indices) * page_size
        fd = os.open(file_path, self._open_flags(create=False))
        try:
            reg_descs, xfer_handler = self._register_single_file(
                fd, file_size, page_size
            )
            try:
                await self._transfer("READ", mem_indices, xfer_handler)
            finally:
                self._deregister_file(reg_descs, xfer_handler)
        finally:
            os.close(fd)

    def dynamic_delete(self, key: ObjectKey) -> None:
        """Delete a storage file from disk."""
        file_path = self._get_file_path_for_key(key)
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            logger.warning("File already deleted: %s", file_path)

    def get_stored_size(self, key: ObjectKey) -> int | None:
        """Return the data-file size for ``key``, or ``None`` if it is absent."""
        try:
            return os.stat(self._get_file_path_for_key(key)).st_size
        except FileNotFoundError:
            return None

    def cleanup(self) -> None:
        """Remove leftover ``*.tmp.*`` files in the storage directory.

        These can be left behind if a store crashed between opening the
        temp file and the atomic rename. Called at shutdown as a best-effort
        GC; orphans don't affect correctness because they're never matched
        by the deterministic ``ObjectKey → filename`` mapping. Walks
        subdirectories so it also reaches temp files under the sharded layout.
        """
        for root, _dirs, files in os.walk(self.file_path):
            for name in files:
                # Temp suffix format: "<final_name>.tmp.<hex>"
                if ".tmp." in name:
                    try:
                        os.unlink(os.path.join(root, name))
                    except FileNotFoundError:
                        pass
                    except OSError as e:
                        logger.warning(
                            "Failed to remove leftover temp file %s: %s", name, e
                        )

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

    def _register_single_file(
        self, fd: int, file_size: int, page_size: int
    ) -> tuple[Any, Any]:
        """Register a single file with NIXL for a transfer."""
        num_pages = file_size // page_size
        reg_list = [(0, file_size, fd, "")]
        xfer_desc = [(offset * page_size, page_size, fd) for offset in range(num_pages)]
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="FILE")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="FILE")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type="FILE"
        )
        return reg_descs, xfer_handler

    def _deregister_file(self, reg_descs: Any, xfer_handler: Any) -> None:
        """Release NIXL resources registered for a single file."""
        self.nixl_agent.release_dlist_handle(xfer_handler)
        self.nixl_agent.deregister_memory(reg_descs)

    def _get_file_path_for_key(self, key: ObjectKey) -> str:
        """Return the on-disk path for ``key``: sharded when ``shard_dirs`` is
        set (see ``_object_key_to_relpath``), otherwise the flat layout.
        """
        if self.shard_dirs:
            return os.path.join(self.file_path, _object_key_to_relpath(key))
        return os.path.join(self.file_path, _object_key_to_filename(key))
