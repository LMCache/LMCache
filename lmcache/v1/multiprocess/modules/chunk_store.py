# SPDX-License-Identifier: Apache-2.0
"""Dynamically loaded chunk-store RPC using the existing transfer module."""

# Standard
from dataclasses import replace

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.server_module import ServerModuleBuildContext


def build_server_modules(ctx: ServerModuleBuildContext) -> "ChunkStoreModule":
    """Build the chunk-store plugin against the server's registered KV contexts.

    Args:
        ctx: Plugin factory context supplied by ``--server-module`` loading.

    Returns:
        A module exposing the chunk-store RPC on the existing request transport.

    Raises:
        ValueError: The server has no LMCache-driven transfer module.
    """
    for module in ctx.modules:
        if isinstance(module, LMCacheDrivenTransferModule):
            return ChunkStoreModule(ctx.server_context, module)
    raise ValueError("The chunk-store plugin requires LMCache-driven transfer")


class ChunkStoreModule:
    """Store chunks independently and report their source-buffer completion.

    The plugin shares registrations, transfer streams, storage and event backend
    with the ordinary store path. Each chunk uses that path's commit and error
    semantics; a later failure does not roll back earlier chunks.

    Args:
        ctx: Shared server context providing chunk geometry.
        transfer_module: Existing owner of registered KV caches and transfers.
    """

    def __init__(
        self, ctx: MPCacheServerContext, transfer_module: LMCacheDrivenTransferModule
    ) -> None:
        self._ctx = ctx
        self._transfer = transfer_module

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared server context."""
        return self._ctx

    def report_status(self) -> dict[str, dict[str, bool]]:
        """Report that the chunk-store plugin is loaded."""
        return {"chunk_store": {"is_healthy": True}}

    def close(self) -> None:
        """Leave shared registrations and streams to their transfer-module owner."""
        return None

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def store_with_chunk_events(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, list[tuple[bytes, int, int]], bool]:
        """Submit a chunk-aligned range and return one completion per chunk.

        Args:
            key: Worker key with the full token prefix and an absolute range
                consisting of complete LMCache chunks.
            instance_id: Worker ID registered through ``register_kv_cache``.
            block_ids: Raw block IDs for the range, in kernel-group order.
            event_ipc_handle: Producer event ordering reads of source KV blocks.

        Returns:
            Last submitted completion handle, ordered ``(handle, start, end)``
            entries, and whether every chunk store avoided a fatal error.
            Ranges are absolute and end-exclusive. Events permit source reuse;
            they do not guarantee cache residency or completed publication.
            Invalid ranges, block counts and unregistered workers return empty
            handles and False before submission. An empty valid range succeeds.
            A failed chunk stops submission, preserving any earlier completions
            and its own completion handle if device work was submitted.

        Raises:
            RuntimeError: The registered backend cannot provide IPC events.

        Notes:
            Handles follow the ordinary store's event-backend lifetime contract.
            No host/device synchronization is added during submission.
        """
        chunk_size = self._ctx.chunk_size
        if (
            key.worker_id is None
            or key.start < 0
            or key.end < key.start
            or key.end > len(key.token_ids)
            or key.start % chunk_size
            or key.end % chunk_size
        ):
            return b"", [], False
        entry = self._transfer.get_and_touch_context_entry(instance_id)
        if entry is None:
            return b"", [], False
        cache = entry.cache_context
        strides = [
            cache.calculate_num_blocks(chunk_size, group)
            for group in range(cache.kv_layer_groups_manager.num_kernel_groups)
        ]
        num_chunks = (key.end - key.start) // chunk_size
        if len(block_ids) != len(strides) or any(
            len(ids) != num_chunks * stride
            for ids, stride in zip(block_ids, strides, strict=True)
        ):
            return b"", [], False

        terminal = b""
        completions: list[tuple[bytes, int, int]] = []
        for chunk, start in enumerate(range(key.start, key.end, chunk_size)):
            end = start + chunk_size
            handle, success = self._transfer.store(
                replace(key, start=start, end=end),
                instance_id,
                [
                    ids[chunk * stride : (chunk + 1) * stride]
                    for ids, stride in zip(block_ids, strides, strict=True)
                ],
                event_ipc_handle,
            )
            if handle:
                terminal = handle
                completions.append((handle, start, end))
            if not success:
                return terminal, completions, False
        return terminal, completions, True
