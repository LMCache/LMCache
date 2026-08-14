# SPDX-License-Identifier: Apache-2.0
"""Generic opaque per-chunk blob store over the distributed StorageManager.

Stores arbitrary auxiliary byte blobs alongside KV: one blob per cacheable
chunk of a request range, keyed by the chunk's content hash in a dedicated
object group so the blobs are reusable across requests and never collide with
KV groups. The store NEVER interprets the bytes -- the caller packs whatever it
wants into each chunk's blob (e.g. all layers' compressed projections for that
chunk), so a single store/retrieve covers all chunks (hence all layers) at
once. Per-chunk byte sizes are supplied by the caller, so blobs may be any size
and reads need no separate size lookup.

Copy discipline: ``store`` moves the caller's packed payload to the L1 device
in one transfer, then writes each chunk in place. ``fetch_into_ipc`` prefetches
the chunks and copies each straight into a caller-provided GPU buffer (a
same-GPU device-to-device copy when L1 is GPU-resident), synchronizing before it
releases the prefetch read locks.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext

logger = init_logger(__name__)


class AuxBlobStore:
    """Generic content-hash-keyed opaque byte-blob store (one blob per chunk).

    The blob bytes are never interpreted; callers are free to pack any layout
    into each chunk's blob. Keys are resolved through the context's
    session/token-hasher, so a blob is reused whenever its chunk recurs.

    Attributes:
        prefetch_timeout_s (float): Max seconds to wait for a retrieve prefetch.
    """

    def __init__(
        self,
        ctx: "MPCacheServerContext",
        prefetch_timeout_s: float = 5.0,
    ) -> None:
        """Initialize the store.

        Args:
            ctx (MPCacheServerContext): Server context providing
                ``resolve_obj_keys`` and ``storage_manager``.
            prefetch_timeout_s (float): Retrieve prefetch wait timeout in
                seconds. Defaults to ``5.0``.
        """
        self._ctx = ctx
        self.prefetch_timeout_s = prefetch_timeout_s

    def _obj_keys(self, key: IPCCacheServerKey, group: int) -> list:
        """Resolve per-chunk content-hashed object keys for a range.

        Args:
            key (IPCCacheServerKey): Request key (token IDs + ``[start, end)``).
            group (int): Object-group id for this aux stream (disjoint from KV
                groups); the same ``group`` must be used for store and retrieve.

        Returns:
            list[ObjectKey]: One content-hashed key per chunk in the range
            (reusable across requests that share the chunk).
        """
        return self._ctx.resolve_obj_keys(key, [group])[0]

    @staticmethod
    def _layout(size: int) -> MemoryLayoutDesc:
        """Single-group memory layout for ONE aux chunk-object.

        ``reserve_write``/``submit_prefetch_task`` apply ONE
        layout to EVERY key — the ``shapes`` list describes an object's internal
        GROUPS (e.g. K and V), NOT one-shape-per-key. Each aux chunk is a
        separate SINGLE-group object, so the layout has exactly one shape;
        passing N shapes (one per chunk) made every object N groups (so
        ``get_size()`` = N× and the ``.tensor`` reshape to one group's shape
        failed). The distributed allocator also works in 2-byte (bf16) units, so
        a ``uint8 [bytes]`` layout over-allocates 2x — lay the opaque blob out as
        ``bfloat16 [bytes // 2]`` (aux byte sizes are even: bf16 elements ×2).
        bf16 is only the allocation unit; the bytes stay opaque.

        Args:
            size (int): One chunk's blob byte length (even).

        Returns:
            MemoryLayoutDesc: A single ``[size // 2]`` bfloat16 group.
        """
        return MemoryLayoutDesc(
            shapes=[torch.Size([size // 2])],
            dtypes=[torch.bfloat16],
        )

    def store(
        self,
        key: IPCCacheServerKey,
        group: int,
        sizes: list[int],
        blob: torch.Tensor,
    ) -> bool:
        """Store one opaque blob per chunk of the request range.

        Args:
            key (IPCCacheServerKey): Store key for the range being cached.
            group (int): Object-group id for this aux stream.
            sizes (list[int]): Per-chunk blob byte lengths (chunk order); must
                have one entry per chunk in the range and sum to
                ``blob.numel()``.
            blob (torch.Tensor): 1-D ``uint8`` tensor of the concatenated
                per-chunk blobs.

        Returns:
            bool: ``True`` if the store path ran (including the no-op case of
            already-resident chunks); ``False`` on a size/chunk-count mismatch
            or an internal error (the caller recomputes on the next hit).
        """
        try:
            obj_keys = self._obj_keys(key, group)
            n = len(obj_keys)
            if n == 0 or n != len(sizes) or len(set(sizes)) != 1:
                # Single-group layout requires equal-sized chunks (true for the
                # aligned full-block aux chunks); bail otherwise.
                return False
            reserved = self._ctx.storage_manager.reserve_write(
                obj_keys, self._layout(sizes[0]), mode="new"
            )
            if not reserved:
                return True  # all chunks already resident
            # One transfer to the L1 device, then same-device per-chunk writes.
            # Layout + blob are bf16; per-chunk lengths are byte sizes, so the
            # bf16 element count is ``s // 2`` (offsets advance in elements).
            dev = None
            for _o in reserved.values():
                _t = _o.tensor if _o is not None else None
                if _t is not None:
                    dev = _t.device
                    break
            if dev is None:
                return True  # nothing new to write (all chunks resident)
            blob = blob.reshape(-1).view(torch.bfloat16).to(dev)
            written: list = []
            off = 0
            for i, k in enumerate(obj_keys):
                nel = sizes[i] // 2
                seg = blob.narrow(0, off, nel)
                off += nel
                obj = reserved.get(k)
                tensor = obj.tensor if obj is not None else None
                if tensor is None:
                    continue  # already resident (re-store) or reserve failed
                tensor.reshape(-1).narrow(0, 0, nel).copy_(seg)
                written.append(k)
            if written:
                self._ctx.storage_manager.finish_write(written)
            return True
        except Exception:
            logger.exception("aux store failed for request %s", key.request_id)
            return False

    def fetch_into_ipc(
        self,
        obj_keys: list,
        sizes: list[int],
        dst: torch.Tensor,
    ) -> bool:
        """Copy the requested chunks' blobs into a caller-provided GPU buffer.

        Prefetches the per-chunk objects into L1, then copies each chunk directly
        into ``dst`` at its byte offset. ``dst`` is a GPU buffer the caller has
        mapped into this process (e.g. a worker-exported receive buffer via
        ``CudaIPCWrapper.to_tensor``), so when L1 is GPU-resident each copy is a
        same-device (D2D) transfer on the current stream — the data never leaves
        the GPU. The current stream is synchronized before the prefetch read
        locks are released, so the L1 source cannot be evicted while a copy is
        still reading it. The caller selects the stream/device (enter the desired
        CUDA stream context before calling) and owns ``dst``'s lifetime.

        Args:
            obj_keys (list[ObjectKey]): Resolved per-chunk object keys.
            sizes (list[int]): Per-chunk byte lengths (all equal; one per key).
            dst (torch.Tensor): 1-D ``uint8`` GPU buffer of ``>= sum(sizes)``
                bytes, written at per-chunk byte offsets ``i * sizes[i]``.

        Returns:
            bool: ``True`` if every chunk was present and copied; ``False`` on
            empty/size-mismatch or any missing chunk (caller recomputes).
        """
        # First Party
        from lmcache.v1.gpu_connector.gpu_ops import lmcache_memcpy_async_h2d

        n = len(obj_keys)
        if n == 0 or n != len(sizes) or len(set(sizes)) != 1:
            return False
        sm = self._ctx.storage_manager
        # SPARSE: per-chunk fingerprint fetch, not a prefix — mirrors the CB
        # KV retrieve's prefetch shape.
        handle = sm.submit_prefetch_task(
            PrefetchRequestSpec(
                keys=obj_keys,
                # Keyed by object-group index WITHIN the request (must cover
                # exactly range(attn_desc.num_object_groups)), not by the aux
                # stream id: each aux chunk is its own single-group object.
                group_layout_descs={0: self._layout(sizes[0])},
                policy=TrimPolicy.SPARSE,
            )
        )
        sm.wait_prefetch_status(handle, self.prefetch_timeout_s)
        with sm.read_prefetched_results(obj_keys) as memory_objs:
            if memory_objs is None or len(memory_objs) != n:
                return False
            off = 0
            for i, m in enumerate(memory_objs):
                s = sizes[i]
                # dst slice MUST be exactly s bytes: lmcache_memcpy_async_h2d
                # asserts memory_obj.get_size() == gpu_buffer.nbytes. Same
                # allocator-aware copy the KV scatter uses (GDS/Lazy/GPU).
                lmcache_memcpy_async_h2d(m, dst.narrow(0, off, s))
                off += s
        # the copies above are ASYNC on the caller's stream. Sync
        # so they finish READING the L1 source before we drop the read locks
        # (else eviction under L1 pressure could free the source mid-copy). Then
        # release them (read_prefetched_results keeps them on success) — exactly
        # like the KV path; otherwise the lock leaks and the durable aux object
        # TTL-expires into evictability -> later same-chunk fetches MISS -> crash.
        # First Party
        from lmcache import torch_dev

        torch_dev.current_stream().synchronize()
        self._ctx.storage_manager.finish_read_prefetched(obj_keys)
        return True
