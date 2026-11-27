# SPDX-License-Identifier: Apache-2.0
"""GCU IPC wrapper for cross-process KV-cache tensor sharing.

:class:`GcuIPCWrapper` mirrors :class:`~lmcache.v1.platform.devices.cuda.
ipc_wrapper.CudaIPCWrapper` but uses the torch_gcu interprocess storage
primitives (``UntypedStorage._share_gcu_`` / ``_new_shared_gcu``) instead of
the CUDA ones. Subclassing :class:`DeviceIPCWrapper` keeps the single msgspec
ext code shared across all wrappers; the data-only descriptor preserves the
concrete class so the server's ``to_tensor`` reconstructs a GCU tensor.

vLLM allocates the KV cache as one large contiguous allocation and exposes
each attention layer as a VIEW into it.  Sharing (worker side) and opening
(server side) that allocation once per layer re-exports/re-maps the same
multi-GiB region dozens of times, which exceeds the GCU driver's total
IPC-mapped-memory budget and aborts the server with
``Cannot attach IPC handle with flags: 1``.  We therefore cache the shared
handle per unique storage (worker side) and the opened storage per unique
handle (server side) so each allocation is exported and opened exactly once.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, ClassVar
import threading

# Third Party
import msgspec
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.common.logging import init_logger
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.gcu.ipc_wire import GcuIPCData

logger = init_logger(__name__)


class GcuIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper for GCU KV-cache tensors from the torch caching allocator."""

    device_type: ClassVar[str] = "gcu"
    wire_type: ClassVar[type[msgspec.Struct]] = GcuIPCData

    _ordinal_fallback_warned: ClassVar[bool] = False

    # Process-local caches.  The vLLM worker and the LMCache server are
    # separate processes, so each keeps its own cache.  Guarded by locks
    # because the LMCache server is threaded (--max-workers).
    _share_lock: ClassVar[threading.Lock] = threading.Lock()
    _open_lock: ClassVar[threading.Lock] = threading.Lock()
    _shared_handle_cache: ClassVar[dict[tuple[int, int], Any]] = {}
    _opened_storage_cache: ClassVar[dict[tuple, "torch.UntypedStorage"]] = {}

    @classmethod
    def clear_ipc_caches(cls) -> None:
        """Drop the process-local IPC caches (teardown / re-registration)."""
        with cls._share_lock:
            cls._shared_handle_cache.clear()
        with cls._open_lock:
            cls._opened_storage_cache.clear()

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "GcuIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A GCU tensor backed by PyTorch's caching allocator.

        Returns:
            A new :class:`GcuIPCWrapper` wrapping ``tensor`` for the wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.common.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Permute any non-contiguous view (e.g. vLLM's NHD-over-HND) so the
        # shape/stride we encode across IPC reflects the physical layout.
        tensor = attempt_permute_to_contiguous_view(tensor)

        storage = tensor.untyped_storage()
        # (base pointer, storage size) uniquely identifies the underlying
        # allocation.  Layer views into one KV-cache pool share the same
        # key, so the storage is exported only once.
        cache_key = (int(storage.data_ptr()), int(storage.nbytes()))
        with self._share_lock:
            handle = self._shared_handle_cache.get(cache_key)
            if handle is None:
                handle = storage._share_gcu_()  # noqa: SLF001
                self._shared_handle_cache[cache_key] = handle
                logger.debug(
                    "GCU WRAP new storage ptr=%s nbytes=%s ref_off=%s",
                    cache_key[0],
                    cache_key[1],
                    handle[5],
                )
            else:
                logger.debug(
                    "GCU WRAP reuse storage ptr=%s ref_off=%s",
                    cache_key[0],
                    handle[5],
                )

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    @classmethod
    def _get_device_uuid(cls, device_index: int) -> str:
        """Return a stable, process-independent identifier for a GCU device.

        Prefers ``get_device_properties(idx).uuid`` (``localUuidId`` on current
        torch_gcu). Falls back to the raw ordinal on torch_gcu versions that
        expose no physical identifier; that fallback is correct only when the
        worker and server enumerate GCUs in the same order (single-node, no
        per-process device masking).

        Args:
            device_index: Local device ordinal to identify.

        Returns:
            A stable string identifier for the device.
        """
        props = torch_dev.get_device_properties(device_index)
        uuid = getattr(props, "uuid", None)
        if uuid:
            return f"gcu-uuid:{uuid}"

        if not cls._ordinal_fallback_warned:
            cls._ordinal_fallback_warned = True
            logger.warning(
                "GCU device properties expose no physical (uuid/pci) "
                "identifier; using the device ordinal for cross-process IPC "
                "mapping. This is correct only when the vLLM worker and the "
                "lmcache server enumerate GCUs in the same order (single-node, "
                "no per-process device masking)."
            )
        return f"gcu-ordinal:{device_index}"

    def to_wire(self) -> GcuIPCData:
        """Return the shared-storage handle and tensor metadata."""
        return GcuIPCData(version=1, metadata=self.wire_metadata(), handle=self.handle)

    @classmethod
    def from_wire(cls, data: msgspec.Struct) -> "GcuIPCWrapper":
        """Restore a validated shared-storage descriptor without opening it."""
        if not isinstance(data, GcuIPCData):
            raise ValueError("Expected GCU IPC descriptor")
        obj = object.__new__(cls)
        obj.restore_metadata(data.metadata)
        obj.handle = data.handle
        return obj

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the GCU KV-cache tensor from the IPC handle.

        ``_new_shared_gcu`` does not implicitly run the torch_gcu lazy init, so
        the caching-allocator pool and peer access must be initialized before
        the driver's ``topsIpcOpenMemHandle`` is reached (mirrors torch_gcu's
        own ``rebuild_gcu_tensor``).

        Returns:
            The reconstructed GCU KV-cache tensor.
        """
        device_index = self._get_device_index_from_uuid(self.device_uuid)

        lazy_init = getattr(torch_dev, "_lazy_init", None)
        if lazy_init is not None:
            lazy_init()

        # Open each unique IPC region once; layer views sharing that region
        # reuse the already-opened storage instead of re-mapping the whole
        # multi-GiB allocation (which trips the GCU driver's IPC-memory cap).
        cache_key = self.handle
        with self._open_lock:
            storage = self._opened_storage_cache.get(cache_key)
            if storage is None:
                storage = torch.UntypedStorage._new_shared_gcu(  # noqa: SLF001
                    device_index, *self.handle[1:]
                )
                self._opened_storage_cache[cache_key] = storage
                logger.debug(
                    "GCU OPEN new handle uuid=%s dev=%s shape=%s ref_off=%s",
                    self.device_uuid,
                    device_index,
                    self.shape,
                    self.handle[5],
                )
            else:
                logger.debug(
                    "GCU OPEN reuse handle uuid=%s dev=%s shape=%s ref_off=%s",
                    self.device_uuid,
                    device_index,
                    self.shape,
                    self.handle[5],
                )

        t = torch.empty(
            (), device=f"{torch_device_type}:{device_index}", dtype=self.dtype
        )
        t.set_(storage, self.storage_offset, self.shape, self.stride)
        return t
