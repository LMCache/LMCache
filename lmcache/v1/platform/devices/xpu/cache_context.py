# SPDX-License-Identifier: Apache-2.0
"""XPU cache context for LMCache-driven multiprocess transfer.

:class:`XPUCacheContext` is the server-side owner of the device staging
buffer on XPU. ``create_cache_context`` builds one per registered worker
from :mod:`~lmcache.v1.multiprocess.modules.lmcache_driven_transfer`. It
imports the worker's KV tensors through
:class:`~lmcache.v1.platform.devices.xpu.ipc_wrapper.XpuIPCWrapper`,
allocates a flat uint8 staging buffer in local VRAM, and registers that
buffer with the GDS context.

That one buffer serves every LMCache-driven L1 tier, not only GDS.
:mod:`~lmcache.v1.multiprocess.object_group_transfer` stages ordinary
host-memory (DRAM) copies through it as well. GDS registration only adds
the ability for
:func:`~lmcache.v1.gpu_connector.gpu_ops.transfer_async` to DMA between
the slab file and the buffer: with no GDS L1 slab configured the
process-global GDS context stays uninitialized, registration is inert,
and the context still drives the DRAM path.

"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
import array

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gds_context import (
    deregister_gds_gpu_buffer,
    register_gds_gpu_buffer,
)
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_device,
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import engine_group_layer_indices
from lmcache.v1.platform.base.cache_context import BaseCacheContext

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    """Return tensors reconstructed from XPU IPC wrappers.

    Args:
        kv_caches: XPU IPC wrappers received over the multiprocess wire.

    Returns:
        The reconstructed XPU KV-cache tensors.
    """
    return [wrapper.to_tensor() for wrapper in kv_caches]


def list_to_xpu_tensor(list: list[int], device: torch.device) -> torch.Tensor:
    """Copy a list of pointers to the device as an int64 tensor.

    The staging ``array`` is built with the *unsigned* ``"Q"`` type code even
    though the result is an ``int64`` tensor. Level Zero places USM device
    allocations high in the address space, so an XPU ``Tensor.data_ptr()``
    routinely exceeds ``INT64_MAX`` and ``array("q", ...)`` would raise
    ``OverflowError``. ``frombuffer`` then reinterprets the identical bytes as
    signed, which is the two's-complement form the kernels cast back to a
    pointer.

    Args:
        list: XPU device addresses, each an engine KV cache
            ``Tensor.data_ptr()``. Only the list itself is host-side. Packed
            as unsigned, so a negative value raises ``OverflowError``.
        device: Destination XPU device.

    Returns:
        A one-dimensional ``int64`` device tensor.
    """
    return torch.frombuffer(array.array("Q", list), dtype=torch.long).to(
        device, non_blocking=True
    )


class _XpuHostCallbackStream:
    """CuPy-free adapter for the stream-ordered callback call sites.

    CUDA contexts expose a CuPy ``ExternalStream`` as ``cupy_stream``.
    CuPy has no XPU build, so this adapter provides the same surface over
    a ``torch.xpu.Stream``. Callers use exactly two members: ``ptr`` --
    the native stream handle forwarded to
    ``DeviceOps.record_completion_on_stream`` and
    ``record_event_on_stream`` -- and ``launch_host_func``.
    """

    def __init__(self, stream: object) -> None:
        self._stream = stream

    @property
    def ptr(self) -> int:
        """Return the SYCL queue pointer backing the wrapped stream.

        ``torch.xpu.Stream`` exposes the underlying ``sycl::queue *`` as
        ``sycl_queue``.

        Raises:
            RuntimeError: If the stream exposes no native handle.
        """
        pointer = getattr(self._stream, "sycl_queue", None)
        if pointer is None:
            pointer = getattr(self._stream, "ptr", None)
        if pointer is None:
            raise RuntimeError("XPU stream does not expose a sycl_queue")
        return int(pointer)

    def synchronize(self) -> None:
        """Wait for all work submitted to the wrapped XPU stream."""
        synchronize = getattr(self._stream, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError("XPU stream does not support synchronization")
        synchronize()

    def launch_host_func(self, callback: Any, arg: Any = None) -> None:
        """Schedule or run ``callback(arg)``.

        Delegates when the backend stream provides ``launch_host_func``
        itself. SYCL exposes no ``cudaLaunchHostFunc`` analogue through
        torch, so the fallback drains the stream and runs the callback
        inline, preserving the ordering guarantee at the cost of a
        host-side sync.
        """
        launch_host_func = getattr(self._stream, "launch_host_func", None)
        if callable(launch_host_func):
            launch_host_func(callback, arg)
            return
        self.synchronize()
        callback(arg)


class _TempXPUBuffer:
    """Owns the XPU staging buffer for MP block-transfer batches.

    One flat uint8 allocation laid out as (batch, object group, kernel
    group), as on CUDA. DRAM and GDS L1 batches both stage through it,
    and the shared ``gpu_ops`` helpers pick the tier per memory object.
    Each object-group region is copied to or from its memory object as
    one contiguous range, so it must keep the memory object's byte
    format.
    """

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        self._kv_groups_manager = kv_layer_groups_manager
        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._max_batch_size = max_batch_size

        self._temp_buffer = torch.empty(
            self._get_size_for_single_batch() * max_batch_size,
            dtype=torch.uint8,
            device=device,
        )

        # (batch_idx, kernel_group_idx) -> (byte offset, size in bytes)
        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        # (batch_idx, object_group_idx) -> (byte offset, size in bytes)
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_start_offset = offset
                object_group_size = 0
                object_group = self._kv_groups_manager.object_groups[object_group_idx]
                for kernel_group_idx in object_group.kernel_group_indices:
                    size = self._get_size_for_kernel_group(kernel_group_idx)
                    self._offset_map_kernel_group_only[
                        (batch_idx, kernel_group_idx)
                    ] = (offset, size)
                    offset += size
                    object_group_size += size

                self._offset_map_object_group_only[(batch_idx, object_group_idx)] = (
                    object_group_start_offset,
                    object_group_size,
                )

        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk, kernel_group_idx
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            self._shape_cache_kernel_group[kernel_group_idx] = (
                shape,
                group.dtype,
            )

    @property
    def max_batch_size(self) -> int:
        """Return the number of chunks that fit in the staging buffer."""
        return self._max_batch_size

    @property
    def buffer(self) -> torch.Tensor:
        """Return the flat staging tensor, for GDS cuFile registration."""
        return self._temp_buffer

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Return a typed staging view for a batch/kernel-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            kernel_group_idx: Index of the kernel group.

        Returns:
            A view with the kernel group's shape and dtype.

        Raises:
            ValueError: If either index is out of range.
        """
        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )

        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Return a flat ``uint8`` staging view for a batch/object-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            object_group_idx: Index of the object group.

        Returns:
            A flat view covering the object's staging region.

        Raises:
            ValueError: If either index is out of range.
        """
        key = (batch_idx, object_group_idx)
        if key not in self._offset_map_object_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or object_group_idx {object_group_idx}"
            )

        offset, size = self._offset_map_object_group_only[key]
        return self._temp_buffer[offset : offset + size]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return ``(shape, dtype)`` for a kernel group and token count."""
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        shape = self._get_shape_for_kernel_group(num_tokens, kernel_group_idx)
        return shape, dtype

    def get_cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
        """Return the staging shape for a kernel group and token count.

        Raises:
            ValueError: If ``num_tokens`` is not a whole number of LMCache
                chunks.
        """
        if num_tokens % self._lmcache_tokens_per_chunk != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be a multiple of "
                f"lmcache_tokens_per_chunk ({self._lmcache_tokens_per_chunk})"
            )

        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        sd = group.shape_desc
        num_chunks = num_tokens // self._lmcache_tokens_per_chunk
        num_slots = (
            self._kv_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)
            * num_chunks
        )
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def _get_size_for_kernel_group(self, kernel_group_idx: int) -> int:
        """Return one kernel group's staging size in bytes, for one chunk."""
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk, kernel_group_idx
        )
        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        return shape.numel() * group.dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        """Return one object group's staging size in bytes, for one chunk."""
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        """Return the staging size in bytes for a single chunk."""
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )


class XPUCacheContext(BaseCacheContext):
    """Cache context for XPU-backed KV tensors in MP handle mode."""

    device_type = "xpu"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: "Sequence[EngineGroupInfo]" = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = False,
        full_sw_kv: bool = False,
    ) -> None:
        """Build an XPU cache context from IPC-wrapped KV tensors.

        Args:
            kv_caches: XPU IPC wrappers received during server registration.
            lmcache_tokens_per_chunk: Tokens per LMCache object.
            layout_hints: Optional KV layout hints from the serving engine.
            engine_group_infos: Engine-neutral group metadata.
            engine_type: Serving engine that produced the KV cache.
            separate_object_groups: Whether to split object groups by window.
            full_sw_kv: Whether sliding-window groups transfer their full KV.

        Raises:
            ValueError: If the reconstructed tensors are not XPU tensors.
        """
        # Kept for close(): IPC wrappers hold mappings that pin the
        # exporter's device memory until explicitly closed.
        self._kv_wrappers: KVCache = list(kv_caches)
        self._gds_registered = False
        try:
            self._init_impl(
                kv_caches,
                lmcache_tokens_per_chunk,
                layout_hints,
                engine_group_infos,
                engine_type,
                separate_object_groups,
                full_sw_kv,
            )
        except BaseException:
            # Partial construction may have imported some mappings
            # already; roll them back so a failed registration does not
            # pin the worker's KV pool.
            self.close()
            raise

    def _init_impl(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int,
        layout_hints: LayoutHints | None,
        engine_group_infos: "Sequence[EngineGroupInfo]",
        engine_type: EngineType,
        separate_object_groups: bool,
        full_sw_kv: bool,
    ) -> None:
        """Body of ``__init__``; split out so the constructor can roll back
        partially imported KV mappings on failure.
        """
        unwrapped = unwrap_kv_cache_tensors(kv_caches)
        kv_caches_norm, engine_kv_formats = normalize_and_discover_per_layer_formats(
            unwrapped,
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        self.device_ = get_device(kv_caches_norm)
        if self.device_.type != "xpu":
            raise ValueError(
                f"XPUCacheContext expected XPU tensors, got {self.device_.type!r}"
            )
        num_layers_val = len(engine_kv_formats)

        kv_layer_groups_manager = KVLayerGroupsManager(
            kv_caches_norm,
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            separate_object_groups=separate_object_groups,
        )
        # Set full_sw_kv before the temp buffer / object groups are built so
        # both are sized for the full (un-windowed) per-chunk geometry.
        if full_sw_kv:
            kv_layer_groups_manager.enable_full_sw_kv()

        # Pre-allocated device buffer for block IDs (up to 1M elements).
        # The caller copies block_ids into this buffer before launching the
        # block-level kernel. Single-thread assumption: no lock needed.
        _MAX_BLOCK_IDS = 1 << 20
        block_ids_buffer = torch.empty(
            _MAX_BLOCK_IDS, dtype=torch.long, device=self.device_
        )

        super().__init__(
            kv_caches=kv_caches_norm,
            device=self.device_,
            num_layers=num_layers_val,
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        self.group_kv_pointers_: list[torch.Tensor] = []
        self._group_kv_pointer_lists: list[list[int]] = []
        for idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            ptrs = get_group_data_ptrs(
                self.kv_caches_,
                self.get_engine_kv_format(idx),
                group.layer_indices,
            )
            self._group_kv_pointer_lists.append(list(ptrs))
            self.group_kv_pointers_.append(list_to_xpu_tensor(ptrs, self.device_))

        # Temporary device buffer for transfers: a single flat uint8 buffer.
        self._temp_buffer = _TempXPUBuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )

        self.stream_ = torch_dev.Stream(device=self.device_)

        # Register the staging buffer with the GDS context on this
        # context's stream. This is what makes the XPU GDS L1 tier work;
        # close() must deregister before the buffer is freed.
        with torch_dev.stream(self.stream_):
            register_gds_gpu_buffer(self._temp_buffer.buffer)
        self._gds_registered = True

        self.host_callback_stream_ = _XpuHostCallbackStream(self.stream_)
        self.host_callback_stream_.launch_host_func(
            lambda logger: logger.info(
                "Initialized xpu stream on device %s", str(self.device_)
            ),
            logger,
        )

    def close(self) -> None:
        """Drain the stream, deregister the GDS staging buffer and release
        the imported KV mappings (reverse of ``__init__``).

        The stream is synchronized first so no in-flight kernel still
        touches the mappings when they are released. Tensors built over
        those mappings must not be dereferenced afterwards. Safe to call
        on a partially constructed context: the constructor's rollback
        path calls it before the later attributes exist.
        """
        stream = getattr(self, "stream_", None)
        synchronize = getattr(stream, "synchronize", None)
        if callable(synchronize):
            synchronize()

        if self._gds_registered:
            self._gds_registered = False
            with torch_dev.stream(stream):
                deregister_gds_gpu_buffer(self._temp_buffer.buffer)

        kv_tensors = getattr(self, "kv_caches_", None)
        if isinstance(kv_tensors, list):
            kv_tensors.clear()

        self._close_kv_wrappers()

    def _close_kv_wrappers(self) -> None:
        """Close every KV wrapper, continuing past per-wrapper failures."""
        for wrapper in self._kv_wrappers:
            try:
                wrapper.close()
            except Exception:
                logger.warning("KV wrapper close failed", exc_info=True)
        self._kv_wrappers = []

    @property
    def stream(self) -> Any:
        """Return the XPU stream used for transfer work."""
        return self.stream_

    @property
    def cupy_stream(self) -> _XpuHostCallbackStream:
        """Return the host-callback stream adapter for shared MP paths."""
        return self.host_callback_stream_

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Return the pre-computed device tensor of KV pointers for a group."""
        return self.group_kv_pointers_[kernel_group_idx]

    def get_kernel_group_kv_pointer_list(self, kernel_group_idx: int) -> list[int]:
        """Return the host-side KV pointers of a kernel group.

        Same order as :meth:`get_kernel_group_kv_pointers`, served from a
        cached host list so the call never synchronizes the device.
        """
        return self._group_kv_pointer_lists[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Return the staging buffer for a batch/kernel-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            kernel_group_idx: Index of the kernel group to transfer.

        Returns:
            A typed view into the XPU staging allocation.
        """
        return self._temp_buffer.get_temp_kernel_group_buffer(
            batch_idx, kernel_group_idx
        )

    @property
    def max_batch_size(self) -> int:
        """Return the maximum number of chunks transferred per batch."""
        return self._temp_buffer.max_batch_size

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Return the staging buffer for a batch/object-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            object_group_idx: Index of the object group to transfer.

        Returns:
            A flat view covering the object's XPU staging allocation.
        """
        return self._temp_buffer.get_temp_object_group_buffer(
            batch_idx, object_group_idx
        )

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return the shape and dtype for a kernel-group allocation.

        Args:
            num_tokens: Number of tokens represented by the allocation.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The allocation shape and element dtype.
        """
        return self._temp_buffer.get_kernel_group_shape_dtype(
            num_tokens, kernel_group_idx
        )

    def cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._temp_buffer.get_cache_size_per_token()
