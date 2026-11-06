# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU ops backend: bulk-bind the plugin's ``lmcache_ascend.ops``.

:class:`NpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the external Ascend plugin's compiled extension on top of the
torch baseline. All NPU-specific kernels live in the ``lmcache_ascend``
plugin, which curates an ops-only binding surface (``lmcache_ascend.ops``,
mirroring ``lmcache.cuda_ops``); this package only wires detection and
backend selection. If the plugin is missing, :meth:`ensure_native` raises
:class:`ImportError`: the torch baseline would reconstruct pointer-mode
staging buffers as CPU tensors and silently transfer in the wrong
address space. It overrides :meth:`multi_layer_block_kv_transfer` to lower
tensor operands to the plugin's address-taking kernel.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, cast
import ctypes
import math

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import synchronize_device, torch_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache

logger = init_logger(__name__)

#: Chunks per kernel launch, matching the CUDA kernel's limit.
MAX_CHUNKS_PER_LAUNCH = 4


def _synchronize_npu_stream_pointer(stream_ptr: int) -> None:
    """Synchronize a raw NPU stream pointer through the CANN runtime.

    Args:
        stream_ptr: Raw ``aclrtStream`` handle.

    Raises:
        TypeError: If ``stream_ptr`` is not an int.
        RuntimeError: If the acl runtime module is unavailable or reports
            an error.
    """
    if not isinstance(stream_ptr, int):
        raise TypeError("NPU stream pointer must be an int")
    try:
        # Third Party
        from acl import rt as aclrt
    except ImportError as exc:
        raise RuntimeError(
            f"Unable to synchronize NPU stream pointer {stream_ptr}: "
            "the acl runtime module is unavailable"
        ) from exc
    try:
        ret = aclrt.synchronize_stream(stream_ptr)
    except Exception as exc:
        raise RuntimeError(
            f"aclrtSynchronizeStream raised for stream {stream_ptr}"
        ) from exc
    # Some CANN binding versions return a NumPy scalar rather than a plain
    # int; coerce so a non-zero error code is never mistaken for success.
    if ret is not None and int(ret) != 0:
        raise RuntimeError(
            f"aclrtSynchronizeStream failed with error {ret} for stream {stream_ptr}"
        )


def _paged_pointer_table(
    paged_kv: DiscoverableKVCache,
    engine_kv_format: lmcache_native.EngineKVFormat,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the kernel-ordered per-layer base addresses of ``paged_kv``.

    Args:
        paged_kv: Normalized paged KV tensors.
        engine_kv_format: Layout of ``paged_kv``.
        num_layers: Number of layers to address.
        device: Device the kernel runs on.

    Returns:
        A 1-D int64 tensor on ``device`` with the addresses in the order
        ``KVFormatSpec.data_ptrs`` defines for the format.
    """
    # Imported lazily, not at module scope: the specs package reads
    # ``lmcache.device_ops``, which is resolved only once the backends that
    # import this module have been imported.
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec

    addresses = get_spec(paged_kv, engine_kv_format).data_ptrs(list(range(num_layers)))
    # Addresses are unsigned; the kernel reads the same bits as int64.
    as_int64 = np.array(addresses, dtype=np.uint64).view(np.int64)
    return torch.from_numpy(as_int64).to(device=device)


def allocate_pinned_chunk(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a page-locked host chunk for the Ascend kernel.

    Pinned chunks keep the asynchronous path; unpinned ones are staged or
    pinned before launch.

    Args:
        shape: Canonical chunk shape.
        dtype: KV element dtype.

    Returns:
        An uninitialized pinned CPU tensor of ``shape`` and ``dtype``.
    """
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _stage_unpinned_host_chunks(
    chunks: list[torch.Tensor], is_d2h: bool
) -> tuple[list[torch.Tensor], bool]:
    """Give the kernel chunks it can reach, and say whether a sync is owed.

    The kernel dereferences host chunk addresses directly, which only works
    for page-locked memory. ``D2H`` cannot pin the caller's tensors in place
    (that would replace them and break the in-place fill), so it writes
    pinned staging buffers that are copied back after the sync. ``H2D``
    pins each unpinned chunk into a temporary whose lifetime the sync
    protects.

    Args:
        chunks: Caller-supplied host chunk tensors.
        is_d2h: Whether the transfer gathers paged KV into the chunks.

    Returns:
        ``(kernel_chunks, needs_sync)``: the chunks whose addresses the
        kernel may use, and whether unpinned input forced staging or
        pinning that must be completed by a device sync.
    """
    if all(chunk.is_pinned() for chunk in chunks):
        return chunks, False
    if is_d2h:
        logger.warning(
            "Unpinned host chunks in a native D2H block transfer (likely "
            "shared memory); staging through pinned buffers adds a host "
            "copy and a device sync."
        )
        return [torch.empty_like(c, pin_memory=True) for c in chunks], True
    logger.warning(
        "Unpinned host chunks in a native H2D block transfer; pinning "
        "temporaries adds a device sync."
    )
    return [c if c.is_pinned() else c.pin_memory() for c in chunks], True


def _launch_block_transfer_batches(
    kernel: Callable[..., None],
    paged_addresses: torch.Tensor,
    chunk_addresses: list[int],
    block_id_tensor: torch.Tensor,
    device: torch.device,
    direction: lmcache_native.TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Launch ``kernel`` in batches of at most ``MAX_CHUNKS_PER_LAUNCH``.

    ``skip_prefix_n_blocks`` counts leading blocks across the whole
    transfer, so a skip longer than one launch carries into the following
    launches.
    """
    blocks_per_launch = MAX_CHUNKS_PER_LAUNCH * lmcache_chunk_size // int(shape_desc.bs)
    for launch, first in enumerate(
        range(0, len(chunk_addresses), MAX_CHUNKS_PER_LAUNCH)
    ):
        start = launch * blocks_per_launch
        launch_block_ids = block_id_tensor[start : start + blocks_per_launch]
        kernel(
            paged_addresses,
            chunk_addresses[first : first + MAX_CHUNKS_PER_LAUNCH],
            launch_block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            # A skip longer than one launch carries into the next launches.
            max(0, min(skip_prefix_n_blocks - start, len(launch_block_ids))),
        )


def transfer_with_address_kernel(
    kernel: Callable[..., None],
    paged_buffer_ptrs_tensor: torch.Tensor | list,
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: lmcache_native.TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Serve :meth:`DeviceOps.multi_layer_block_kv_transfer` on an address kernel.

    Operands follow :meth:`DeviceOps.multi_layer_block_kv_transfer`. Tensor
    operands are lowered to the kernel's address form, in launches of at most
    :data:`MAX_CHUNKS_PER_LAUNCH` chunks. Unpinned host chunks are staged
    through pinned buffers for ``D2H`` and pinned into temporaries for
    ``H2D``; both cases synchronize ``device`` before returning, because the
    kernel reaches them through raw addresses that torch cannot track. Pinned
    chunks keep the asynchronous path. The legacy address form is forwarded
    to the kernel unchanged.

    Args:
        kernel: The compiled ``multi_layer_block_kv_transfer``.
        paged_buffer_ptrs_tensor: Normalized paged KV tensors on ``device``.
        lmcache_objects_ptrs: Host chunk tensors.
        block_ids: Paged-block IDs in chunk order.
        device: Device that owns the paged KV tensors.
        direction: ``D2H`` or ``H2D``.
        shape_desc: Paged-buffer shape descriptor.
        lmcache_chunk_size: Tokens per chunk.
        engine_kv_format: Engine KV layout.
        skip_prefix_n_blocks: Leading blocks to skip across the whole
            transfer; each launch skips the part that falls inside it.
    """
    if not all(isinstance(chunk, torch.Tensor) for chunk in lmcache_objects_ptrs):
        # LMCache-driven callers still pass addresses they batched
        # themselves; forward them untouched until #4465 step 3.10.
        kernel(
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
        return

    paged_kv = cast("DiscoverableKVCache", paged_buffer_ptrs_tensor)
    chunks = cast("list[torch.Tensor]", lmcache_objects_ptrs)
    device = torch.device(device)
    is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
    kernel_chunks, needs_sync = _stage_unpinned_host_chunks(chunks, is_d2h)

    paged_addresses = _paged_pointer_table(
        paged_kv, engine_kv_format, int(shape_desc.nl), device
    )
    _launch_block_transfer_batches(
        kernel,
        paged_addresses,
        [chunk.data_ptr() for chunk in kernel_chunks],
        torch.as_tensor(block_ids, dtype=torch.int64, device=device),
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    )

    if needs_sync:
        # The kernel reaches these buffers through raw addresses that torch's
        # stream tracking cannot see: wait before the host reads the staging
        # copies (D2H) or the pinned temporaries are released (H2D).
        synchronize_device(device)
        if is_d2h:
            for dst, src in zip(chunks, kernel_chunks, strict=True):
                dst.copy_(src)


class NpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "npu"

    def __init__(self) -> None:
        super().__init__()
        # The compiled address-taking block kernel, set by ensure_native().
        # A missing plugin makes ensure_native() raise rather than fall back,
        # so it is None only on an instance that has not been initialized.
        self._block_transfer_kernel: Callable[..., None] | None = None

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        try:
            # Third Party
            import lmcache_ascend.ops as native
        except ImportError as exc:
            raise ImportError(
                "NPU requires the LMCache-Ascend native ops (lmcache_ascend.ops): "
                "Install LMCache-Ascend and retry."
            ) from exc
        # Every supported build exports the block kernel, so a module without
        # it is a broken install: let the AttributeError propagate rather than
        # serving the torch baseline. ``_native_bound`` stays False, so the
        # next call fails the same way.
        self._block_transfer_kernel = native.multi_layer_block_kv_transfer
        self.bind_native(native, skip={"multi_layer_block_kv_transfer"})
        self._native_bound = True

    def allocate_block_transfer_output(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Allocate pinned host chunks when the compiled kernel will read them.

        Args:
            shape: Canonical chunk shape.
            dtype: KV element dtype.

        Returns:
            A pinned CPU tensor while the kernel is bound, otherwise the
            :class:`DeviceOps` allocation.
        """
        if self._block_transfer_kernel is None:
            return super().allocate_block_transfer_output(shape, dtype)
        return allocate_pinned_chunk(shape, dtype)

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: torch.Tensor | list,
        lmcache_objects_ptrs: list[int] | list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Run the block transfer on the compiled Ascend kernel.

        Operands follow :meth:`DeviceOps.multi_layer_block_kv_transfer`.
        :meth:`DeviceSpec.get_ops` binds the kernel before returning this
        backend, and a missing plugin raises :class:`ImportError` there
        instead of falling back. Only an instance that has not run
        :meth:`ensure_native` uses the torch baseline.
        """
        args = (
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
        if self._block_transfer_kernel is None:
            super().multi_layer_block_kv_transfer(*args)
            return
        transfer_with_address_kernel(self._block_transfer_kernel, *args)

    def record_completion_on_stream(
        self,
        stream_ptr: int,
        kind: str,
        payload: bytes,
    ) -> None:
        """Publish a completion only after prior NPU stream work finishes.

        torch_npu does not expose the CUDA host-callback primitive used by
        LMCache's native completion recorder. Synchronizing here preserves
        the storage ownership contract until the plugin ships an async
        callback backend; the plugin's ``lmcache_ascend.ops`` surface
        filters torch-fallback re-exports, so either a genuine native
        binding or this override serves -- never the fallback.

        Args:
            stream_ptr: Raw NPU stream pointer from the generic recorder path.
            kind: Completion handler key.
            payload: Encoded completion payload.

        Raises:
            RuntimeError: If the NPU stream cannot be synchronized.
        """
        _synchronize_npu_stream_pointer(stream_ptr)
        super().record_completion_on_stream(0, kind, payload)

    def record_event_on_stream(
        self,
        stream_ptr: int,
        event_type_name: str,
        session_id: str,
        str_metadata: dict[str, str],
        int_metadata: dict[str, int],
    ) -> None:
        """Record an event only after prior NPU stream work finishes.

        Args:
            stream_ptr: Raw NPU stream pointer from the generic recorder path.
            event_type_name: Serialized event type.
            session_id: Session associated with the event.
            str_metadata: String-valued event metadata.
            int_metadata: Integer-valued event metadata.

        Raises:
            RuntimeError: If the NPU stream cannot be synchronized.
        """
        _synchronize_npu_stream_pointer(stream_ptr)
        super().record_event_on_stream(
            0,
            event_type_name,
            session_id,
            str_metadata,
            int_metadata,
        )

    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a non-owning NPU tensor from a raw device pointer.

        A copy fallback is intentionally not provided: writes through a
        copied tensor would not update the original paged buffer.

        Args:
            ptr: Non-zero pointer in the current process.
            shape: Logical dense tensor shape.
            dtype: Element dtype of the pointed-to storage.
            device: NPU device that owns the pointer.

        Returns:
            A tensor aliasing ``ptr``; the caller owns the allocation
            lifetime.

        Raises:
            RuntimeError: If TorchNPU cannot construct the non-owning view.
        """
        numel = math.prod(shape)
        try:
            storage = torch._C._construct_storage_from_data_pointer(
                ptr, device, numel * dtype.itemsize
            )
            tensor = torch.empty(0, dtype=dtype, device=storage.device)
            # Runtime accepts the 3-arg form (contiguous stride); the stubs
            # only declare the 4-arg overload.
            tensor.set_(storage, 0, shape)  # type: ignore[call-overload]
            return tensor
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "TorchNPU failed to construct a non-owning tensor from a device pointer"
            ) from exc

    def lmcache_memcpy_async(
        self,
        dest: int | torch.Tensor,
        src: int | torch.Tensor,
        nbytes: int,
        direction: lmcache_native.TransferDirection,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        """Copy ``nbytes`` between an NPU device pointer and host memory.

        Tensor operands delegate to the torch baseline (``copy_``-based).
        Pointer mode must not: the baseline drives ``cudaMemcpy`` through
        libcudart, which cannot address NPU memory and crashes the process.
        Instead build zero-copy views over both pointers and issue one
        stream-ordered ``copy_`` on the current stream. The CUDA-specific
        ``host_buffer_offset`` / ``host_buffer_alignments`` chunking
        parameters are ignored: a single ``copy_`` needs no boundary
        splitting.

        Args:
            dest: Destination address or tensor. Host for D2H, device for
                H2D.
            src: Source address or tensor. Device for D2H, host for H2D.
            nbytes: Number of bytes to copy.
            direction: H2D or D2H; selects which side is host memory.
            host_buffer_offset: Unused in pointer mode (CUDA chunking
                parameter).
            host_buffer_alignments: Unused in pointer mode (CUDA chunking
                parameter).

        Raises:
            ValueError: If ``nbytes`` is not positive or the direction is
                unsupported.
        """
        if isinstance(dest, torch.Tensor) or isinstance(src, torch.Tensor):
            torch_ops.lmcache_memcpy_async(
                dest,
                src,
                nbytes,
                direction,
                host_buffer_offset,
                host_buffer_alignments,
            )
            return
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
        is_h2d = int(direction) == int(lmcache_native.TransferDirection.H2D)
        if not (is_d2h or is_h2d):
            raise ValueError(f"Unsupported transfer direction: {direction!r}")
        host_ptr, dev_ptr = (dest, src) if is_d2h else (src, dest)
        # torch_npu patches ``torch.npu`` in at runtime; the hasattr guard
        # both fails loudly without it and narrows the attribute for mypy.
        if not hasattr(torch, "npu"):
            raise RuntimeError("torch.npu is unavailable on this build")
        device = torch.device(f"npu:{torch.npu.current_device()}")
        dev_view = self.tensor_from_ptr(dev_ptr, (nbytes,), torch.uint8, device)
        host_view = torch.frombuffer(
            (ctypes.c_uint8 * nbytes).from_address(host_ptr), dtype=torch.uint8
        )
        if is_d2h:
            host_view.copy_(dev_view, non_blocking=True)
        else:
            dev_view.copy_(host_view, non_blocking=True)
