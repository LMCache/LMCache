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
from typing import ClassVar
import ctypes
import math

# Third Party
import torch

# First Party
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.native_block_transfer import (
    allocate_pinned_chunk,
    transfer_with_address_kernel,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native


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
