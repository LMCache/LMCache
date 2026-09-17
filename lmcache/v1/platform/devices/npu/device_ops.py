# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU ops backend: bulk-bind the plugin's ``lmcache_ascend.ops``.

:class:`NpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the external Ascend plugin's compiled extension on top of the
torch baseline. All NPU-specific kernels live in the ``lmcache_ascend``
plugin, which curates an ops-only binding surface (``lmcache_ascend.ops``,
mirroring ``lmcache.cuda_ops``); this package only wires detection and
backend selection. If the plugin is missing, a warning is logged and the
instance stays on the torch fallback (soft-fail, same as CUDA).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar
import ctypes

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


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

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # Third Party
            import lmcache_ascend.ops as native
        except ImportError:
            logger.warning(
                "lmcache_ascend.ops surface not found; NpuDeviceOps stays "
                "on the torch baseline for all ops."
            )
            return
        self.bind_native(native)

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

    def lmcache_memcpy_async(
        self,
        dest: int,
        src: int,
        nbytes: int,
        direction: lmcache_native.TransferDirection,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        """Copy ``nbytes`` between an NPU device pointer and host memory.

        The torch baseline's pointer mode drives ``cudaMemcpy`` through
        libcudart, which cannot address NPU memory and crashes the process.
        Instead build zero-copy views over both pointers and issue one
        stream-ordered ``copy_`` on the current stream. The CUDA-specific
        ``host_buffer_offset`` / ``host_buffer_alignments`` chunking
        parameters are ignored: a single ``copy_`` needs no boundary
        splitting.

        Args:
            dest: Destination address. Host for D2H, device for H2D.
            src: Source address. Device for D2H, host for H2D.
            nbytes: Number of bytes to copy.
            direction: H2D or D2H; selects which side is host memory.
            host_buffer_offset: Unused (CUDA chunking parameter).
            host_buffer_alignments: Unused (CUDA chunking parameter).

        Raises:
            ValueError: If ``nbytes`` is not positive or the direction is
                unsupported.
        """
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
        dev_view = torch_ops._tensor_from_ptr(dev_ptr, (nbytes,), torch.uint8, device)
        host_view = torch.frombuffer(
            (ctypes.c_uint8 * nbytes).from_address(host_ptr), dtype=torch.uint8
        )
        if is_d2h:
            host_view.copy_(dev_view, non_blocking=True)
        else:
            dev_view.copy_(host_view, non_blocking=True)
