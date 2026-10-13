# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the SYCL extension on top of the torch baseline.  If the extension
is not built, a warning is logged and the instance stays on the torch
fallback.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.devices.xpu.tensor_from_ptr import (
    construct_xpu_tensor_from_data_pointer,
)

if TYPE_CHECKING:
    # Third Party
    import torch

logger = init_logger(__name__)


def _synchronize_all_xpu_devices() -> None:
    """Wait for all queued work on every XPU device.

    torch cannot rebuild a stream from a raw ``sycl::queue`` pointer, so
    the Python fallback recorders cannot wait on one queue only.
    """
    # Third Party
    import torch

    for device_index in range(torch.xpu.device_count()):
        torch.xpu.synchronize(device_index)


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"

    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: "torch.dtype",
        device: "torch.device",
    ) -> "torch.Tensor":
        """Create a non-owning XPU tensor from a process-local pointer.

        The LMCache-driven transfer path passes device buffers as raw
        pointers, so the torch fallback for the block-transfer kernels can
        only run on XPU once those pointers can be turned back into tensors.
        """
        return construct_xpu_tensor_from_data_pointer(ptr, shape, dtype, device)

    def record_completion_on_stream(
        self,
        stream_ptr: int,
        kind: str,
        payload: bytes,
    ) -> None:
        """Publish a completion only after prior XPU work finishes.

        Fallback for builds without ``lmcache.xpu_ops``. The native
        recorder replaces this through :meth:`bind_native` and orders the
        record with a SYCL ``host_task`` instead of blocking the caller.

        Args:
            stream_ptr: ``sycl::queue *`` of the stream the work was
                submitted to; the fallback waits on every device instead.
            kind: Completion handler key.
            payload: Encoded completion payload.
        """
        _synchronize_all_xpu_devices()
        super().record_completion_on_stream(stream_ptr, kind, payload)

    def record_event_on_stream(
        self,
        stream_ptr: int,
        event_type_name: str,
        session_id: str,
        str_metadata: dict[str, str],
        int_metadata: dict[str, int],
    ) -> None:
        """Record an event only after prior XPU work finishes.

        Fallback for builds without ``lmcache.xpu_ops``; see
        :meth:`record_completion_on_stream`.

        Args:
            stream_ptr: ``sycl::queue *`` of the stream the work was
                submitted to; the fallback waits on every device instead.
            event_type_name: Serialized event type.
            session_id: Session associated with the event.
            str_metadata: String-valued event metadata.
            int_metadata: Integer-valued event metadata.
        """
        _synchronize_all_xpu_devices()
        super().record_event_on_stream(
            stream_ptr, event_type_name, session_id, str_metadata, int_metadata
        )

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.xpu_ops as sycl
        except ImportError:
            logger.warning(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the "
                "torch baseline for all ops."
            )
            return
        self.bind_native(sycl)
