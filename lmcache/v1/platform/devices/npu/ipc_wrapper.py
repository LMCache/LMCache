# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU IPC wrapper implementation.

:class:`NpuIPCWrapper` shares KV-cache tensors across processes through
torch_npu's storage IPC (``_share_npu_`` / ``_new_shared_npu``), mirroring
what :class:`~lmcache.v1.platform.cuda.ipc_wrapper.CudaIPCWrapper` does for
CUDA. Device identity comes straight from the base class
(``get_device_properties(i).uuid``); torch_npu populates a per-chip UUID,
so no platform-specific discovery is needed.

The wrapper is **plane-aggregating**: engines may register a layer as one
tensor or as a sequence of paged planes (vLLM-Ascend hands per-layer
``(K, V)`` pairs and MLA/DSA ``(latent, rope[, dsa][, scale])`` tuples).
``wrap`` accepts either and :meth:`to_tensor` reconstructs the same value
-- bare tensor or tuple of planes -- so server-side format detection sees
the registered per-layer structure directly.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import Any, ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

#: Per-plane IPC record: ``(handle, dtype, shape, stride, storage_offset)``.
#: ``handle`` is the opaque payload of ``UntypedStorage._share_npu_()``;
#: everything else mirrors the base-class interface fields, per plane.
PlaneRecord = tuple[Any, torch.dtype, tuple[int, ...], tuple[int, ...], int]


class NpuIPCWrapper(DeviceIPCWrapper):
    """Plane-aggregating IPC wrapper for Ascend NPU KV tensors.

    This class exercises the documented multi-plane exception of
    :class:`DeviceIPCWrapper`: it does not populate the base-class
    singular fields (``handle`` / ``dtype`` / ``shape`` / ``stride`` /
    ``storage_offset``) but keeps one :data:`PlaneRecord` per plane in
    ``_plane_records`` plus the registered form in ``_bare``. Only the NPU
    wrapper implements the exception today.
    """

    #: ``torch.device.type`` this wrapper handles; also read by the
    #: server's ``_detect_device_type`` to route to the NPU device spec.
    device_type: ClassVar[str] = "npu"

    #: Per-plane ``(handle, dtype, shape, stride, storage_offset)`` records.
    _plane_records: tuple[PlaneRecord, ...]

    #: Whether the registered value was a bare tensor; ``to_tensor``
    #: restores the same form.
    _bare: bool

    @classmethod
    def wrap(cls, value: "torch.Tensor | Sequence[torch.Tensor]") -> "NpuIPCWrapper":
        """Factory used by :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            value: A single KV tensor, or one layer's paged planes as a
                sequence (e.g. vLLM-Ascend's ``(K, V)`` / ``(latent, rope)``
                tuples). Plane order is preserved across the wire.

        Returns:
            A new :class:`NpuIPCWrapper` aggregating ``value``'s planes for
            the multiprocess wire.
        """
        return cls(value)

    def __init__(self, value: "torch.Tensor | Sequence[torch.Tensor]") -> None:
        """Share every plane of ``value`` for cross-process reconstruction.

        Args:
            value: A single KV tensor or one layer's plane sequence. All
                planes must live on the same NPU device.

        Raises:
            RuntimeError: If a plane's storage cannot be shared through
                torch_npu storage IPC.
        """
        planes: list[torch.Tensor] = (
            [value] if isinstance(value, torch.Tensor) else list(value)
        )
        if not planes:
            raise ValueError(
                "NpuIPCWrapper requires at least one plane, got an empty sequence."
            )
        self._bare = isinstance(value, torch.Tensor)
        records: list[PlaneRecord] = []
        for plane in planes:
            storage = plane.untyped_storage()
            handle = storage._share_npu_()  # type: ignore[attr-defined]  # noqa: SLF001
            records.append(
                (
                    handle,
                    plane.dtype,
                    tuple(plane.shape),
                    tuple(plane.stride()),
                    int(plane.storage_offset()),
                )
            )
        self._plane_records = tuple(records)

        # Device identity is shared by every plane of the layer; probe the
        # first one. Base-class singular fields stay unset (see class
        # docstring: the documented multi-plane exception).
        device_index = planes[0].device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> "torch.Tensor | tuple[torch.Tensor, ...]":  # type: ignore[override]
        """Reconstruct the wrapped layer in this process.

        Note:
            ``torch.npu`` must be initialized before this function is
            called (guarded by ``torch_dev.init()`` at the call sites).

        Returns:
            The bare tensor, or the tuple of the layer's plane tensors in
            registration order (a 1-element sequence stays a 1-tuple).
        """
        device_index = self._get_device_index_from_uuid(self.device_uuid)
        tensors: list[torch.Tensor] = []
        for handle, dtype, shape, stride, storage_offset in self._plane_records:
            storage = torch.UntypedStorage._new_shared_npu(  # type: ignore[attr-defined]  # noqa: SLF001
                device_index, *handle[1:]
            )
            t = torch.empty((), device=device_index, dtype=dtype)
            t.set_(storage, storage_offset, shape, stride)
            tensors.append(t)
        if self._bare:
            return tensors[0]
        return tuple(tensors)

    def __eq__(self, other: object) -> bool:
        # Base-class equality compares the singular interface fields this
        # wrapper does not populate; compare the per-plane records instead.
        if not isinstance(other, NpuIPCWrapper):
            return False
        return (
            self._plane_records == other._plane_records
            and self._bare == other._bare
            and self.device_uuid == other.device_uuid
        )

    def __hash__(self) -> int:
        return hash((self._plane_records, self._bare, self.device_uuid))
