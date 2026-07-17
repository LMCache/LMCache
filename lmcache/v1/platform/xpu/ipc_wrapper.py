# SPDX-License-Identifier: Apache-2.0
"""SYCL IPC wrapper for XPU tensors."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import ClassVar

# Third Party
import dpctl
import torch
from dpctl.memory import IPCMemoryHandle, MemoryUSMDevice

# First Party
from lmcache import torch_dev
from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper


@dataclass
class _OpenedMapping:
    raw_tensor: torch.Tensor
    usm_memory: object


class _TorchUSMArrayInterface:
    def __init__(self, tensor: torch.Tensor, queue: object) -> None:
        self._tensor = tensor
        self.__sycl_usm_array_interface__ = {
            "version": 1,
            "shape": (tensor.nbytes,),
            "typestr": "|u1",
            "data": (tensor.data_ptr(), False),
            "strides": None,
            "offset": 0,
            "syclobj": queue,
        }


class RawSyclIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper for XPU tensors backed by SYCL USM memory."""

    device_type: ClassVar[str] = "xpu"
    _is_default_wrapper: ClassVar[bool] = True
    _opened_ipc_mappings: ClassVar[dict[tuple[bytes, int, int], _OpenedMapping]] = {}

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "RawSyclIPCWrapper":
        """Create a raw SYCL IPC wrapper for ``tensor``."""
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != "xpu":
            raise ValueError(
                f"RawSyclIPCWrapper expects an XPU tensor, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError("RawSyclIPCWrapper requires contiguous XPU tensors")

        device_index = tensor.device.index
        if device_index is None:
            device_index = torch_dev.current_device()

        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        tensor_ptr = tensor.data_ptr()
        byte_offset = tensor_ptr - storage_ptr
        if byte_offset < 0:
            raise ValueError(
                "RawSyclIPCWrapper tensor data pointer precedes storage"
            )

        storage_tensor = self._storage_tensor_view(tensor, storage)
        usm_memory = self._to_usm_memory(storage_tensor, device_index)
        self._ipc_handle = IPCMemoryHandle(usm_memory).to_bytes()
        self._nbytes = tensor.nbytes
        self._storage_nbytes = storage.nbytes()
        self._byte_offset = byte_offset

        self.handle = self._ipc_handle
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())
        self.device_uuid = f"xpu:{device_index}"
        self.device_index = device_index

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process from a SYCL IPC handle."""
        storage_nbytes = getattr(self, "_storage_nbytes", self._nbytes)
        byte_offset = getattr(self, "_byte_offset", 0)
        cache_key = (self._ipc_handle, storage_nbytes, self.device_index)
        mapping = self._opened_ipc_mappings.get(cache_key)
        if mapping is None:
            device = self._ipc_device(self.device_index)
            usm_memory = IPCMemoryHandle.open(
                self._ipc_handle,
                device,
                nbytes=storage_nbytes,
            )
            raw = self._storage_tensor_from_usm_memory(
                usm_memory,
                storage_nbytes,
                self.device_index,
            )
            mapping = _OpenedMapping(
                raw_tensor=raw,
                usm_memory=usm_memory,
            )
            self._opened_ipc_mappings[cache_key] = mapping

        element_size = self.dtype.itemsize
        if byte_offset % element_size != 0:
            raise ValueError(
                f"SYCL IPC byte offset {byte_offset} is not aligned to {self.dtype}"
            )
        typed = mapping.raw_tensor.view(self.dtype)
        return torch.as_strided(
            typed,
            size=self.shape,
            stride=self.stride,
            storage_offset=byte_offset // element_size,
        )

    @classmethod
    def clear_opened_ipc_tensors(cls) -> None:
        """Close cached SYCL IPC mappings held by this process."""
        for mapping in cls._opened_ipc_mappings.values():
            IPCMemoryHandle.close_mapping(mapping.usm_memory)
        cls._opened_ipc_mappings.clear()

    @staticmethod
    def _storage_tensor_view(
        tensor: torch.Tensor,
        storage: torch.UntypedStorage,
    ) -> torch.Tensor:
        """Return a byte tensor spanning the tensor's full backing storage."""
        raw = torch.empty((), device=tensor.device, dtype=torch.uint8)
        raw.set_(storage, 0, (storage.nbytes(),), (1,))
        return raw

    @staticmethod
    def _ipc_device(device_index: int) -> object:
        devices = [
            device
            for device in dpctl.get_devices(device_type="gpu")
            if getattr(device, "has_aspect_ext_oneapi_ipc_memory", False)
        ]
        if device_index < 0 or device_index >= len(devices):
            raise RuntimeError(
                "XPU device does not support SYCL aspect::ext_oneapi_ipc_memory"
            )
        return devices[device_index]

    @classmethod
    def _to_usm_memory(cls, tensor: torch.Tensor, device_index: int) -> MemoryUSMDevice:
        queue = dpctl.SyclQueue(cls._ipc_device(device_index))
        return MemoryUSMDevice(_TorchUSMArrayInterface(tensor, queue))

    @staticmethod
    def _storage_tensor_from_usm_memory(
        usm_memory: object,
        nbytes: int,
        device_index: int,
    ) -> torch.Tensor:
        iface = usm_memory.__sycl_usm_array_interface__
        ptr = iface["data"][0]
        ptr = ptr if ptr < (1 << 63) else ptr - (1 << 64)
        device = torch.device("xpu", device_index)
        storage = torch._C._construct_storage_from_data_pointer(ptr, device, nbytes)
        raw = torch.empty((), device=device, dtype=torch.uint8)
        raw.set_(storage, 0, (nbytes,), (1,))
        return raw
