# SPDX-License-Identifier: Apache-2.0
"""Base class for device IPC wrappers.

:class:`DeviceIPCWrapper` is the abstract base for KV-cache IPC wrapper
implementations.  Every concrete wrapper (e.g. :class:`~.cpu.shm.CpuShmTensorWrapper`,
:class:`~.cuda.ipc_wrapper.CudaIPCWrapper`) subclasses it so they share
the single msgspec ext code (1). A data-only descriptor selects the
concrete implementation from a local allowlist.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Tuple, cast
import importlib
import threading

# Third Party
import msgspec
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.platform.base.ipc_wire import (
    DTYPES,
    WRAPPER_TYPES,
    TensorMetadata,
    WireIPCData,
)


class DeviceIPCWrapper:
    """Base class for KV-cache IPC wrapper.

    Holds the device-agnostic mechanism shared by all transports: the
    interface fields (``dtype``/``shape``/``stride``/``storage_offset``/
    ``device_uuid``), UUID<->ordinal discovery via the ``torch_dev``
    abstraction, equality, and data-only (de)serialization.

    Every wire-level wrapper subclasses this so they share the single
    msgspec ext code (1) registered for ``DeviceIPCWrapper``. Explicit
    descriptor tags preserve concrete wrapper identity without unpickling.

    Subclasses implement ``__init__`` (populate the interface fields from a
    tensor) and ``to_tensor`` (reconstruct the tensor from the handle).

    The default wrapper for each device is bound to that device's
    :class:`~lmcache.v1.platform.base.device_spec.DeviceSpec` via
    :attr:`~lmcache.v1.platform.base.device_spec.DeviceSpec.ipc_wrapper_cls`;
    :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory` reads that
    binding and returns the wrapper's ``wrap`` classmethod so callers can
    dispatch by ``tensor.device.type`` without any if/elif chain.
    """

    # Interface fields populated by each concrete subclass's
    # ``__init__``.  Declared here so the base-class ``__eq__`` (and
    # type-checkers) can see them; ``handle`` is intentionally typed as
    # ``Any`` because each backend stores a different opaque payload
    # (``tuple`` for CUDA shared-storage IPC, ``None`` for the SHM /
    # raw-CUDA paths that override ``to_tensor``).
    handle: Any
    dtype: torch.dtype
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    device_uuid: str

    _discovered_device_mapping: dict[str, int] = {}
    _device_mapping_lock = threading.Lock()

    @classmethod
    def _get_device_uuid(cls, device_index: int) -> str:
        """Get the UUID of a device given its index."""
        return str(torch_dev.get_device_properties(device_index).uuid)

    @classmethod
    def _discover_devices(cls) -> None:
        """Discover all available accelerator devices and map their UUIDs
        to the physical device ordinals.
        """
        if not torch_dev.is_available():
            return

        num_devices = torch_dev.device_count()
        with DeviceIPCWrapper._device_mapping_lock:
            if DeviceIPCWrapper._discovered_device_mapping:
                return  # Already discovered

            for i in range(num_devices):
                device_uuid = cls._get_device_uuid(i)
                DeviceIPCWrapper._discovered_device_mapping[device_uuid] = i

    @classmethod
    def _get_device_index_from_uuid(cls, device_uuid: str) -> int:
        """Get the physical device ordinal from its UUID."""
        cls._discover_devices()

        with DeviceIPCWrapper._device_mapping_lock:
            device_index = DeviceIPCWrapper._discovered_device_mapping.get(
                device_uuid, None
            )

        if device_index is None:
            raise RuntimeError(
                f"Device UUID {device_uuid} not found in the discovered "
                "devices. Please make sure the process can see all the "
                "accelerator devices"
            )
        return device_index

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process from the IPC handle.

        Subclasses implement the transport-specific reconstruction.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release process-local resources held for reconstruction.

        Called by cache-context teardown once the tensors returned by
        :meth:`to_tensor` are no longer used. The default is a no-op.
        """
        return None

    def __eq__(self, other: object) -> bool:
        # ``isinstance`` first so type-checkers can narrow ``other`` to
        # ``DeviceIPCWrapper`` before we touch its attributes; the
        # exact-type check that follows then enforces that, e.g., a
        # ``CudaIPCWrapper`` is never considered equal to a
        # ``RawCudaIPCWrapper`` even though they share a base class.
        if not isinstance(other, DeviceIPCWrapper):
            return False
        if type(self) is not type(other):
            return False
        return (
            self.handle == other.handle
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.stride == other.stride
            and self.storage_offset == other.storage_offset
            and self.device_uuid == other.device_uuid
        )

    def wire_metadata(self) -> TensorMetadata:
        """Return the data-only tensor view metadata for this wrapper."""
        return TensorMetadata(
            dtype=str(self.dtype).removeprefix("torch."),
            shape=self.shape,
            stride=self.stride,
            storage_offset=self.storage_offset,
            device_uuid=self.device_uuid,
        )

    def restore_metadata(self, metadata: TensorMetadata) -> None:
        """Restore validated tensor metadata without opening IPC resources."""
        self.dtype = DTYPES[metadata.dtype]
        self.shape = metadata.shape
        self.stride = metadata.stride
        self.storage_offset = metadata.storage_offset
        self.device_uuid = metadata.device_uuid

    def to_wire(self) -> WireIPCData:
        """Return an explicit descriptor for a supported IPC implementation."""
        raise NotImplementedError

    @classmethod
    def from_wire(cls, data: WireIPCData) -> "DeviceIPCWrapper":
        """Restore a wrapper from its validated descriptor without opening it."""
        raise NotImplementedError

    @staticmethod
    def Serialize(obj: "DeviceIPCWrapper") -> bytes:
        """Encode an IPC descriptor as MessagePack, without Python objects.

        Args:
            obj: A supported concrete IPC wrapper.

        Returns:
            Versioned descriptor bytes for both MP transports.

        Raises:
            ValueError: If the wrapper is not an explicitly supported type.
        """
        data = obj.to_wire()
        module, name = WRAPPER_TYPES[type(data)]
        if (type(obj).__module__, type(obj).__name__) != (
            f"lmcache.v1.platform.devices.{module}",
            name,
        ):
            raise ValueError("Unsupported IPC wrapper type")
        return msgspec.msgpack.encode(data)

    @staticmethod
    def Deserialize(data: bytes) -> "DeviceIPCWrapper":
        """Decode a data-only IPC descriptor. Legacy pickle is rejected.

        Args:
            data: Bytes produced by :meth:`Serialize`.

        Returns:
            A wrapper with no receiver-local resources opened.

        Raises:
            msgspec.DecodeError: If the descriptor is malformed or unsupported.
        """
        descriptor = msgspec.msgpack.decode(data, type=WireIPCData)
        module, name = WRAPPER_TYPES[type(descriptor)]
        # Module and class names come exclusively from the local allowlist.
        wrapper_cls = cast(
            type[DeviceIPCWrapper],
            getattr(
                importlib.import_module(f"lmcache.v1.platform.devices.{module}"), name
            ),
        )
        return wrapper_cls.from_wire(descriptor)
