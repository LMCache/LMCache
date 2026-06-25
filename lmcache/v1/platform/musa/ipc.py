# SPDX-License-Identifier: Apache-2.0
"""Optional MUSA IPC wrapper used by the Stage4 MP handle path.

The MUSA handle path is deliberately fail-closed. This module exposes a
``DeviceIPCWrapper`` subclass for the platform registry to auto-discover, but
it only constructs wrappers when the user explicitly enables the path and every
required handle-path capability is present.
"""

# Standard
from typing import ClassVar, Protocol, cast
import importlib
import inspect
import os
import threading

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)
from lmcache.v1.gpu_connector.utils import assert_contiguous
from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper

ENV_MUSA_HANDLE_TRANSFER = "LMCACHE_MUSA_HANDLE_TRANSFER"

_REQUIRED_TORCH_MUSA_IPC_SYMBOLS = (
    "ipc_get_mem_handle",
    "ipc_open_mem_handle",
)


class _TorchMusaIPCModule(Protocol):
    """TorchMUSA surface required by the Stage4.1 handle capability gate."""

    def ipc_get_mem_handle(self, tensor: torch.Tensor) -> object:
        """Export a MUSA tensor as a process-portable IPC handle."""

    def ipc_open_mem_handle(
        self,
        handle: bytes,
        nbytes: int,
        device_index: int,
    ) -> object:
        """Import a MUSA IPC handle in the current process."""


def is_musa_handle_transfer_enabled() -> bool:
    """Return whether users opted into the experimental MUSA handle path."""
    return os.environ.get(ENV_MUSA_HANDLE_TRANSFER, "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def is_torch_musa_available() -> bool:
    """Return whether TorchMUSA is importable and reports a visible device."""
    torch_musa = get_torch_musa_module()
    if torch_musa is None:
        return False
    is_available = getattr(torch_musa, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def get_torch_musa_module() -> object | None:
    """Return the TorchMUSA module when this PyTorch build exposes it.

    Returns:
        ``torch.musa`` when present, otherwise ``None``.
    """
    if not hasattr(torch, "musa"):
        # torch_musa registers torch.musa as an import side effect.
        try:
            importlib.import_module("torch_musa")
        except ImportError:
            pass
    if not hasattr(torch, "musa"):
        return None
    return torch.musa  # type: ignore[attr-defined]


def check_torch_musa_ipc_support(torch_musa: object) -> bool:
    """Return whether TorchMUSA exposes memory IPC primitives."""
    return all(
        callable(getattr(torch_musa, symbol, None))
        for symbol in _REQUIRED_TORCH_MUSA_IPC_SYMBOLS
    )


def check_torch_musa_event_support(torch_musa: object) -> bool:
    """Return whether TorchMUSA exposes IPC event ordering primitives."""
    event_cls = getattr(torch_musa, "Event", None)
    if event_cls is None or not hasattr(event_cls, "from_ipc_handle"):
        return False
    return _has_interprocess_parameter(event_cls) or _has_interprocess_parameter(
        getattr(event_cls, "__new__", None)
    )


def is_musa_block_transfer_available() -> bool:
    """Return whether server-side MUSA block transfer is production-ready.

    Stage4.1 only adds the safe integration boundary. Until Stage4.3 wires and
    validates the server-side MUSA block-transfer primitive, forced MUSA handle
    mode must remain unavailable even if TorchMUSA memory/event IPC exists.
    """
    return False


def is_musa_handle_transfer_available() -> bool:
    """Return whether forced MUSA MP handle mode may be selected.

    Availability requires all four conditions:
    - explicit user opt-in through :data:`ENV_MUSA_HANDLE_TRANSFER`;
    - a visible TorchMUSA runtime;
    - TorchMUSA memory IPC and interprocess event support;
    - a validated server-side MUSA block-transfer primitive.
    """
    if not is_musa_handle_transfer_enabled():
        return False
    if not is_torch_musa_available():
        return False
    torch_musa = get_torch_musa_module()
    return (
        torch_musa is not None
        and check_torch_musa_ipc_support(torch_musa)
        and check_torch_musa_event_support(torch_musa)
        and is_musa_block_transfer_available()
    )


class MusaIPCWrapper(DeviceIPCWrapper):
    """Wire-compatible wrapper for MUSA memory IPC handles.

    The class subclasses :class:`DeviceIPCWrapper` so it shares the
    device-agnostic ``KVCache = list[DeviceIPCWrapper]`` msgspec wire type with
    CUDA and CPU wrappers. Pickle preserves the subclass identity, and the
    receiver calls this class's :meth:`to_tensor` implementation to reconstruct
    a MUSA tensor.
    """

    #: ``torch.device.type`` this wrapper handles (used by auto-discovery).
    device_type: ClassVar[str] = "musa"

    #: Marked ``True`` so auto-discovery picks this as the default factory.
    _is_default_wrapper: ClassVar[bool] = True

    _discovered_device_mapping: dict[str, int] = {}
    _device_mapping_lock = threading.Lock()

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "MusaIPCWrapper":
        """Factory used by the platform registry auto-discovery.

        Args:
            tensor: A MUSA tensor to export through TorchMUSA IPC.

        Returns:
            A new :class:`MusaIPCWrapper` wrapping ``tensor`` for the
            multiprocess wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        """Export a MUSA tensor through the TorchMUSA IPC runtime.

        Args:
            tensor: A contiguous MUSA tensor to export.

        Raises:
            ValueError: If ``tensor`` is not a MUSA tensor or cannot be
                represented as a zero-offset contiguous view.
            RuntimeError: If the Stage4 MUSA handle capability is unavailable.
        """
        tensor_view = attempt_permute_to_contiguous_view(tensor)
        if (
            not isinstance(tensor_view, torch.Tensor)
            or tensor_view.device.type != "musa"
        ):
            raise ValueError("expected a MUSA tensor for MusaIPCWrapper")
        tensor = tensor_view
        assert_contiguous(tensor)

        module = _torch_musa_module_if_ready()
        if module is None:
            raise RuntimeError(
                "MUSA IPC handle transfer is not available. Set "
                f"{ENV_MUSA_HANDLE_TRANSFER}=1 with compatible TorchMUSA "
                "memory/event IPC support and a validated MUSA block-transfer "
                "backend, or use MP transfer mode 'engine_driven' or 'auto'."
            )

        self._musa_ipc_handle = _coerce_ipc_handle(module.ipc_get_mem_handle(tensor))
        self._nbytes = tensor.untyped_storage().nbytes()

        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(
            0 if device_index is None else device_index
        )

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the MUSA tensor in this process."""
        module = _torch_musa_module_if_ready()
        if module is None:
            raise RuntimeError(
                "MUSA IPC handle transfer is not available in the receiver process."
            )

        device_index = self._get_device_index_from_uuid(self.device_uuid)
        raw = module.ipc_open_mem_handle(
            self._musa_ipc_handle,
            self._nbytes,
            device_index,
        )
        if isinstance(raw, torch.Tensor):
            tensor = raw
        else:
            tensor = torch.from_dlpack(raw)

        if tensor.dtype == torch.uint8 and self.dtype != torch.uint8:
            tensor = tensor.view(self.dtype)
        return tensor.as_strided(self.shape, self.stride, self.storage_offset)

    @classmethod
    def _get_device_uuid(cls, device_index: int) -> str:
        """Return a stable MUSA device identifier for ``device_index``."""
        props = torch.musa.get_device_properties(device_index)  # type: ignore[attr-defined]
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            return str(uuid)
        pci_bus_id = getattr(props, "pci_bus_id", None)
        if pci_bus_id is not None:
            return str(pci_bus_id)
        name = getattr(props, "name", "musa")
        return f"{name}:{device_index}"

    @classmethod
    def _discover_devices(cls) -> None:
        """Discover visible MUSA devices and cache their identifiers."""
        if not is_torch_musa_available():
            return

        with cls._device_mapping_lock:
            if cls._discovered_device_mapping:
                return

            for i in range(torch.musa.device_count()):  # type: ignore[attr-defined]
                device_uuid = cls._get_device_uuid(i)
                cls._discovered_device_mapping[device_uuid] = i

    @classmethod
    def _get_device_index_from_uuid(cls, device_uuid: str) -> int:
        """Resolve the sender's MUSA device identifier in this process."""
        cls._discover_devices()

        with cls._device_mapping_lock:
            device_index = cls._discovered_device_mapping.get(device_uuid)

        if device_index is None:
            raise RuntimeError(
                f"MUSA device UUID {device_uuid} not found in visible devices. "
                "Please make sure the worker and server see the same MUSA devices."
            )
        return device_index


def _torch_musa_module_if_ready() -> _TorchMusaIPCModule | None:
    """Return TorchMUSA when all handle-path capability gates pass."""
    if not is_musa_handle_transfer_enabled():
        return None
    if not is_torch_musa_available():
        return None
    module = get_torch_musa_module()
    if (
        module is None
        or not check_torch_musa_ipc_support(module)
        or not check_torch_musa_event_support(module)
        or not is_musa_block_transfer_available()
    ):
        return None
    return cast(_TorchMusaIPCModule, module)


def _coerce_ipc_handle(handle: object) -> bytes:
    """Normalize a MUSA IPC handle into bytes for pickle/msgspec transport."""
    if isinstance(handle, bytes):
        return handle
    if isinstance(handle, bytearray):
        return bytes(handle)
    if isinstance(handle, memoryview):
        return handle.tobytes()
    raise TypeError("MUSA IPC handle must be bytes-like")


def _has_interprocess_parameter(obj: object) -> bool:
    """Return whether ``obj`` accepts the ``interprocess`` event parameter."""
    if not callable(obj):
        return False
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    return "interprocess" in signature.parameters
