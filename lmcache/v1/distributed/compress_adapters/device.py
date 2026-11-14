# SPDX-License-Identifier: Apache-2.0
"""Compression-local device values for GPU decompression requests.

These values adapt LMCache's existing platform and cache-context APIs without
exposing CUDA, HIP, nvCOMP, or hipCOMP types. They intentionally support only
contiguous :class:`torch.Tensor` buffers in the first interface slice.
``MemoryObj`` support requires an explicit allocator lease and is therefore
deferred until the production integration owns that lifetime correctly.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.v1.platform import current_device_spec
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext


def _require_nonnegative_int(name: str, value: int) -> None:
    """Validate an integer bound without accepting booleans as integers.

    Args:
        name: Field name used in validation errors.
        value: Integer that must be zero or greater.

    Returns:
        None when the value is valid.

    Raises:
        TypeError: If ``value`` is not an exact ``int``, including ``bool``.
        ValueError: If ``value`` is negative.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


def _normalized_device_index(device: torch.device, spec: DeviceSpec) -> int:
    """Resolve an index-free device through its selected Torch runtime.

    Args:
        device: Torch device whose type already matches ``spec``.
        spec: Selected LMCache backend providing the Torch module name.

    Returns:
        The explicit index, zero for an index-free CPU device, or the runtime's
        nonnegative current accelerator index.

    Raises:
        TypeError: If the runtime returns an index that is not an exact int.
        ValueError: If the runtime has no current-device query or returns a
            negative index.
        RuntimeError: If the runtime cannot initialize or query the device.
        MemoryError: If the runtime cannot allocate required host memory.

    Notes:
        Exceptions raised by the runtime query propagate unchanged.
    """
    if device.index is not None:
        return device.index
    if device.type == "cpu":
        return 0

    torch_module = getattr(torch, spec.torch_module_name, None)
    current_device = getattr(torch_module, "current_device", None)
    if not callable(current_device):
        raise ValueError(
            f"device {device} has no index and backend {spec.backend_name!r} "
            "cannot resolve a current device"
        )
    device_index = current_device()
    _require_nonnegative_int("device_index", device_index)
    return device_index


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """Normalized identity of a concrete LMCache device backend.

    Args:
        device_type: Torch-facing device type, such as ``"cuda"`` or
            ``"cpu"``.
        backend_name: LMCache :class:`DeviceSpec` backend selector, such as
            ``"cuda"`` or ``"rocm"``.
        device_index: Nonnegative concrete device index.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a name is empty or ``device_index`` is negative.

    Notes:
        CUDA and ROCm intentionally share Torch's ``"cuda"`` device type.
        ``backend_name`` preserves the distinction already represented by
        LMCache's device registry.
    """

    device_type: str
    backend_name: str
    device_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("device_type", self.device_type),
            ("backend_name", self.backend_name),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str, got {type(value).__name__}")
            if not value:
                raise ValueError(f"{name} must not be empty")
        _require_nonnegative_int("device_index", self.device_index)

    @classmethod
    def from_device(cls, device: torch.device) -> DeviceIdentity:
        """Derive an identity from a Torch device and detected LMCache spec.

        Args:
            device: Torch device owned by the buffer or cache context.

        Returns:
            The normalized device type, backend name, and concrete index.

        Raises:
            TypeError: If ``device`` is not a :class:`torch.device` or the
                runtime returns an index that is not an exact int.
            ValueError: If the selected LMCache backend does not match the
                device, has no current-device query for an implicit index,
                or returns a negative index.
            RuntimeError: If the runtime cannot initialize or query the device.
            MemoryError: If the runtime cannot allocate required host memory.

        Notes:
            Resolving an implicit accelerator index may initialize the runtime.
            Exceptions from that query propagate unchanged so callers can
            distinguish runtime failures from invalid device arguments.
        """
        if not isinstance(device, torch.device):
            raise TypeError(
                f"device must be a torch.device, got {type(device).__name__}"
            )

        spec = current_device_spec
        if spec.device_type != device.type:
            raise ValueError(
                f"detected backend {spec.backend_name!r} handles device type "
                f"{spec.device_type!r}, not {device.type!r}"
            )
        return cls(
            device_type=device.type,
            backend_name=spec.backend_name,
            device_index=_normalized_device_index(device, spec),
        )


@dataclass(frozen=True, slots=True, init=False)
class DeviceBufferRange:
    """Validated byte range within one contiguous tensor view.

    Args:
        tensor: Contiguous tensor that owns the addressable view. Packed
            quantized dtypes are not supported.
        byte_offset: Range start relative to ``tensor.data_ptr()``.
        byte_length: Number of addressable bytes in the range.

    Raises:
        TypeError: If the owner or range values have the wrong type.
        ValueError: If the tensor is non-contiguous, uses a packed quantized
            dtype, or the range falls outside
            ``tensor.numel() * tensor.element_size()``.

    Notes:
        This immutable value retains the tensor and snapshots its address and
        capacity. It is not yet a lease: callers must not resize or reuse the
        allocation while native work is active. The later submission layer
        will own that lease and revalidate the live native address.

        Equality and hashing describe the snapshotted device byte range. The
        retained tensor is excluded because tensor equality is element-wise,
        and it is omitted from representations to avoid printing buffer data.
    """

    tensor: torch.Tensor = field(repr=False, compare=False)
    device: DeviceIdentity
    byte_offset: int
    byte_length: int
    capacity: int
    base_address: int

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        byte_offset: int,
        byte_length: int,
    ) -> DeviceBufferRange:
        """Construct a bounded byte range from a contiguous tensor.

        Args:
            tensor: Contiguous tensor whose exact view provides the addressable
                capacity. Packed quantized dtypes are not supported.
            byte_offset: Nonnegative byte offset into the tensor view.
            byte_length: Nonnegative number of represented bytes.

        Returns:
            A frozen range retaining the tensor owner and derived metadata.

        Raises:
            TypeError: If ``tensor`` or either range value has the wrong type.
            ValueError: If the tensor is non-contiguous, uses a packed quantized
                dtype, or the requested range exceeds the tensor view.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"tensor must be a torch.Tensor, got {type(tensor).__name__}"
            )
        _require_nonnegative_int("byte_offset", byte_offset)
        _require_nonnegative_int("byte_length", byte_length)
        if tensor.dtype in (torch.quint4x2, torch.quint2x4):
            raise ValueError("packed quantized tensors are not supported")
        if not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous")

        capacity = tensor.numel() * tensor.element_size()
        if byte_offset > capacity or byte_length > capacity - byte_offset:
            raise ValueError(
                f"byte range [{byte_offset}, {byte_offset + byte_length}) "
                f"exceeds tensor capacity {capacity}"
            )

        instance = object.__new__(cls)
        object.__setattr__(instance, "tensor", tensor)
        object.__setattr__(
            instance,
            "device",
            DeviceIdentity.from_device(tensor.device),
        )
        object.__setattr__(instance, "byte_offset", byte_offset)
        object.__setattr__(instance, "byte_length", byte_length)
        object.__setattr__(instance, "capacity", capacity)
        object.__setattr__(instance, "base_address", tensor.data_ptr())
        return instance

    @property
    def address(self) -> int:
        """Return the snapshotted native address at the range start."""
        return self.base_address + self.byte_offset

    @property
    def end_offset(self) -> int:
        """Return the exclusive range end relative to the tensor view."""
        return self.byte_offset + self.byte_length


@dataclass(frozen=True, slots=True, init=False)
class DeviceExecutionContext:
    """Device identity and opaque stream retained from a cache context.

    Args:
        cache_context: Existing LMCache cache context providing public
            ``device`` and ``stream`` properties.

    Raises:
        TypeError: If the object does not expose the required public API.
        ValueError: If its stream is ``None`` or its device cannot be matched
            to the selected LMCache backend and a concrete index.
        RuntimeError: If resolving an implicit index fails in the runtime.
        MemoryError: If the runtime cannot allocate required host memory.

    Notes:
        The whole cache context is retained because it owns the stream and
        staging tensors. Retention does not prevent a concurrent ``close()``;
        production integration must quiesce handlers before context shutdown.
        The stream is an opaque ``object``; only a matching backend interprets
        its concrete type.
    """

    cache_context: BaseCacheContext
    device: DeviceIdentity
    stream: object

    @classmethod
    def from_cache_context(
        cls,
        cache_context: BaseCacheContext,
    ) -> DeviceExecutionContext:
        """Adapt an existing cache context without interpreting its stream.

        Args:
            cache_context: Context exposing public ``device`` and ``stream``
                properties.

        Returns:
            A frozen adapter retaining the context, identity, and opaque
            stream.

        Raises:
            TypeError: If the required public properties are absent or the
                device or resolved index has the wrong type.
            ValueError: If the stream is ``None`` or the device does not match
                the selected LMCache backend and a concrete index.
            RuntimeError: If resolving an implicit index fails in the runtime.
            MemoryError: If the runtime cannot allocate required host memory.

        Notes:
            Device resolution uses :meth:`DeviceIdentity.from_device` and
            propagates runtime-query exceptions unchanged.
        """
        try:
            device = cache_context.device
            stream = cache_context.stream
        except AttributeError as exc:
            raise TypeError(
                "cache_context must expose public device and stream properties"
            ) from exc
        if not isinstance(device, torch.device):
            raise TypeError(
                "cache_context.device must be a torch.device, got "
                f"{type(device).__name__}"
            )
        if stream is None:
            raise ValueError("cache_context.stream must not be None")

        instance = object.__new__(cls)
        object.__setattr__(instance, "cache_context", cache_context)
        object.__setattr__(instance, "device", DeviceIdentity.from_device(device))
        object.__setattr__(instance, "stream", stream)
        return instance
