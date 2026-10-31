# SPDX-License-Identifier: Apache-2.0
"""Data-only device IPC descriptors shared by ZMQ and gRPC."""

# Standard
from typing import Annotated, Literal, TypeAlias, cast

# Third Party
import msgspec
import torch

NonNegative: TypeAlias = Annotated[int, msgspec.Meta(ge=0)]
DTYPES = {
    str(value).removeprefix("torch."): value
    for value in vars(torch).values()
    if isinstance(value, torch.dtype)
}


class TensorMetadata(msgspec.Struct, forbid_unknown_fields=True):
    """Tensor view metadata; decoding never imports or opens device resources."""

    dtype: str
    shape: tuple[NonNegative, ...]
    stride: tuple[NonNegative, ...]
    storage_offset: NonNegative
    device_uuid: str

    def byte_span(self) -> int:
        """Return the exclusive byte end of the tensor view within storage."""
        if 0 in self.shape:
            return 0
        return DTYPES[self.dtype].itemsize * (
            self.storage_offset
            + 1
            + sum(
                (size - 1) * stride
                for size, stride in zip(self.shape, self.stride, strict=False)
            )
        )

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError("Unsupported IPC tensor dtype")
        if len(self.shape) != len(self.stride):
            raise ValueError("IPC shape and stride ranks must match")


class IPCData(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Versioned wire descriptor; subclasses have fixed, local type tags."""

    version: Literal[1]
    metadata: TensorMetadata


class CudaIPCData(IPCData, tag="cuda"):
    """PyTorch shared CUDA storage descriptor."""

    handle: tuple[
        int,
        bytes | None,
        NonNegative,
        NonNegative,
        bytes | None,
        NonNegative,
        bytes | None,
        bool,
    ]

    def __post_init__(self) -> None:
        if self.metadata.byte_span() > self.handle[2]:
            raise ValueError("IPC view exceeds CUDA storage")


class RawCudaIPCData(IPCData, tag="cuda_raw"):
    """Driver IPC memory descriptor with a byte range."""

    ipc_handle: Annotated[bytes, msgspec.Meta(min_length=64, max_length=64)]
    alloc_offset: NonNegative
    nbytes: NonNegative

    def __post_init__(self) -> None:
        if self.metadata.byte_span() > self.nbytes:
            raise ValueError("IPC view exceeds raw CUDA byte range")


class VmmCudaIPCData(IPCData, tag="cuda_vmm"):
    """VMM allocation descriptor; process-local file descriptors are excluded."""

    export_id: Annotated[bytes, msgspec.Meta(min_length=16, max_length=16)]
    kind: Literal["posix_fd", "fabric"]
    fabric_blob: bytes | None
    alloc_size: NonNegative
    alloc_offset: NonNegative
    nbytes: NonNegative

    def __post_init__(self) -> None:
        if self.metadata.byte_span() > self.nbytes:
            raise ValueError("IPC view exceeds VMM byte range")
        if self.alloc_offset + self.nbytes > self.alloc_size:
            raise ValueError("IPC view exceeds VMM allocation")
        if (self.kind == "fabric") != (self.fabric_blob is not None):
            raise ValueError("IPC fabric handle does not match allocation kind")


class CpuShmIPCData(IPCData, tag="cpu_shm"):
    """Named CPU shared-memory descriptor."""

    shm_name: str
    nbytes: NonNegative

    def __post_init__(self) -> None:
        if self.metadata.byte_span() > self.nbytes:
            raise ValueError("IPC view exceeds CPU shared-memory segment")


class MusaIPCData(IPCData, tag="musa"):
    """MUSA handle restricted to MessagePack primitive values."""

    handle: object

    def __post_init__(self) -> None:
        validate_musa_handle(self.handle)


def validate_musa_handle(value: object) -> None:
    """Reject opaque objects and extensions in a MUSA handle.

    Args:
        value: A handle composed of scalars, lists and string-keyed dictionaries.

    Raises:
        ValueError: If the handle would require Python object reconstruction.
    """
    if type(value) in (type(None), bool, int, float, str, bytes):
        return
    if type(value) is list:
        for item in cast(list[object], value):
            validate_musa_handle(item)
        return
    if type(value) is dict and all(
        type(key) is str for key in cast(dict[object, object], value)
    ):
        for item in cast(dict[str, object], value).values():
            validate_musa_handle(item)
        return
    raise ValueError("MUSA IPC handles must contain only MessagePack primitive values")


class NpuPlaneData(msgspec.Struct, forbid_unknown_fields=True):
    """One NPU storage handle and its tensor view."""

    handle: tuple[object, ...]
    metadata: TensorMetadata

    def __post_init__(self) -> None:
        if not self.handle or any(
            type(item) not in (int, bytes, str, bool, type(None))
            for item in self.handle
        ):
            raise ValueError("NPU IPC handles must contain only scalar values")


class NpuIPCData(msgspec.Struct, tag="npu", forbid_unknown_fields=True):
    """NPU layer descriptor preserving bare tensors versus plane sequences."""

    version: Literal[1]
    bare: bool
    planes: tuple[NpuPlaneData, ...]

    def __post_init__(self) -> None:
        if not self.planes or (self.bare and len(self.planes) != 1):
            raise ValueError("Invalid NPU IPC plane count")
        device_uuid = self.planes[0].metadata.device_uuid
        if any(plane.metadata.device_uuid != device_uuid for plane in self.planes):
            raise ValueError("NPU IPC planes must share a device")


WireIPCData: TypeAlias = (
    CudaIPCData
    | RawCudaIPCData
    | VmmCudaIPCData
    | CpuShmIPCData
    | MusaIPCData
    | NpuIPCData
)

# Only these local implementations can be selected by a decoded descriptor.
WRAPPER_TYPES: dict[type[msgspec.Struct], tuple[str, str]] = {
    CudaIPCData: ("cuda.ipc_wrapper", "CudaIPCWrapper"),
    RawCudaIPCData: ("cuda.ipc_wrapper", "RawCudaIPCWrapper"),
    VmmCudaIPCData: ("cuda.ipc_wrapper", "VmmCudaIPCWrapper"),
    CpuShmIPCData: ("cpu.shm", "CpuShmTensorWrapper"),
    MusaIPCData: ("musa.ipc_wrapper", "MusaIPCWrapper"),
    NpuIPCData: ("npu.ipc_wrapper", "NpuIPCWrapper"),
}
