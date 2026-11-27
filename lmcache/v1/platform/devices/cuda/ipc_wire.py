# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC wire descriptors."""

# Standard
from typing import Annotated, Literal

# Third Party
import msgspec

# First Party
from lmcache.v1.platform.base.ipc_wire import IPCData, NonNegative


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
