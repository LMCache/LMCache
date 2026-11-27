# SPDX-License-Identifier: Apache-2.0
"""Shared parts of the data-only device IPC descriptors for ZMQ and gRPC.

Each device package defines its own tagged descriptors on top of
:class:`IPCData` and :class:`TensorMetadata`.
"""

# Standard
from typing import Annotated, Literal, TypeAlias

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
