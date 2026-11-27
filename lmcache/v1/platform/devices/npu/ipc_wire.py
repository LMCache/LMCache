# SPDX-License-Identifier: Apache-2.0
"""NPU IPC wire descriptors."""

# Standard
from typing import Literal

# Third Party
import msgspec

# First Party
from lmcache.v1.platform.base.ipc_wire import TensorMetadata


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
