# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import struct

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat

MAX_KEY_LENGTH = 150


class Constants:
    CLIENT_PUT = 1
    CLIENT_GET = 2
    CLIENT_EXIST = 3
    CLIENT_LIST = 4
    CLIENT_HEALTH = 5

    SERVER_SUCCESS = 200
    SERVER_FAIL = 400


DTYPE_TO_INT = {
    None: 0,
    torch.half: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float: 4,
    torch.float32: 4,
    torch.float64: 5,
    torch.double: 5,
    torch.uint8: 6,
    torch.float8_e4m3fn: 7,
    torch.float8_e5m2: 8,
}

INT_TO_DTYPE = {
    0: None,
    1: torch.half,
    2: torch.float16,
    3: torch.bfloat16,
    4: torch.float,
    5: torch.float64,
    6: torch.uint8,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}

# TODO (Jiayi): Add more backends
LOCATION_TO_INT = {
    None: 0,
    "LocalCPUBackend": 1,
    "LocalDiskBackend": 2,
}

INT_TO_LOCATION = {
    0: None,
    1: "LocalCPUBackend",
    2: "LocalDiskBackend",
}


@dataclass
class RemoteMetadata:
    length: int
    shape: torch.Size
    dtype: Optional[torch.dtype]
    fmt: MemoryFormat

    def serialize_into(self, buffer):
        assert len(self.shape) == 4, "Shape dimension should be 4"

        struct.pack_into(
            "iiiiiii",
            buffer,
            0,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
        )

    def serialize(self) -> bytes:
        # NOTE(Jiayi): 4 is the maximum dimension of memory object.
        # Pass in shape [x, 0, 0, 0] if it is a bytes memory object
        assert len(self.shape) == 4, "Shape dimension should be 4"

        packed_bytes = struct.pack(
            "iiiiiii",
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "RemoteMetadata":
        length, fmt, dtype, shape0, shape1, shape2, shape3 = struct.unpack_from(
            "iiiiiii", s
        )
        return RemoteMetadata(
            length,
            torch.Size([shape0, shape1, shape2, shape3]),
            INT_TO_DTYPE[dtype],
            MemoryFormat(fmt),
        )


# TODO(Jiayi): Server and client message can be merged into one.


@dataclass
class ClientMetaMessage:
    """
    Request message from LMCache workers or servers.
    """

    command: int
    key: CacheEngineKey
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        key_str = self.key.to_string()
        assert len(key_str) <= MAX_KEY_LENGTH, (
            f"Key length {len(key_str)} exceeds maximum {MAX_KEY_LENGTH}"
        )

        # NOTE(Jiayi): 4 is the maximum dimension of memory object.
        # Pass in shape [x, 0, 0, 0] if it is a bytes memory object
        assert len(self.shape) == 4, "Shape dimension should be 4"

        packed_bytes = struct.pack(
            f"iiiiiiiii{MAX_KEY_LENGTH}s",
            self.command,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            LOCATION_TO_INT[self.location],
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
            key_str.encode().ljust(MAX_KEY_LENGTH),
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "ClientMetaMessage":
        command, length, fmt, dtype, location, shape0, shape1, shape2, shape3, key = (
            struct.unpack(f"iiiiiiiii{MAX_KEY_LENGTH}s", s)
        )
        return ClientMetaMessage(
            command,
            CacheEngineKey.from_string(key.decode().strip()),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            torch.Size([shape0, shape1, shape2, shape3]),
            INT_TO_LOCATION[location],
        )

    @staticmethod
    def packlength() -> int:
        # NOTE: 9 is the number of integers
        return 4 * 9 + MAX_KEY_LENGTH


@dataclass
class ServerMetaMessage:
    """
    Reply message from LMCache workers or servers.
    """

    code: int
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        assert len(self.shape) == 4, "Shape dimension should be 4"
        packed_bytes = struct.pack(
            "iiiiiiiii",
            self.code,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
            LOCATION_TO_INT[self.location],
        )
        return packed_bytes

    @staticmethod
    def packlength() -> int:
        return 4 * 9

    @staticmethod
    def deserialize(s: bytes) -> "ServerMetaMessage":
        code, length, fmt, dtype, shape0, shape1, shape2, shape3, location = (
            struct.unpack("iiiiiiiii", s)
        )
        return ServerMetaMessage(
            code,
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            torch.Size([shape0, shape1, shape2, shape3]),
            INT_TO_LOCATION[location],
        )
