# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, Union
import struct

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, parse_cache_key
from lmcache.v1.memory_management import MemoryFormat

MAX_KEY_LENGTH = 150


class ClientCommand(IntEnum):
    PUT = auto()
    GET = auto()
    EXIST = auto()
    LIST = auto()
    HEALTH = auto()


class ServerReturnCode(IntEnum):
    SUCCESS = 200
    FAIL = 400


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
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    fmt: MemoryFormat

    def _prepare_params(self):
        params = [self.length, int(self.fmt.value)]
        for shape, dtype in zip(self.shapes, self.dtypes, strict=True):
            assert len(shape) == 4, "Shape dimension should be 4"
            params.append(DTYPE_TO_INT[dtype])
            params.append(shape[0])
            params.append(shape[1])
            params.append(shape[2])
            params.append(shape[3])
        return params

    def serialize_into(self, buffer, fmt: str):
        params = self._prepare_params()
        assert len(fmt) == len(params)
        struct.pack_into(fmt, buffer, 0, *params)

    def serialize(self, fmt: str) -> bytes:
        params = self._prepare_params()
        assert len(fmt) == len(params)
        packed_bytes = struct.pack(fmt, *params)
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes, fmt: str) -> "RemoteMetadata":
        # length, fmt, dtype, shape0, shape1, shape2, shape3
        result = struct.unpack_from(fmt, s)
        assert len(fmt) == len(result)
        length = result[0]
        memory_fmt = MemoryFormat(result[1])
        shapes = []
        dtypes = []
        for i in range(2, len(result), 5):
            shapes.append(torch.Size(result[i + 1 : i + 5]))
            dtypes.append(INT_TO_DTYPE[result[i]])

        return RemoteMetadata(
            length,
            shapes,
            dtypes,
            memory_fmt,
        )


# TODO(Jiayi): Server and client message can be merged into one.


@dataclass
class ClientMetaMessage:
    """
    Request message from LMCache workers or servers.
    """

    command: ClientCommand
    key: Union[CacheEngineKey, LayerCacheEngineKey]
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
            self.command.value,
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
            ClientCommand(command),
            parse_cache_key(key.decode().strip()),
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

    code: ServerReturnCode
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        assert len(self.shape) == 4, "Shape dimension should be 4"
        packed_bytes = struct.pack(
            "iiiiiiiii",
            self.code.value,
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
            ServerReturnCode(code),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            torch.Size([shape0, shape1, shape2, shape3]),
            INT_TO_LOCATION[location],
        )
