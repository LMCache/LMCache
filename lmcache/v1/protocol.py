# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


@dataclass
class ClientMetaMessage:
    """
    Control message from LMCServerConnector to LMCacheServer
    """

    command: int
    key: CacheEngineKey
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size

    def serialize(self) -> bytes:
        key_str = self.key.to_string()
        assert len(key_str) <= MAX_KEY_LENGTH, (
            f"Key length {len(key_str)} exceeds maximum {MAX_KEY_LENGTH}"
        )

        # NOTE(Jiayi): 4 is the maximum dimension of memory object.
        # Pass in shape [x, 0, 0, 0] if it is a bytes memory object
        assert len(self.shape) == 4, "Shape dimension should be 4"

        packed_bytes = struct.pack(
            f"iiiiiiii{MAX_KEY_LENGTH}s",
            self.command,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
            key_str.encode().ljust(MAX_KEY_LENGTH),
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "ClientMetaMessage":
        command, length, fmt, dtype, shape0, shape1, shape2, shape3, key = (
            struct.unpack(f"iiiiiiii{MAX_KEY_LENGTH}s", s)
        )
        return ClientMetaMessage(
            command,
            CacheEngineKey.from_string(key.decode().strip()),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            torch.Size([shape0, shape1, shape2, shape3]),
        )

    @staticmethod
    def packlength() -> int:
        # NOTE: 8 is the number of integers
        return 4 * 8 + MAX_KEY_LENGTH


@dataclass
class ServerMetaMessage:
    """
    Control message from LMCacheServer to LMCServerConnector
    """

    code: int
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size

    def serialize(self) -> bytes:
        assert len(self.shape) == 4, "Shape dimension should be 4"
        packed_bytes = struct.pack(
            "iiiiiiii",
            self.code,
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
    def packlength() -> int:
        return 4 * 8

    @staticmethod
    def deserialize(s: bytes) -> "ServerMetaMessage":
        code, length, fmt, dtype, shape0, shape1, shape2, shape3 = struct.unpack(
            "iiiiiiii", s
        )
        return ServerMetaMessage(
            code,
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            torch.Size([shape0, shape1, shape2, shape3]),
        )
