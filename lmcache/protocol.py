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
import struct

MAX_KEY_LENGTH = 150


class Constants:
    CLIENT_PUT = 1
    CLIENT_GET = 2
    CLIENT_EXIST = 3
    CLIENT_LIST = 4

    SERVER_SUCCESS = 200
    SERVER_FAIL = 400


@dataclass
class ClientMetaMessage:
    """
    Control message from LMCServerConnector to LMCacheServer
    """

    command: int
    key: str
    length: int

    def serialize(self) -> bytes:
        assert len(self.key) <= MAX_KEY_LENGTH, (
            f"Key length {len(self.key)} exceeds maximum {MAX_KEY_LENGTH}"
        )
        packed_bytes = struct.pack(
            f"ii{MAX_KEY_LENGTH}s",
            self.command,
            self.length,
            self.key.encode().ljust(MAX_KEY_LENGTH),
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "ClientMetaMessage":
        command, length, key = struct.unpack(f"ii{MAX_KEY_LENGTH}s", s)
        return ClientMetaMessage(command, key.decode().strip(), length)

    @staticmethod
    def packlength() -> int:
        return 4 * 2 + MAX_KEY_LENGTH


@dataclass
class ServerMetaMessage:
    """
    Control message from LMCacheServer to LMCServerConnector
    """

    code: int
    length: int

    def serialize(self) -> bytes:
        packed_bytes = struct.pack("ii", self.code, self.length)
        return packed_bytes

    @staticmethod
    def packlength() -> int:
        return 8

    @staticmethod
    def deserialize(s: bytes) -> "ServerMetaMessage":
        code, length = struct.unpack("ii", s)
        return ServerMetaMessage(code, length)
