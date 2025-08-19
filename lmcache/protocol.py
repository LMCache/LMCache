# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import Enum, auto
import struct

MAX_KEY_LENGTH = 150


class Constants(Enum):
    CLIENT_PUT = auto()
    CLIENT_GET = auto()
    CLIENT_EXIST = auto()
    CLIENT_LIST = auto()

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
