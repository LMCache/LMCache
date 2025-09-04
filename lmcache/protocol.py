# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import IntEnum, auto
import struct

MAX_KEY_LENGTH = 150


class ClientCommand(IntEnum):
    PUT = auto()
    GET = auto()
    EXIST = auto()
    LIST = auto()


class ServerReturnCode(IntEnum):
    # keep the same as HTTP status codes
    SUCCESS = 200
    FAIL = 400


@dataclass
class ClientMetaMessage:
    """
    Control message from LMCServerConnector to LMCacheServer
    """

    command: ClientCommand
    key: str
    length: int

    def serialize(self) -> bytes:
        assert len(self.key) <= MAX_KEY_LENGTH, (
            f"Key length {len(self.key)} exceeds maximum {MAX_KEY_LENGTH}"
        )
        packed_bytes = struct.pack(
            f"ii{MAX_KEY_LENGTH}s",
            self.command.value,
            self.length,
            self.key.encode().ljust(MAX_KEY_LENGTH),
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "ClientMetaMessage":
        command, length, key = struct.unpack(f"ii{MAX_KEY_LENGTH}s", s)
        return ClientMetaMessage(ClientCommand(command), key.decode().strip(), length)

    @staticmethod
    def packlength() -> int:
        return 4 * 2 + MAX_KEY_LENGTH


@dataclass
class ServerMetaMessage:
    """
    Control message from LMCacheServer to LMCServerConnector
    """

    code: ServerReturnCode
    length: int

    def serialize(self) -> bytes:
        packed_bytes = struct.pack("ii", self.code.value, self.length)
        return packed_bytes

    @staticmethod
    def packlength() -> int:
        return 8

    @staticmethod
    def deserialize(s: bytes) -> "ServerMetaMessage":
        code, length = struct.unpack("ii", s)
        return ServerMetaMessage(ServerReturnCode(code), length)
