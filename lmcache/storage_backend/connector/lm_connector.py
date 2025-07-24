# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import socket
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.protocol import ClientMetaMessage, Constants, ServerMetaMessage
from lmcache.storage_backend.connector.base_connector import (
    RemoteBytesConnector,
)
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


# TODO: performance optimization for this class, consider using C/C++/Rust
# for communication + deserialization
class LMCServerConnector(RemoteBytesConnector):
    def __init__(self, host, port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.socket_lock = threading.Lock()

    def receive_all(self, n):
        received = 0
        buffer = bytearray(n)
        view = memoryview(buffer)

        while received < n:
            num_bytes = self.client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                return None
            received += num_bytes

        return buffer

    def send_all(self, data):
        """
        Thread-safe function to send the data
        """
        with self.socket_lock:
            self.client_socket.sendall(data)

    def exists(self, key: str) -> bool:
        logger.debug("Call to exists()!")
        self.send_all(ClientMetaMessage(Constants.CLIENT_EXIST, key, 0).serialize())
        response = self.client_socket.recv(ServerMetaMessage.packlength())
        return ServerMetaMessage.deserialize(response).code == Constants.SERVER_SUCCESS

    def set(self, key: str, obj: bytes):  # type: ignore[override]
        logger.debug("Call to set()!")
        self.send_all(
            ClientMetaMessage(Constants.CLIENT_PUT, key, len(obj)).serialize()
        )
        self.send_all(obj)
        # response = self.client_socket.recv(ServerMetaMessage.packlength())
        # if ServerMetaMessage.deserialize(response).code
        #   != Constants.SERVER_SUCCESS:
        #    raise RuntimeError(f"Failed to set key:
        # {ServerMetaMessage.deserialize(response).code}")

    @_lmcache_nvtx_annotate
    def get(self, key: str) -> Optional[bytes]:
        self.send_all(ClientMetaMessage(Constants.CLIENT_GET, key, 0).serialize())
        data = self.client_socket.recv(ServerMetaMessage.packlength())
        meta = ServerMetaMessage.deserialize(data)
        if meta.code != Constants.SERVER_SUCCESS:
            return None
        length = meta.length
        data = self.receive_all(length)
        return data if data is None else bytes(data)

    def list(self) -> List[str]:
        self.send_all(ClientMetaMessage(Constants.CLIENT_LIST, "", 0).serialize())
        data = self.client_socket.recv(ServerMetaMessage.packlength())
        meta = ServerMetaMessage.deserialize(data)
        if meta.code != Constants.SERVER_SUCCESS:
            logger.error("LMCServerConnector: Cannot list keys from the remote server!")
            return []
        length = meta.length
        data = self.receive_all(length)
        return list(filter(lambda s: len(s) > 0, data.decode().split("\n")))

    def close(self):
        self.client_socket.close()
        logger.info("Closed the lmserver connection")
