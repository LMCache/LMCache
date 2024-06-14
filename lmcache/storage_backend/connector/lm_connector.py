from typing import Optional, List
import socket
from lmcache.protocol import Constants, ClientMetaMessage, ServerMetaMessage
from lmcache.storage_backend.connector.base_connector import RemoteConnector
from lmcache.logging import init_logger

logger = init_logger(__name__)

# TODO: performance optimization for this class, consider using C/C++/Rust for communication + deserialization
class LMCServerConnector(RemoteConnector):
    def __init__(self, host, port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))

    def receive_all(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def send_all(self, data):
        self.client_socket.sendall(data)

    def exists(self, key: str) -> bool:
        self.send_all(ClientMetaMessage(Constants.CLIENT_EXIST, key, 0).serialize())
        response = self.client_socket.recv(ServerMetaMessage.packlength())
        return ServerMetaMessage.deserialize(response).code == Constants.SERVER_SUCCESS

    def set(self, key: str, obj: bytes):
        self.send_all(ClientMetaMessage(Constants.CLIENT_PUT, key, len(obj)).serialize())
        self.send_all(obj)
        response = self.client_socket.recv(ServerMetaMessage.packlength())
        if ServerMetaMessage.deserialize(response).code != Constants.SERVER_SUCCESS:
            raise RuntimeError(f"Failed to set key: {ServerMetaMessage.deserialize(response).code}")

    def get(self, key: str) -> Optional[bytes]:
        self.send_all(ClientMetaMessage(Constants.CLIENT_GET, key, 0).serialize())
        data = self.client_socket.recv(ServerMetaMessage.packlength())
        meta = ServerMetaMessage.deserialize(data)
        if meta.code != Constants.SERVER_SUCCESS:
            return None
        length = meta.length
        data = self.receive_all(length)
        return data

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
