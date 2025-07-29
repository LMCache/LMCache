# SPDX-License-Identifier: Apache-2.0
# Standard
import socket
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import ClientMetaMessage, Constants, ServerMetaMessage
from lmcache.v1.server.storage_backend import CreateStorageBackend

logger = init_logger(__name__)


class LMCacheServer:
    def __init__(self, host, port, device):
        self.host = host
        self.port = port
        # self.data_store = {}
        self.data_store = CreateStorageBackend(device)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen()

    def receive_all(self, client_socket, n):
        data = bytearray()
        while len(data) < n:
            packet = client_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def handle_client(self, client_socket):
        try:
            while True:
                logger.debug("Waiting for command")
                header = self.receive_all(client_socket, ClientMetaMessage.packlength())
                if not header:
                    break
                meta = ClientMetaMessage.deserialize(header)
                logger.debug(f"Received command: {meta.command}")
                match meta.command:
                    case Constants.CLIENT_PUT:
                        t0 = time.perf_counter()
                        s = self.receive_all(client_socket, meta.length)
                        t1 = time.perf_counter()
                        self.data_store.put(meta, s)
                        t2 = time.perf_counter()
                        logger.debug(
                            f"Time to receive data: {t1 - t0}, time to store "
                            f"data: {t2 - t1}"
                        )

                    case Constants.CLIENT_GET:
                        t0 = time.perf_counter()
                        lms_memory_obj = self.data_store.get(meta.key)
                        t1 = time.perf_counter()
                        if lms_memory_obj is not None:
                            client_socket.sendall(
                                ServerMetaMessage(
                                    Constants.SERVER_SUCCESS,
                                    lms_memory_obj.length,
                                    lms_memory_obj.fmt,
                                    lms_memory_obj.dtype,
                                    lms_memory_obj.shape,
                                ).serialize()
                            )
                            t2 = time.perf_counter()
                            client_socket.sendall(lms_memory_obj.data)
                            t3 = time.perf_counter()
                            logger.debug(
                                f"Time to get data: {t1 - t0}, time to send "
                                f"meta: {t2 - t1}, time to send data: {t3 - t2}"
                            )
                        else:
                            client_socket.sendall(
                                ServerMetaMessage(
                                    Constants.SERVER_FAIL,
                                    0,
                                    MemoryFormat(1),
                                    torch.float16,
                                    torch.Size((0, 0, 0, 0)),
                                ).serialize()
                            )

                    case Constants.CLIENT_EXIST:
                        code = (
                            Constants.SERVER_SUCCESS
                            if self.data_store.contains(meta.key)
                            else Constants.SERVER_FAIL
                        )
                        logger.debug(f"Key exists: {code}")
                        client_socket.sendall(
                            ServerMetaMessage(
                                code,
                                0,
                                MemoryFormat(1),
                                torch.float16,
                                torch.Size((0, 0, 0, 0)),
                            ).serialize()
                        )
                    case Constants.CLIENT_HEALTH:
                        client_socket.sendall(
                            ServerMetaMessage(
                                Constants.SERVER_SUCCESS,
                                0,
                                MemoryFormat(1),
                                torch.float16,
                                torch.Size((0, 0, 0, 0)),
                            ).serialize()
                        )
                        logger.debug("Health check successful")

                    # TODO(Jiayi): Implement List
                    # case Constants.CLIENT_LIST:
                    #     keys = list(self.data_store.list_keys())
                    #     data = "\n".join(keys).encode()
                    #     client_socket.sendall(
                    #         ServerMetaMessage(Constants.SERVER_SUCCESS,
                    #                           len(data)).serialize())
                    #     client_socket.sendall(data)

        finally:
            logger.info("Client disconnected")
            client_socket.close()

    def run(self):
        logger.info(f"Server started at {self.host}:{self.port}")
        try:
            while True:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Connected by {addr}")
                threading.Thread(
                    target=self.handle_client, args=(client_socket,)
                ).start()
        finally:
            self.server_socket.close()


def main():
    # Standard
    import sys

    if len(sys.argv) not in [3, 4]:
        logger.error(f"Usage: {sys.argv[0]} <host> <port> <storage>(default:cpu)")
        exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    if len(sys.argv) == 4:
        device = sys.argv[3]
    else:
        device = "cpu"

    server = LMCacheServer(host, port, device)
    server.run()


if __name__ == "__main__":
    main()
