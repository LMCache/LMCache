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
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.server.storage_backend import CreateStorageBackend
from lmcache.v1.transfer_channel.py_socket_channel import (
    PySocketChannel as TransferChannel,
)

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

    def handle_client(self, client_socket):
        # Create a channel for this client connection
        channel = TransferChannel(
            async_mode=False,
            role="receiver",
            buffer_ptr=0,
            buffer_size=0,
            align_bytes=1,
            tp_rank=0,
            peer_init_url=None,
        )
        # Initialize data socket via base-class API using an fd URL.
        fd = client_socket.detach()
        channel.lazy_init_peer_connection(
            local_id="server",
            peer_id="client",
            peer_init_url=f"fd://{fd}",
        )

        try:
            while True:
                try:
                    logger.debug("Waiting for command")
                    # Receive header using channel
                    header_buf = bytearray(ClientMetaMessage.packlength())
                    recv_count = channel.batched_recv(
                        [header_buf],
                        transfer_spec={"size": ClientMetaMessage.packlength()},
                    )
                    if recv_count == 0:
                        break
                    meta = ClientMetaMessage.deserialize(bytes(header_buf))
                    logger.debug(f"Received command: {meta.command}")
                    match meta.command:
                        case ClientCommand.PUT:
                            t0 = time.perf_counter()
                            # Receive data using channel
                            data_buf = bytearray(meta.length)
                            recv_count = channel.batched_recv(
                                [data_buf],
                                transfer_spec={"size": meta.length},
                            )
                            if recv_count == 0:
                                break
                            t1 = time.perf_counter()
                            # Avoid an extra full-payload copy; backend expects
                            # bytearray.
                            self.data_store.put(meta, data_buf)
                            t2 = time.perf_counter()
                            logger.debug(
                                f"Time to receive data: {t1 - t0}, time to store "
                                f"data: {t2 - t1}"
                            )

                        case ClientCommand.GET:
                            t0 = time.perf_counter()
                            lms_memory_obj = self.data_store.get(meta.key)
                            t1 = time.perf_counter()
                            if lms_memory_obj is not None:
                                # Send response using channel
                                response_meta = ServerMetaMessage(
                                    ServerReturnCode.SUCCESS,
                                    lms_memory_obj.length,
                                    lms_memory_obj.fmt,
                                    lms_memory_obj.dtype,
                                    lms_memory_obj.shape,
                                )
                                channel.batched_send([response_meta.serialize()])
                                t2 = time.perf_counter()
                                channel.batched_send([lms_memory_obj.data])
                                t3 = time.perf_counter()
                                logger.debug(
                                    f"Time to get data: {t1 - t0}, time to send "
                                    f"meta: {t2 - t1}, time to send data: {t3 - t2}"
                                )
                            else:
                                response_meta = ServerMetaMessage(
                                    ServerReturnCode.FAIL,
                                    0,
                                    MemoryFormat(1),
                                    torch.float16,
                                    torch.Size((0, 0, 0, 0)),
                                )
                                channel.batched_send([response_meta.serialize()])

                        case ClientCommand.EXIST:
                            code = (
                                ServerReturnCode.SUCCESS
                                if self.data_store.contains(meta.key)
                                else ServerReturnCode.FAIL
                            )
                            logger.debug(f"Key exists: {code}")
                            response_meta = ServerMetaMessage(
                                code,
                                0,
                                MemoryFormat(1),
                                torch.float16,
                                torch.Size((0, 0, 0, 0)),
                            )
                            channel.batched_send([response_meta.serialize()])
                        case ClientCommand.HEALTH:
                            response_meta = ServerMetaMessage(
                                ServerReturnCode.SUCCESS,
                                0,
                                MemoryFormat(1),
                                torch.float16,
                                torch.Size((0, 0, 0, 0)),
                            )
                            channel.batched_send([response_meta.serialize()])
                            logger.debug("Health check successful")
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    logger.info("Client socket error, closing connection: %s", e)
                    break

                    # TODO(Jiayi): Implement List
                    # case ClientCommand.LIST:
                    #     keys = list(self.data_store.list_keys())
                    #     data = "\n".join(keys).encode()
                    #     response_meta = ServerMetaMessage(
                    #         ServerReturnCode.SUCCESS,
                    #         len(data),
                    #     )
                    #     channel.batched_send([response_meta.serialize()])
                    #     channel.batched_send([data])

        finally:
            logger.info("Client disconnected")
            channel.close()

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
