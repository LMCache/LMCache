import socket
import time
import threading
import torch
from io import BytesIO
from lmcache.protocol import ClientMetaMessage, ServerMetaMessage, Constants
from lmcache.server.server_storage_backend import CreateStorageBackend

class LMCacheServer:
    def __init__(self, host, port, device):
        self.host = host
        self.port = port
        #self.data_store = {}
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
                header = self.receive_all(client_socket, ClientMetaMessage.packlength())
                if not header:
                    break
                meta = ClientMetaMessage.deserialize(header)

                match meta.command:
                    case Constants.CLIENT_PUT:
                        t0 = time.perf_counter()
                        s = self.receive_all(client_socket, meta.length)
                        t1 = time.perf_counter()
                        #self.data_store[meta.key] = s
                        self.data_store.put(meta.key, s)
                        t2 = time.perf_counter()
                        #client_socket.sendall(ServerMetaMessage(Constants.SERVER_SUCCESS, 0).serialize())
                        #t3 = time.perf_counter()
                        print(f"Time to receive data: {t1 - t0}, time to store data: {t2 - t1}")

                    case Constants.CLIENT_GET:
                        t0 = time.perf_counter()
                        #data_string = self.data_store.get(meta.key, None)
                        data_string = self.data_store.get(meta.key)
                        t1 = time.perf_counter()
                        if data_string is not None:
                            client_socket.sendall(ServerMetaMessage(Constants.SERVER_SUCCESS, len(data_string)).serialize())
                            t2 = time.perf_counter()
                            client_socket.sendall(data_string)
                            t3 = time.perf_counter()
                            print(f"Time to get data: {t1 - t0}, time to send meta: {t2 - t1}, time to send data: {t3 - t2}")
                        else:
                            client_socket.sendall(ServerMetaMessage(Constants.SERVER_FAIL, 0).serialize())

                    case Constants.CLIENT_EXIST:
                        #code = Constants.SERVER_SUCCESS if meta.key in self.data_store else Constants.SERVER_FAIL
                        code = Constants.SERVER_SUCCESS if meta.key in self.data_store.list_keys() else Constants.SERVER_FAIL
                        client_socket.sendall(ServerMetaMessage(code, 0).serialize())

                    case Constants.CLIENT_LIST:
                        keys = list(self.data_store.list_keys())
                        data = "\n".join(keys).encode()
                        client_socket.sendall(ServerMetaMessage(Constants.SERVER_SUCCESS, len(data)).serialize())
                        client_socket.sendall(data)

        finally:
            client_socket.close()

    def run(self):
        print(f"Server started at {self.host}:{self.port}")
        try:
            while True:
                client_socket, addr = self.server_socket.accept()
                print(f"Connected by {addr}")
                threading.Thread(target=self.handle_client, args=(client_socket,)).start()
        finally:
            self.server_socket.close()

def main():
    import os, sys
    if len(sys.argv) not in [3,4]:
        print(f"Usage: {sys.argv[0]} <host> <port> <storage>(default:cpu)")
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
