import os
from typing import List

import zmq.asyncio


def get_zmq_context():
    return zmq.asyncio.Context.instance()


def get_unix_socket_path(instance_id: str, worker_id: int):
    return f"/tmp/{instance_id}_{worker_id}.sock"


def clean_old_sockets(instance_id: str, workers: List[int]):
    for i in workers:
        path = get_unix_socket_path(instance_id, i)
        if os.path.exists(path):
            os.remove(path)


def get_server_socket(context, socket_path: str):
    socket = context.socket(zmq.REP)  # type: ignore[attr-defined]
    socket.bind(f"ipc://{socket_path}")
    return socket


def get_client_socket(context, socket_path: str):
    socket = context.socket(zmq.REQ)  # type: ignore[attr-defined]
    socket.connect(f"ipc://{socket_path}")
    return socket
