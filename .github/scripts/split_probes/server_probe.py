# SPDX-License-Identifier: Apache-2.0
"""Real transport servers with small typed handlers; no storage-manager claim."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))

# Third Party
import zmq

# First Party
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.grpc_impl.server import GrpcMultiprocessServer
from lmcache.v1.multiprocess.transport.zmq_impl.server_mq import MessageQueueServer


class Handlers:
    """Only management calls; absent lookup intentionally tests error behavior."""

    @request_handler()
    def get_chunk_size(self) -> int:
        return 4

    @request_handler()
    def negotiate_chunk_size(self, required_chunk_alignment: int) -> int:
        if 4 % required_chunk_alignment:
            raise ValueError("alignment does not divide the test chunk size")
        return 4

    @request_handler()
    def ping(self, instance_id: int | None) -> bool:
        return True

    @request_handler()
    def noop(self) -> str:
        return "split-probe"


def main() -> None:
    context = zmq.Context()
    zserver = (
        MessageQueueServer(sys.argv[2], context) if sys.argv[2] != "none" else None
    )
    handlers = Handlers()
    for name in ("get_chunk_size", "negotiate_chunk_size", "ping", "noop"):
        if zserver is not None:
            zserver.add_handler(name, HandlerType.SERIALIZED, getattr(handlers, name))
    gserver = GrpcMultiprocessServer(
        "grpc://127.0.0.1:0",
        max_cpu_workers=1,
        max_gpu_workers=1,
        grpc_server_workers=2,
    )
    gserver.add_modules([handlers])
    if zserver is not None:
        zserver.start()
    gserver.start()
    zmq_endpoint = (
        zserver.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        if zserver is not None
        else None
    )
    print(
        json.dumps(
            {"zmq": zmq_endpoint, "grpc": f"grpc://127.0.0.1:{gserver.bound_port}"}
        ),
        flush=True,
    )
    try:
        sys.stdin.readline()
    finally:
        gserver.close()
        shutdown = {}
        if zserver is not None:
            zserver.close()
            shutdown = {
                "zmq_socket_closed": zserver.socket.closed,
                "zmq_worker_alive": zserver.worker_thread.is_alive(),
            }
        # term() blocks until every socket of the context is closed.
        context.term()
        shutdown["context_terminated"] = context.closed
        print(json.dumps({"shutdown": shutdown}), flush=True)


if __name__ == "__main__":
    main()
