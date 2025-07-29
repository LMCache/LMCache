# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List
import os
import threading

# Third Party
from vllm.utils import make_zmq_socket
import msgspec
import zmq

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.offload_server.abstract_server import OffloadServerInterface
from lmcache.v1.offload_server.message import OffloadMsg, OffloadRetMsg
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig


class ZMQOffloadServer(OffloadServerInterface):
    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        vllm_config: "VllmConfig",
        tp_rank: int,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        offload_rpc_port = int(os.environ.get("LMCACHE_OFFLOAD_RPC_PORT", 100))
        socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "offload", offload_rpc_port, tp_rank
        )
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            while self.running:
                frame = self.socket.recv(copy=False)
                offload_msg = msgspec.msgpack.decode(frame, type=OffloadMsg)
                result = self.offload(
                    offload_msg.hashes,
                    offload_msg.slot_mapping,
                    offload_msg.offsets,
                )
                response = OffloadRetMsg(success=result)
                response = msgspec.msgpack.encode(response)
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def offload(
        self,
        hashes: List[int],
        slot_mapping: List[int],
        offsets: List[int],
    ) -> bool:
        self.lmcache_engine.store(
            hashes=hashes, slot_mapping=slot_mapping, offsets=offsets
        )
        return True

    def close(self) -> None:
        self.socket.close(linger=0)
        self.running = False
        self.thread.join()
