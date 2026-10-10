# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List
import os
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.offload_server.abstract_server import OffloadServerInterface
from lmcache.v1.offload_server.message import OffloadMsg, OffloadRetMsg
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_rpc_path_lmcache,
    get_zmq_socket,
)

logger = init_logger(__name__)


class ZMQOffloadServer(OffloadServerInterface):
    """Serve offload requests over a ZMQ REP socket on a background thread.

    Attempt one reply per request: ``success`` is true when the engine call
    returns normally and false when the request is invalid or the engine
    raises. A REP socket must reply before receiving another request.
    Transport errors stop the thread, and ``running`` is cleared on exit.
    """

    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        tp_rank: int,
    ) -> None:
        """Bind the offload endpoint and start its request thread.

        Args:
            lmcache_engine: LMCacheEngine handling offload operations.
            tp_rank: Integer rank suffix used to construct the IPC endpoint.

        Raises:
            ValueError: If ``LMCACHE_OFFLOAD_RPC_PORT`` is not an integer, or
                the base RPC directory cannot fit even a shortened socket name
                within the IPC path limit.
            zmq.ZMQError: If the socket cannot be created or bound.
        """
        metadata = lmcache_engine.metadata
        self.ctx = get_zmq_context(use_asyncio=False)
        offload_rpc_port = int(os.environ.get("LMCACHE_OFFLOAD_RPC_PORT", 100))
        engine_id = metadata.engine_id or "default"
        socket_path = get_zmq_rpc_path_lmcache(
            engine_id, "offload", offload_rpc_port, tp_rank
        )
        self.socket = get_zmq_socket(
            self.ctx,
            socket_path,
            "ipc",
            zmq.REP,  # type: ignore[attr-defined]
            "bind",
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request() -> None:
            """Complete each REP receive/send cycle, including failed requests."""
            try:
                while self.running:
                    try:
                        frames = self.socket.recv_multipart(copy=False)
                        try:
                            if len(frames) != 1:
                                raise ValueError(
                                    "Offload requests must use one frame, "
                                    f"got {len(frames)}"
                                )
                            offload_msg = msgspec.msgpack.decode(
                                frames[0], type=OffloadMsg
                            )
                            result = self.offload(
                                offload_msg.hashes,
                                offload_msg.slot_mapping,
                                offload_msg.offsets,
                            )
                        except Exception:
                            logger.exception("Failed to process offload request")
                            result = False

                        # REP must reply even when decoding or storing failed.
                        response = msgspec.msgpack.encode(OffloadRetMsg(success=result))
                        self.socket.send(response)
                    except zmq.ZMQError as e:
                        if not self.running:
                            logger.info(
                                "ZMQ socket closed, exiting offload server thread"
                            )
                        else:
                            logger.error("ZMQ error in offload server: %s", e)
                        break
            finally:
                self.running = False

        self.thread = threading.Thread(
            target=process_request, daemon=True, name="offload-server-thread"
        )
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
        logger.info("Closing ZMQOffloadServer...")
        self.running = False

        # Close socket to interrupt blocking recv()
        try:
            self.socket.close(linger=0)
            logger.info("ZMQ socket closed")
        except Exception as e:
            logger.warning("Error closing ZMQ socket: %s", e)

        # Wait for thread with timeout to prevent deadlock
        if self.thread.is_alive():
            logger.info("Waiting for offload server thread to finish...")
            self.thread.join(timeout=5.0)

            if self.thread.is_alive():
                logger.warning(
                    "Offload server thread did not terminate within timeout. "
                    "Thread may be stuck in blocking recv(). "
                    "Proceeding with shutdown anyway."
                )
            else:
                logger.info("Offload server thread terminated successfully")
        else:
            logger.info("Offload server thread already stopped")
