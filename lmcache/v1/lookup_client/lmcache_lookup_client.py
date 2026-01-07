# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import namedtuple
from typing import TYPE_CHECKING, Optional, Union
import json
import threading

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_rpc_path_lmcache,
    get_zmq_socket,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class LMCacheLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - lookup_server_worker_ids:
        is a config to control create lookup server on some workers.
        if mla is not enabled, default is [];
        if mla is enabled, default is [0];
        - if lookup_server_worker_ids is [], start lookup server on all workers
        - if lookup_server_worker_ids is [0], start lookup server on worker0
        - if lookup_server_worker_ids is [0, 3, 6], start lookup server on
          worker0, worker3 and worker6
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ):
        self.encoder = msgspec.msgpack.Encoder()
        self.ctx = get_zmq_context(use_asyncio=False)
        self.config = config
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        self.pipeline_parallel_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_ranks = self.tensor_parallel_size * self.pipeline_parallel_size
        self.lookup_server_worker_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )

        self.sockets = []
        if len(self.lookup_server_worker_ids) > 0:
            ranks = self.lookup_server_worker_ids
            self.num_ranks = len(self.lookup_server_worker_ids)
        else:
            ranks = [i for i in range(self.num_ranks)]

        # Store socket creation parameters for recreation
        SocketParams = namedtuple("SocketParams", ["socket_path", "rank"])
        self.socket_params = [
            SocketParams(
                socket_path=get_zmq_rpc_path_lmcache(
                    vllm_config, "lookup", rpc_port, rank
                ),
                rank=rank,
            )
            for rank in ranks
        ]
        self.timeout_ms = config.lookup_timeout_ms

        # NOTE: map from lookup_id (i.e., req_id) to req's status.
        # int indicates number of hit tokens.
        # The assumption here is that once a request is looked up,
        # the following lookups of the same request must have the
        # same result.
        self.reqs_status: dict[str, int] = {}

        for params in self.socket_params:
            logger.info(
                "lmcache lookup client connect to rank %s with socket path %s",
                params.rank,
                params.socket_path,
            )
            socket = get_zmq_socket(
                self.ctx,
                params.socket_path,
                "ipc",
                zmq.REQ,
                "connect",
            )

            # Set socket timeout during initialization
            socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

            self.sockets.append(socket)

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.enable_blending = config.enable_blending
        self.token_database: TokenDatabase
        if self.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

    def _recreate_socket(self) -> None:
        """Recreate all sockets."""
        for rank_idx in range(self.num_ranks):
            # Close old socket
            old_socket = self.sockets[rank_idx]
            if old_socket is not None:
                try:
                    old_socket.close(linger=0)
                except zmq.ZMQError as e:
                    logger.warning(
                        "ZMQ error closing old socket for rank %s: %s",
                        rank_idx,
                        e,
                    )
                except AttributeError:
                    # Socket already closed or invalid
                    pass

            # Create new socket using stored parameters
            params = self.socket_params[rank_idx]
            logger.info(
                "Recreating socket for rank %s with path %s",
                params.rank,
                params.socket_path,
            )

            new_socket = get_zmq_socket(
                self.ctx,
                params.socket_path,
                "ipc",
                zmq.REQ,
                "connect",
            )
            new_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            new_socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

            self.sockets[rank_idx] = new_socket

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        "-1 means not found;
        None means ongoing; (this semantic is not supported in sync lookup client)
        int >= 0 means number of hit tokens
        """
        return self.reqs_status.get(lookup_id, -1)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        lookup_id_buf = lookup_id.encode("utf-8")
        request_configs_str = ""
        if request_configs is not None and len(request_configs) != 0:
            request_configs_str = json.dumps(request_configs)
        request_configs_buf = request_configs_str.encode("utf-8")

        # NOTE(Jiayi): We cannot only send hashes when blending enabled
        # because the blender need the input embedding.
        if not self.enable_blending:
            hashes = []
            offsets = []

            # We already have hashes here so we can skip the chunks that are already
            # in GPU cache. Don't pass num_computed_tokens to lookup server.

            for start, end, key in self.token_database.process_tokens(
                token_ids, make_key=False
            ):
                hashes.append(key)
                offsets.append(end - start)

            # if the token database returns no hashes, return 0
            if not hashes:
                return 0

            hash_buf = self.encoder.encode(hashes)
            offset_buf = self.encoder.encode(offsets)
            msg_buf = [
                hash_buf,
                offset_buf,
                lookup_id_buf,
                request_configs_buf,
            ]
        else:
            # print(len(token_ids))
            tokens_buf = self.encoder.encode(token_ids)
            msg_buf = [
                tokens_buf,
                lookup_id_buf,
                request_configs_buf,
            ]

        results = []
        failed_rank = -1
        try:
            for i in range(self.num_ranks):
                failed_rank = i
                self.sockets[i].send_multipart(msg_buf, copy=False)

            # TODO(Jiayi): we can use zmq poll to optimize a bit
            for i in range(self.num_ranks):
                failed_rank = i
                resp = self.sockets[i].recv()
                result = int.from_bytes(resp, "big")
                results.append(result)
        except zmq.Again as e:
            logger.error(
                "Timeout occurred for rank %s, recreating all sockets. Error: %s",
                failed_rank,
                e,
            )
            self._recreate_socket()
            return 0
        except zmq.ZMQError as e:
            logger.error(
                "ZMQ error for rank %s: %s, recreating all sockets",
                failed_rank,
                e,
            )
            self._recreate_socket()
            return 0

        assert len(results) == self.num_ranks
        if len(set(results)) > 1:
            logger.warning(
                "Lookup results (number of hit tokens) differ "
                "across (TP and PP) ranks: %s.",
                results,
            )
        # NOTE: it is possible that the number of hit tokens is different
        # across (TP and PP) ranks, so we can use the minimum value as the
        # number of hit tokens.
        num_hit_toks = min(results)
        self.reqs_status[lookup_id] = num_hit_toks

        return num_hit_toks

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.reqs_status.pop(lookup_id, None)

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        for socket in self.sockets:
            try:
                socket.close(linger=0)
            except Exception as e:
                logger.warning("Error closing socket: %s", e)

        try:
            if self.ctx:
                self.ctx.term()
        except Exception as e:
            logger.warning("Error terminating ZMQ context: %s", e)


class LMCacheLookupServer:
    """ZMQ-based lookup server that handles lookup requests using LMCacheEngine."""

    def __init__(self, lmcache_engine: LMCacheEngine, vllm_config: "VllmConfig"):
        self.decoder = msgspec.msgpack.Decoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup", rpc_port, vllm_config.parallel_config.rank
        )
        self.socket = get_zmq_socket(
            self.ctx,
            socket_path,
            "ipc",
            zmq.REP,  # type: ignore[attr-defined]
            "bind",
        )
        # Set socket timeout to allow periodic check of running flag
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

        self.lmcache_engine = lmcache_engine
        self.running = True

        self.enable_blending = lmcache_engine.config.enable_blending

        def process_request():
            while self.running:
                try:
                    frames = self.socket.recv_multipart(copy=False)
                except zmq.Again:
                    # Timeout occurred, check running flag and continue
                    continue
                lookup_id = frames[-2].bytes.decode("utf-8")
                request_configs_str = frames[-1].bytes.decode("utf-8")
                request_configs = None
                if request_configs_str != "":
                    request_configs = json.loads(request_configs_str)
                if not self.enable_blending:
                    hash_frames = frames[0]
                    offset_frames = frames[1]
                    hashes = self.decoder.decode(hash_frames)
                    offsets = self.decoder.decode(offset_frames)
                    result = self.lmcache_engine.lookup(
                        hashes=hashes,
                        offsets=offsets,
                        lookup_id=lookup_id,
                        pin=True,
                        request_configs=request_configs,
                    )
                else:
                    token_frames = frames[0]
                    tokens = self.decoder.decode(token_frames)
                    result = self.lmcache_engine.lookup(
                        tokens=tokens,
                        lookup_id=lookup_id,
                        pin=True,
                        request_configs=request_configs,
                    )
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        logger.info("lmcache lookup server start on %s", socket_path)
        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        # Stop the processing thread first
        self.running = False

        # Wait for thread to finish with timeout
        # Thread will exit within 1 second due to socket RCVTIMEO
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Lookup server thread did not terminate gracefully")

        # Close the socket after thread is stopped
        self.socket.close(linger=0)
