# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional
import threading

# Third Party
from vllm.utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
import torch
import zmq

# First Party
from lmcache.integration.vllm.utils import mla_enabled
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class LMCacheLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - create_lookup_server_only_on_worker_0_for_mla:
        is a flag to control whether to create lookup server only on worker 0.
    """

    def __init__(self, vllm_config: "VllmConfig", config: LMCacheEngineConfig):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        use_mla = mla_enabled(vllm_config.model_config)
        self.create_lookup_server_only_on_worker_0_for_mla = (
            config.get_extra_config_value(
                "create_lookup_server_only_on_worker_0_for_mla", use_mla
            )
        )
        ranks = self.tensor_parallel_size
        self.sockets = []
        if self.create_lookup_server_only_on_worker_0_for_mla:
            ranks = 1
        for tp_rank in range(ranks):
            socket_path = get_zmq_rpc_path_lmcache(
                vllm_config, "lookup", rpc_port, tp_rank
            )
            logger.info(
                f"lmcache lookup client connect to tp_rank {tp_rank} "
                f"with socket path {socket_path}"
            )
            socket = self.socket = make_zmq_socket(
                self.ctx,
                socket_path,
                zmq.REQ,  # type: ignore[attr-defined]
                bind=False,
            )

            self.sockets.append(socket)

    def lookup(
        self,
        token_ids: torch.Tensor,
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> int:
        token_bufs = self.encoder.encode(token_ids)
        lookup_id_buf = lookup_id.encode("utf-8")
        request_configs_str = ""
        if request_configs is not None and len(request_configs) != 0:
            request_configs_str = "@".join(
                [f"{k}%{v}" for k, v in request_configs.items()]
            )
        request_configs_buf = request_configs_str.encode("utf-8")
        ranks = self.tensor_parallel_size
        if self.create_lookup_server_only_on_worker_0_for_mla:
            ranks = 1
        results = []
        msg_buf = token_bufs + [lookup_id_buf, request_configs_buf]
        for i in range(ranks):
            self.sockets[i].send_multipart(msg_buf, copy=False)

        # TODO(Jiayi): we can use zmq poll to optimize a bit
        for i in range(ranks):
            resp = self.sockets[i].recv()
            result = int.from_bytes(resp, "big")
            results.append(result)

        if not all(x == results[0] for x in results):
            raise RuntimeError(
                f"Lookup results (number of hit tokens) differ "
                f"across tensor parallel ranks: {results}."
            )
        return results[0]

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:
    """ZMQ-based lookup server that handles lookup requests using LMCacheEngine."""

    def __init__(self, lmcache_engine: LMCacheEngine, vllm_config: "VllmConfig"):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup", rpc_port, vllm_config.parallel_config.rank
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
                # try:
                # request = self.socket.recv()
                frames = self.socket.recv_multipart(copy=False)
                token_frames = frames[:-2]
                lookup_id = frames[-2].bytes.decode("utf-8")
                request_configs_str = frames[-1].bytes.decode("utf-8")
                request_configs = None
                if request_configs_str != "":
                    request_configs = {}
                    request_configs_list = request_configs_str.split("@")
                    for kv in request_configs_list:
                        kvs = kv.split("%", 1)
                        if len(kvs) != 2:
                            raise ValueError("Unexpected tags_str: {tags_str}")
                        request_configs[kvs[0]] = kvs[1]

                token_ids = self.decoder.decode(token_frames)
                result = self.lmcache_engine.lookup(
                    token_ids,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )
                response = result.to_bytes(4, "big")
                self.socket.send(response)
                # except Exception as e:
                #    logger.error("Error in LMCache lookup server: %s", e)
                #    break
                # continue

        logger.info(f"lmcache lookup server start on {socket_path}")
        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
