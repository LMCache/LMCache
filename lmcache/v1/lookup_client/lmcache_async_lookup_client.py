# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union
import threading

# Third Party
from vllm.utils import make_zmq_socket
import msgspec
import torch
import zmq

# First Party
from lmcache.integration.vllm.utils import create_lmcache_metadata, mla_enabled
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# NOTE(Jiayi): Prefetch could load extra redundant cache if multiple
# workers has different hit tokens.
class LMCacheLookupAndPrefetchClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - create_lookup_server_only_on_worker_0_for_mla:
        is a flag to control whether to create lookup server only on worker 0.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
    ):
        metadata, config = create_lmcache_metadata(vllm_config)

        self.encoder = msgspec.msgpack.Encoder()
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
        self.push_sockets = []
        if self.create_lookup_server_only_on_worker_0_for_mla:
            ranks = 1
        for tp_rank in range(ranks):
            worker_socket_path = get_zmq_rpc_path_lmcache(
                vllm_config, "lookup_worker", rpc_port, tp_rank
            )
            logger.info(
                f"lmcache lookup client connect to tp_rank {tp_rank} "
                f"with worker socket path {worker_socket_path}"
            )

            push_socket = make_zmq_socket(
                self.ctx,
                socket_path,
                zmq.REQ,  # type: ignore[attr-defined]
                bind=False,
            )

            self.push_sockets.append(push_socket)
        
        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup_scheduler", rpc_port, 0
        )
        self.pull_socket = make_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            zmq.PULL,  # type: ignore[attr-defined]
            bind=True,
        )
        logger.info(
            f"lmcache lookup client connect to scheduler "
            f"with socket path {scheduler_socket_path}"
        )


        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.token_database: TokenDatabase
        if config.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)
        
        # A lock is needed since we need another thread to pull
        # responses from the lookup_and_prefetch server 
        # (e.g., worker process).
        self.lock = threading.Lock()

        # map from req_id to req's status.
        # None indicates ongoing.
        # int indicates number of hit tokens.
        self.reqs_status: dict[str, Optional[int]] = []

        # map from req_id to number of hit tokens for each worker
        self.res_for_each_worker: dict[str, list[int]] = {}

        # The two parts are [req_id, num_hit_tokens]
        self.num_parts = 2

        self.running = True

        self.thread = threading.Thread(
            target=process_responses_from_workers, daemon=True)
        self.thread.start()
        

    # TODO(Jiayi): We might want to differentiate sync and async lookup.
    # For example, we might want sync lookup if for local lookup.
    def batched_lookup_and_prefetch(
        self,
        batched_req_ids: list[str],
        batched_token_ids: Union[list[torch.Tensor], list[list[int]]],
        batched_request_configs: list[Optional[dict]],
    ) -> list[Optional[int]]:

        batched_res = []

        batched_msg_buf = []
        for req_id, token_ids, request_configs in zip(
            batched_req_ids, batched_token_ids, batched_request_configs
        ):
            with self.lock:
                req_status = self.reqs_status.get(req_id, -1)
                if req_id != 1:
                    batched_res.append(req_status)
                    if req_status is not None:
                        self.reqs_status.pop(req_id)
                    continue
                requests_status[req_id] = None

            for start, end, key in self.token_database.process_tokens(
                token_ids, make_key=False
            ):
                hashes.append(key)
                offsets.append(end - start)
            hash_buf = self.encoder.encode(hashes)
            offset_buf = self.encoder.encode(offsets)

            req_id_buf = req_id.encode("utf-8")
            request_configs_str = ""
            if request_configs is not None and len(request_configs) != 0:
                request_configs_str = "@".join(
                    [f"{k}%{v}" for k, v in request_configs.items()]
                )
            request_configs_buf = request_configs_str.encode("utf-8")

            batched_msg_buf.extend([
                hash_buf,
                offset_buf,
                req_id_buf,
                request_configs_buf,
            ])

        ranks = self.tensor_parallel_size
        if self.create_lookup_server_only_on_worker_0_for_mla:
            ranks = 1
        for i in range(ranks):
            self.push_sockets[i].send_multipart(
                batched_msg_buf, copy=False)
            

    
    def process_responses_from_workers(self):
        while self.running:
            frames = self.pull_socket.recv_multipart(copy=False)
            assert len(frames) == self.num_parts
            req_id = frames[0].bytes.decode("utf-8")
            res = int.from_bytes(frames[1], "big")

            with self.lock:
                if req_id not in self.ress_for_each_worker:
                    self.res_for_each_worker[req_id] = [res]
                else:
                    self.res_for_each_worker[req_id].append(res)
                all_res = self.res_for_each_worker[req_id]

                if len(all_res) == self.tensor_parallel_size:
                    self.res_for_each_worker.pop(req_id)

                    # NOTE: it is possible that the number of hit 
                    # tokens is different across TP ranks, so we 
                    # can use the minimum value as the number of 
                    # hit tokens.
                    self.reqs_status[req_id] = min(all_res)


    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:
    """ZMQ-based lookup server that handles lookup requests using LMCacheEngine."""

    def __init__(self, lmcache_engine: LMCacheEngine, vllm_config: "VllmConfig"):
        self.decoder = msgspec.msgpack.Decoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        worker_socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup_worker", rpc_port, vllm_config.parallel_config.rank
        )
        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup_scheduler", rpc_port, 0
        )
        self.push_socket = make_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            zmq.PUSH,  # type: ignore[attr-defined]
            bind=False,
        )
        self.pull_socket = = make_zmq_socket(
            self.ctx,
            worker_socket_path,
            zmq.PULL,  # type: ignore[attr-defined]
            bind=True,
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        logger.info("lmcache lookup server start with"
                    f" scheduler socket path {scheduler_socket_path}, "
                    f"worker socket path {worker_socket_path}")
        self.thread = threading.Thread(
            target=process_requests_from_scheduler, daemon=True)
        self.thread.start()

        # The four parts are [hash, offset, req_id, request_configs]
        self.num_parts = 4

    
    def process_requests_from_scheduler(self):
        while self.running:
            frames = self.pull_socket.recv_multipart(copy=False)
            num_framses = len(frames)
            assert num_frames % self.num_parts == 0
            for i in range(0, num_frames, self.num_parts):
                
                req_id = frames[i].bytes.decode("utf-8")

                hash_frame = frames[i+1]
                hashes = self.decoder.decode(hash_frames)

                offset_frame = frames[i+2]
                offsets = self.decoder.decode(offset_frames)

                request_configs_str = frames[i+3].bytes.decode("utf-8")
                request_configs = None
                if request_configs_str != "":
                    request_configs = {}
                    request_configs_list = request_configs_str.split("@")
                    for kv in request_configs_list:
                        kvs = kv.split("%", 1)
                        if len(kvs) != 2:
                            raise ValueError(f"Unexpected tags_str: {kvs}")
                        request_configs[kvs[0]] = kvs[1]
            
                self.lmcache_engine.lookup_and_prefetch(
                    lookup_id=req_id,
                    hashes=hashes, 
                    offsets=offsets,
                    pin=True,
                    request_configs=request_configs,
                )
    
    def send_response_to_scheduler(
        self, 
        req_id: str, 
        num_hit_tokens: int
    ):
        req_id_buf = req_id.encode("utf-8")
        num_hit_tokens_buf = num_hit_tokens.to_bytes(4, "big")
        self.push_socket.send_multipart(
            [req_id_buf, num_hit_tokens_buf], copy=False)

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
