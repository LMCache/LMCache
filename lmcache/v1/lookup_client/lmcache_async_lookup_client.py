# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union
import threading
import time

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.integration.vllm.utils import create_lmcache_metadata, mla_enabled
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
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


# NOTE(Jiayi): Prefetch could load extra redundant cache if multiple
# workers has different hit tokens.
class LMCacheAsyncLookupClient(LookupClientInterface):
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
        self.ctx = get_zmq_context(use_asyncio=False)
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

            push_socket = get_zmq_socket(
                self.ctx,
                worker_socket_path,
                "ipc",
                zmq.PUSH,  # type: ignore[attr-defined]
                "connect",
            )

            self.push_sockets.append(push_socket)

        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup_scheduler", rpc_port, 0
        )
        self.pull_socket = get_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            "ipc",
            zmq.PULL,  # type: ignore[attr-defined]
            "bind",
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

        # map from lookup_id (i.e., req_id) to req's status.
        # None indicates ongoing.
        # int indicates number of hit tokens.
        self.reqs_status: dict[str, Optional[int]] = {}

        # map from lookup_id (i.e., req_id) to number of hit tokens for each worker
        self.res_for_each_worker: dict[str, list[int]] = {}

        # The two parts are [lookup_id (i.e., req_id), num_hit_tokens]
        self.num_parts = 2

        self.running = True

        self.thread = threading.Thread(
            target=self.process_responses_from_workers, daemon=True
        )
        self.thread.start()

        # default backoff time
        self.lookup_backoff_time = 0.01
        if config.extra_config is not None:
            self.lookup_backoff_time = float(
                config.extra_config.get("lookup_backoff_time", self.lookup_backoff_time)
            )

    # TODO(Jiayi): Consider batching here
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        with self.lock:
            # -1 indicates not found; None indicates ongoing.
            req_status = self.reqs_status.get(lookup_id, -1)
            if req_status is None:
                time.sleep(self.lookup_backoff_time)
                return None
            elif req_status != -1:
                return req_status
            self.reqs_status[lookup_id] = None
        hashes = []
        offsets = []
        for start, end, hash_val in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(hash_val)
            offsets.append(end - start)
        hash_buf = self.encoder.encode(hashes)
        offset_buf = self.encoder.encode(offsets)

        lookup_id_buf = lookup_id.encode("utf-8")
        request_configs_str = ""
        if request_configs is not None and len(request_configs) != 0:
            request_configs_str = "@".join(
                [f"{k}%{v}" for k, v in request_configs.items()]
            )
        request_configs_buf = request_configs_str.encode("utf-8")

        msg_buf = [
            lookup_id_buf,
            hash_buf,
            offset_buf,
            request_configs_buf,
        ]

        ranks = self.tensor_parallel_size
        if self.create_lookup_server_only_on_worker_0_for_mla:
            ranks = 1
        for i in range(ranks):
            self.push_sockets[i].send_multipart(msg_buf, copy=False)
        time.sleep(self.lookup_backoff_time)
        return None

    def process_responses_from_workers(self):
        while self.running:
            frames = self.pull_socket.recv_multipart(copy=False)
            assert len(frames) == self.num_parts
            lookup_id = frames[0].bytes.decode("utf-8")
            res = int.from_bytes(frames[1], "big")

            with self.lock:
                if lookup_id not in self.res_for_each_worker:
                    self.res_for_each_worker[lookup_id] = [res]
                else:
                    self.res_for_each_worker[lookup_id].append(res)
                all_res = self.res_for_each_worker[lookup_id]

                if len(all_res) == self.tensor_parallel_size or (
                    self.create_lookup_server_only_on_worker_0_for_mla
                    and len(all_res) == 1
                ):
                    self.res_for_each_worker.pop(lookup_id)

                    # NOTE: it is possible that the number of hit
                    # tokens is different across TP ranks, so we
                    # can use the minimum value as the number of
                    # hit tokens.
                    self.reqs_status[lookup_id] = min(all_res)

    def clear_lookup_status(self, lookup_id: str) -> None:
        with self.lock:
            self.reqs_status.pop(lookup_id, None)

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            for s in self.push_sockets:
                s.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning(f"Failed to join thread during close: {e}")


class LMCacheAsyncLookupServer:
    """ZMQ-based async lookup server that handles lookup and prefetch
    requests using LMCacheEngine."""

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
        self.push_socket = get_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            "ipc",
            zmq.PUSH,  # type: ignore[attr-defined]
            "connect",
        )
        self.pull_socket = get_zmq_socket(
            self.ctx,
            worker_socket_path,
            "ipc",
            zmq.PULL,  # type: ignore[attr-defined]
            "bind",
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        logger.info(
            "lmcache lookup server start with"
            f" scheduler socket path {scheduler_socket_path}, "
            f"worker socket path {worker_socket_path}"
        )
        self.thread = threading.Thread(
            target=self.process_requests_from_scheduler, daemon=True
        )
        self.thread.start()

        # The four parts are [hash, offset, lookup_id, request_configs]
        self.num_parts = 4

    def process_requests_from_scheduler(self):
        while self.running:
            frames = self.pull_socket.recv_multipart(copy=False)
            num_frames = len(frames)
            assert num_frames % self.num_parts == 0
            for i in range(0, num_frames, self.num_parts):
                lookup_id = frames[i].bytes.decode("utf-8")

                hash_frame = frames[i + 1]
                hashes = self.decoder.decode(hash_frame)

                offset_frame = frames[i + 2]
                offsets = self.decoder.decode(offset_frame)

                request_configs_str = frames[i + 3].bytes.decode("utf-8")
                request_configs = None
                if request_configs_str != "":
                    request_configs = {}
                    request_configs_list = request_configs_str.split("@")
                    for kv in request_configs_list:
                        kvs = kv.split("%", 1)
                        if len(kvs) != 2:
                            raise ValueError(f"Unexpected tags_str: {kvs}")
                        request_configs[kvs[0]] = kvs[1]

                self.lmcache_engine.async_lookup_and_prefetch(
                    lookup_id=lookup_id,
                    hashes=hashes,
                    offsets=offsets,
                    pin=True,
                    request_configs=request_configs,
                )

    def send_response_to_scheduler(self, lookup_id: str, num_hit_tokens: int):
        lookup_id_buf = lookup_id.encode("utf-8")
        num_hit_tokens_buf = num_hit_tokens.to_bytes(4, "big")
        self.push_socket.send_multipart([lookup_id_buf, num_hit_tokens_buf], copy=False)

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            for s in self.push_sockets:
                s.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning(f"Failed to join thread during close: {e}")
