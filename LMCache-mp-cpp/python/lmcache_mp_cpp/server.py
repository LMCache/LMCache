# SPDX-License-Identifier: Apache-2.0
"""C++-backed LMCache MP server launcher."""

# Future
from __future__ import annotations

# Standard
import argparse
import threading
import time

# Third Party
from lmcache_mp_cpp.storage_manager import CxxTieredStorageManager
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import add_storage_manager_args
from lmcache.v1.mp_observability.config import (
    DEFAULT_OBSERVABILITY_CONFIG,
    ObservabilityConfig,
    add_observability_args,
    init_observability,
    parse_args_to_observability_config,
)
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.server import MPCacheEngine, add_handler_helper
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


class CxxBackedMPCacheEngine(MPCacheEngine):
    """MPCacheEngine with the storage tier replaced by C++."""

    def __init__(
        self,
        storage_manager: CxxTieredStorageManager,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ) -> None:
        self.gpu_contexts = {}
        self.gpu_context_meta = {}
        self.chunk_size = chunk_size
        self.lock = threading.Lock()
        self.storage_manager = storage_manager  # type: ignore[assignment]
        self.token_hasher = TokenHasher(
            chunk_size=chunk_size,
            hash_algorithm=hash_algorithm,
        )
        self.session_manager = SessionManager(self.token_hasher)

        # First Party
        from lmcache.v1.mp_observability.event_bus import get_event_bus

        self._event_bus = get_event_bus()
        self._prefetch_jobs = {}
        self._prefetch_job_lock = threading.Lock()
        self._setup_metrics()


def run_cpp_cache_server(
    mp_config: MPServerConfig,
    dram_capacity_bytes: int,
    disk_path: str,
    obs_config: ObservabilityConfig = DEFAULT_OBSERVABILITY_CONFIG,
    return_engine: bool = False,
    start_prometheus_http_server: bool = True,
):
    event_bus = init_observability(
        obs_config,
        start_prometheus_http_server=start_prometheus_http_server,
    )

    if obs_config.trace_level is not None:
        logger.warning(
            "Trace recording is not implemented for lmcache_mp_cpp storage yet; "
            "continuing without a trace recorder."
        )

    storage_manager = CxxTieredStorageManager(
        dram_capacity_bytes=dram_capacity_bytes,
        disk_path=disk_path,
    )
    engine = CxxBackedMPCacheEngine(
        storage_manager=storage_manager,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
    )

    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=context,
    )

    add_handler_helper(server, RequestType.REGISTER_KV_CACHE, engine.register_kv_cache)
    add_handler_helper(
        server, RequestType.UNREGISTER_KV_CACHE, engine.unregister_kv_cache
    )
    add_handler_helper(server, RequestType.STORE, engine.store)
    add_handler_helper(server, RequestType.LOOKUP, engine.lookup)
    add_handler_helper(
        server, RequestType.QUERY_PREFETCH_STATUS, engine.query_prefetch_status
    )
    add_handler_helper(
        server,
        RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        engine.query_prefetch_lookup_hits,
    )
    add_handler_helper(server, RequestType.FREE_LOOKUP_LOCKS, engine.free_lookup_locks)
    add_handler_helper(server, RequestType.RETRIEVE, engine.retrieve)
    add_handler_helper(server, RequestType.CLEAR, engine.clear)
    add_handler_helper(server, RequestType.GET_CHUNK_SIZE, engine.get_chunk_size)
    add_handler_helper(server, RequestType.PING, engine.ping)
    add_handler_helper(server, RequestType.END_SESSION, engine.end_session)
    add_handler_helper(server, RequestType.NOOP, engine.debug)
    add_handler_helper(
        server,
        RequestType.REPORT_BLOCK_ALLOCATION,
        engine.report_block_allocations,
    )

    server.add_affinity_thread_pool(
        [RequestType.STORE, RequestType.RETRIEVE],
        max_workers=mp_config.max_gpu_workers,
    )
    server.add_normal_thread_pool(
        [
            RequestType.LOOKUP,
            RequestType.QUERY_PREFETCH_STATUS,
            RequestType.QUERY_PREFETCH_LOOKUP_HITS,
            RequestType.FREE_LOOKUP_LOCKS,
            RequestType.END_SESSION,
            RequestType.CLEAR,
            RequestType.PING,
            RequestType.REPORT_BLOCK_ALLOCATION,
        ],
        max_workers=mp_config.max_cpu_workers,
    )

    logger.info(
        "LMCache C++-backed ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )
    if hasattr(torch_dev, "init"):
        torch_dev.init()
    else:
        logger.warning(
            "Backend '%s' does not support init(), skipping device init",
            torch_device_type,
        )
    server.start()

    if return_engine:
        return server, engine

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down C++-backed server...")
        event_bus.stop()
        server.close()
        engine.close()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server with C++ DRAM/disk storage"
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    parser.add_argument(
        "--cxx-dram-size-gb",
        type=float,
        default=None,
        help="C++ DRAM tier size in GB. Defaults to --l1-size-gb.",
    )
    parser.add_argument(
        "--cxx-disk-path",
        type=str,
        required=True,
        help="Directory for the C++ disk tier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    obs_config = parse_args_to_observability_config(args)
    dram_gb = args.cxx_dram_size_gb
    if dram_gb is None:
        dram_gb = args.l1_size_gb
    run_cpp_cache_server(
        mp_config=mp_config,
        dram_capacity_bytes=int(float(dram_gb) * (1 << 30)),
        disk_path=args.cxx_disk_path,
        obs_config=obs_config,
    )


if __name__ == "__main__":
    main()
