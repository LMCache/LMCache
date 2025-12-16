# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    PinMsg,
    PinRetMsg,
)
from lmcache.v1.cache_controller.observability import PrometheusLogger
from lmcache.v1.cache_controller.utils import RegistryTree
from lmcache.v1.token_database import ChunkedTokenDatabase

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.controllers import RegistrationController

logger = init_logger(__name__)


"""
The kv controller use `(instance_id, worker_id)` -> [location -> set[chunk_hash]] 
as kv_pool. When the the number of instance is small and stable, the time complexity 
of `lookup` in kv controller is O(n). If the number of instance is large or unknown, 
the time complexity will degrade to O(n^2), and the ReverseIndexKVController is a 
better choice.
"""


class KVController:
    def __init__(self, registry: RegistryTree) -> None:
        # TODO(Jiayi): remove this hardcode
        self.token_database = ChunkedTokenDatabase()
        self.registry = registry
        self.cluster_executor: Any = None

    def _setup_metrics(self) -> None:
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.kv_pool_keys_count.set_function(
                self.registry.get_total_kv_count
            )
            prometheus_logger.kv_op_seq_discontinuity_count.set_function(
                self.registry.get_seq_discontinuity_count
            )

    def post_init(
        self, reg_controller: "RegistrationController", cluster_executor: Any
    ) -> None:
        """
        Post initialization of the KV controller.
        """
        self.reg_controller = reg_controller
        self.cluster_executor = cluster_executor
        self._setup_metrics()

    async def clear(self, msg: ClearMsg) -> ClearRetMsg:
        """
        Clear kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("clear", msg)

    async def pin(self, msg: PinMsg) -> PinRetMsg:
        """
        Pin kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("pin", msg)

    async def compress(self, msg: CompressMsg) -> CompressRetMsg:
        """
        Compress kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("compress", msg)

    async def decompress(self, msg: DecompressMsg) -> DecompressRetMsg:
        """
        Decompress kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("decompress", msg)

    async def move(self, msg: MoveMsg) -> MoveRetMsg:
        """
        Move kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("move", msg)

    async def check_finish(self, msg: CheckFinishMsg) -> CheckFinishRetMsg:
        """
        Check if an event is finished.
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("check_finish", msg)

    async def handle_batched_kv_operations(self, msg: BatchedKVOperationMsg) -> None:
        """Handle batched KV operations by forwarding to registry."""
        if not msg.operations:
            return

        if not self.registry.handle_batched_kv_operations(msg):
            logger.warning(
                "Failed to handle batched KV operations, instance: %s, worker: %d",
                msg.instance_id,
                msg.worker_id,
            )

    # TODO(Jiayi): The current implementation does not handle
    # the case where the prefix chunks are evicted while the
    # suffix chunk is still in the system. LMCache should guarantee
    # this does not happen.
    # TODO(Jiayi): The current implementation does not consider
    # the location of the kv chunks. It simply returns the
    # `instance_id` with longest prefix.
    # TODO(Jiayi): Need to get rid of the hash somehow
    async def lookup(self, msg: LookupMsg) -> LookupRetMsg:
        tokens = msg.tokens
        layout_info = {}
        for start, end, key in self.token_database.process_tokens(
            tokens, make_key=False
        ):
            result = self.registry.find_kv(key)
            if result is None:
                break
            matched_instance = result.instance_id
            matched_location = result.location
            layout_info[matched_instance] = (matched_location, end)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)

    # TODO: improve the matching logic, return multi results
    async def batched_p2p_lookup(
        self, msg: BatchedP2PLookupMsg
    ) -> BatchedP2PLookupRetMsg:
        """
        Perform batched P2P lookup for multiple keys.

        :param BatchedP2PLookupMsg msg: The batched P2P lookup message containing keys.

        :return: A BatchedP2PLookupRetMsg containing the lookup results.
        """
        hashes = msg.hashes
        if not hashes:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        # Single lookup to get all needed info (optimized path)
        result = self.registry.find_kv_with_worker_info(
            hashes[0], exclude_instance_id=msg.instance_id
        )
        if result is None:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        kv_info, peer_init_url, current_keys = result
        if peer_init_url is None:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        # Count hits efficiently
        num_hit_chunks = 0
        for key in hashes:
            if key not in current_keys:
                break
            num_hit_chunks += 1

        return BatchedP2PLookupRetMsg(
            layout_info=[
                (kv_info.instance_id, kv_info.location, num_hit_chunks, peer_init_url),
            ]
        )
