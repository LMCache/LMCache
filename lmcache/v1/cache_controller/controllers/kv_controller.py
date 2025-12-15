# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (
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
    KVAdmitMsg,
    KVEvictMsg,
    KVOperationMsg,
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

        # Track sequence discontinuity count for metrics
        self.seq_discontinuity_count = 0

        self.registry = registry
        self.cluster_executor: Any = None

    def _setup_metrics(self) -> None:
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            # TODO(baoloongmao): Cache values for better performance
            prometheus_logger.kv_pool_keys_count.set_function(self._get_kv_pool_size)
            prometheus_logger.kv_op_seq_discontinuity_count.set_function(
                lambda: self.seq_discontinuity_count
            )

    def _get_kv_pool_size(self) -> int:
        return self.registry.get_total_kv_count()

    def check_sequence_number(self, msg: KVOperationMsg) -> None:
        """
        Check if the sequence number is continuous for the given source.

        Args:
            msg: KVOperationMsg
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        location = msg.location
        seq_num = msg.seq_num

        last_seq_num = self.registry.get_seq_num(instance_id, worker_id, location)

        if last_seq_num is None:
            # First message from this source
            self.registry.update_seq_num(instance_id, worker_id, location, seq_num)
            return

        expected_seq = last_seq_num + 1
        if seq_num != expected_seq:
            # Sequence number discontinuity detected
            self.seq_discontinuity_count += 1
            logger.warning(
                "KV operation sequence discontinuity detected: "
                "key=%s, expected_seq=%s, actual_seq=%s, gap=%s",
                (instance_id, worker_id, location),
                expected_seq,
                seq_num,
                seq_num - expected_seq,
            )

        # Update tracker with current sequence number
        self.registry.update_seq_num(instance_id, worker_id, location, seq_num)

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

    async def admit(self, msg: KVAdmitMsg) -> None:
        """
        Admit a new kv chunk.
        """
        self.registry.admit_kv(msg.instance_id, msg.worker_id, msg.location, msg.key)

    async def evict(self, msg: KVEvictMsg) -> None:
        """
        Evict a kv chunk.
        """
        self.registry.evict_kv(msg.instance_id, msg.worker_id, msg.location, msg.key)

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
        if len(msg.hashes) == 0:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        result = self.registry.find_kv(
            msg.hashes[0],
            exclude_instance_id=msg.instance_id,
        )
        if result is None:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        instance_id = result.instance_id
        worker_id = result.worker_id
        location = result.location
        assert self.reg_controller is not None
        peer_init_url = self.reg_controller.get_peer_init_url(instance_id, worker_id)
        assert peer_init_url is not None
        current_instance_keys = self.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        num_hit_chunks = 0
        for key in msg.hashes:
            if key not in current_instance_keys:
                break
            num_hit_chunks += 1

        return BatchedP2PLookupRetMsg(
            layout_info=[
                (instance_id, location, num_hit_chunks, peer_init_url),
            ]
        )
