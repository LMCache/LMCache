# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from typing import Optional

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
from lmcache.v1.token_database import ChunkedTokenDatabase

logger = init_logger(__name__)


"""
The kv controller use `(instance_id, worker_id)` -> [location -> set[chunk_hash]] 
as kv_pool. When the the number of instance is small and stable, the time complexity 
of `lookup` in kv controller is O(n). If the number of instance is large or unknown, 
the time complexity will degrade to O(n^2), and the ReverseIndexKVController is a 
better choice.
"""


class KVController:
    def __init__(self) -> None:
        # Mapping from `(instance_id, worker_id)` -> [location -> set[chunk_hash]]
        self.kv_pool: dict[tuple[str, int], dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # TODO(Jiayi): remove this hardcode
        self.token_database = ChunkedTokenDatabase()

        # Track sequence discontinuity count for metrics
        self.seq_discontinuity_count = 0

        self._setup_metrics()

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.kv_pool_keys_count.set_function(self._get_kv_pool_size)
            prometheus_logger.kv_op_seq_discontinuity_count.set_function(
                lambda: self.seq_discontinuity_count
            )

    def _get_kv_pool_size(self) -> int:
        total_size = 0
        for _, location_kvs in self.kv_pool.items():
            for _, kvs in location_kvs.items():
                total_size += len(kvs)
        return total_size

    def _exists(
        self,
        key: int,
        exclude_instance_id: Optional[str] = None,
        exclude_location: Optional[str] = None,
    ) -> Optional[tuple[str, int, str]]:
        """
        Check if a key exists in the KV pool.

        :param int key: The key to check.

        :param str exclude_instance_id: The instance ID to exclude.

        :param str exclude_location: The location to exclude.

        :return: A tuple of (instance_id, worker_id, location) if the key exists,
        None otherwise.
        """
        for (instance_id, worker_id), location_kvs in self.kv_pool.items():
            if exclude_instance_id is not None and instance_id == exclude_instance_id:
                continue
            for location, kvs in location_kvs.items():
                if exclude_location is not None and location == exclude_location:
                    continue
                if key in kvs:
                    return instance_id, worker_id, location
        return None

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

        last_seq_num = self.reg_controller.registry.get_seq_num(
            instance_id, worker_id, location
        )

        if last_seq_num is None:
            # First message from this source
            self.reg_controller.registry.update_seq_num(
                instance_id, worker_id, location, seq_num
            )
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
        self.reg_controller.registry.update_seq_num(
            instance_id, worker_id, location, seq_num
        )

    def post_init(self, reg_controller, cluster_executor):
        """
        Post initialization of the KV controller.
        """
        self.reg_controller = reg_controller
        self.cluster_executor = cluster_executor

    async def deregister(self, instance_id: str, worker_id: int) -> None:
        """
        Deregister all kv chunks of an instance-worker.
        """
        report_id = (instance_id, worker_id)
        if report_id in self.kv_pool:
            del self.kv_pool[report_id]

    ############################################################
    # Process OrchMsg
    # recv request from user
    ############################################################

    async def clear(self, msg: ClearMsg) -> ClearRetMsg:
        """
        Clear kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("clear", msg)

    async def pin(self, msg: PinMsg) -> PinRetMsg:
        """
        Pin kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("pin", msg)

    async def compress(self, msg: CompressMsg) -> CompressRetMsg:
        """
        Compress kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("compress", msg)

    async def decompress(self, msg: DecompressMsg) -> DecompressRetMsg:
        """
        Decompress kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("decompress", msg)

    async def move(self, msg: MoveMsg) -> MoveRetMsg:
        """
        Move kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("move", msg)

    async def check_finish(self, msg: CheckFinishMsg) -> CheckFinishRetMsg:
        """
        Check if an event is finished.
        """
        return await self.cluster_executor.execute("check_finish", msg)

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
            result = self._exists(key)
            if result is None:
                break
            matched_instance = result[0]
            matched_location = result[2]
            layout_info[matched_instance] = (matched_location, end)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)

    ############################################################
    # Process KVOperationMsg
    # we do not need to return anything
    ############################################################

    async def admit(self, msg: KVAdmitMsg) -> None:
        """
        Admit a new kv chunk.
        """
        report_id = (msg.instance_id, msg.worker_id)
        self.kv_pool[report_id][msg.location].add(msg.key)

    async def evict(self, msg: KVEvictMsg) -> None:
        """
        Evict a kv chunk.
        """
        report_id = (msg.instance_id, msg.worker_id)
        location = msg.location
        key = msg.key

        if (
            report_id not in self.kv_pool
            or location not in self.kv_pool[report_id]
            or key not in self.kv_pool[report_id][location]
        ):
            return

        self.kv_pool[report_id][location].remove(key)
        if not self.kv_pool[report_id][location]:
            del self.kv_pool[report_id][location]
        if not self.kv_pool[report_id]:
            del self.kv_pool[report_id]

    ############################################################
    # Process WorkerReqMsg
    # we must add try-except block, if any error occurs, we should
    # return WorkerReqRetMsg, otherwise the worker will hang or timeout.
    ############################################################

    # TODO: improve the matching logic, return multi results
    async def batched_p2p_lookup(
        self, msg: BatchedP2PLookupMsg
    ) -> BatchedP2PLookupRetMsg:
        """
        Perform batched P2P lookup for multiple keys.

        :param BatchedP2PLookupMsg msg: The batched P2P lookup message containing keys.

        :return: A BatchedP2PLookupRetMsg containing the lookup results.
        """
        try:
            if len(msg.hashes) == 0:
                return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

            result = self._exists(msg.hashes[0], msg.instance_id)
            if result is None:
                return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

            instance_id = result[0]
            worker_id = result[1]
            location = result[2]
            peer_init_url = self.reg_controller.get_peer_init_url(
                instance_id, worker_id
            )
            if peer_init_url is None:
                raise ValueError(
                    f"Peer init url not found for {instance_id}: {worker_id}"
                )
            current_instance_keys = self.kv_pool[(instance_id, worker_id)][location]
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
        except KeyError as e:
            logger.error("Key not found in kv_pool during batched p2p lookup", e)
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])
        except Exception as e:
            logger.error("An unexpected error occurred during batched p2p lookup", e)
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])
