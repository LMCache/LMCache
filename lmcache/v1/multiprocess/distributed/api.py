# SPDX-License-Identifier: Apache-2.0
"""
Defines the data structures that will be used by the
distributed storage manager public functions

Could be implemented by native code in the future
"""

# Standard
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

logger = init_logger(__name__)


@dataclass(frozen=True)
class ObjectKey:
    """
    The unique identifier for an object in the distributed storage manager
    """

    chunk_hash: bytes
    """ Content hash of this particular chunk """

    model_name: str
    """ Name of the model this chunk belongs to """

    kv_rank: int
    """ The rank that uniquely identifies the slice of the KV cache """

    @staticmethod
    def IntHash2Bytes(chunk_hash: int) -> bytes:
        # NOTE: this is only used by tests
        return chunk_hash.to_bytes(4, byteorder="big")

    @staticmethod
    def Bytes2IntHash(chunk_hash: bytes) -> int:
        # NOTE: this is only used by tests
        return int.from_bytes(chunk_hash, byteorder="big") & ((1 << 64) - 1)

    @staticmethod
    def ComputeKVRank(
        world_size: int,
        global_rank: int,
        local_world_size: int,
        local_rank: int,
    ) -> int:
        """
        Compute the kv_rank from world_size and worker_id

        Args:
            world_size (int): The total number of workers (include TP + PP)
            global_rank (int): The global worker id (from 0 to world_size - 1)
            local_world_size (int): The local world size (for local node),
                should NOT be greater than 8
            local_rank (int): The local world rank (for local node)

        Returns:
            The special KV rank (bitmap) used by the objectkey

        Example:
            In the case of TP=4, PP=2, the TP worker 1 on node 1 has:
            - world_size = 8
            - global_rank = 5
            - local_world_size = 4
            - local_rank = 1

            The output KV rank is the bitmap:
            +--head--+
            |00000000|
            |00000000|
            |00000000|
            |00000000| layers
            |00001100|
            |00001100|
            |00001100|
            |00001100|
            +--------+
        """
        # TODO(ApostaC): in the long run, we want to have the above bitmap based
        # representation for asymmetric parallelism (e.g., sharing across different
        # TP/PP settings).
        # For now, let's have a simple implementation that just
        # differentiate between different parallel setups

        # For each number, we use 8-bit, and pack them together
        return (
            (world_size << 24)
            | (global_rank << 16)
            | (local_world_size << 8)
            | local_rank
        )


@dataclass(frozen=True)
class MemoryLayoutDesc:
    """
    Describes the layout of a memory object
    """

    shapes: list[torch.Size]
    dtypes: list[torch.dtype]

    def __post_init__(self):
        if len(self.shapes) != len(self.dtypes):
            raise ValueError(
                "MemoryLayoutDesc: shapes and dtype must have the same length"
            )


def ipc_keys_to_object_keys(ipc_keys: list[IPCCacheEngineKey]) -> list[ObjectKey]:
    """
    Convert a list of IPCCacheEngineKey to a list of ObjectKey

    When the ipc key's worker id is unspecified (None), this function will generate
    (explode) multiple ObjectKeys for all workers in the world_size.

    Args:
        ipc_keys (list[IPCCacheEngineKey]): The list of IPC keys to convert

    Returns:
        list[ObjectKey]: The converted list of ObjectKey

    Note:
        For now, we expect all the ipc keys have the same world size. Although
        it won't break even if they are different, it's not the intended use case.
    """
    if not ipc_keys:
        return []

    all_world_size_same = all(
        ipc_key.world_size == ipc_keys[0].world_size for ipc_key in ipc_keys
    )
    if not all_world_size_same:
        logger.warning(
            "ipc_keys_to_object_keys: ipc keys have different world sizes. "
            "This is not expected."
        )

    storage_keys = []
    for ipc_key in ipc_keys:
        if ipc_key.worker_id is None:
            # For look up request, we want to expand to all workers
            for worker_id in range(ipc_key.world_size):
                # TODO (ApostaC): include local world size/rank info
                # in the future once it's in IPCCacheEngineKey
                kv_rank = ObjectKey.ComputeKVRank(
                    world_size=ipc_key.world_size,
                    global_rank=worker_id,
                    local_world_size=ipc_key.world_size,
                    local_rank=worker_id,
                )

                storage_keys.append(
                    ObjectKey(
                        chunk_hash=ipc_key.chunk_hash,
                        model_name=ipc_key.model_name,
                        kv_rank=kv_rank,
                    )
                )
        else:
            kv_rank = ObjectKey.ComputeKVRank(
                world_size=ipc_key.world_size,
                global_rank=ipc_key.worker_id,
                local_world_size=ipc_key.world_size,
                local_rank=ipc_key.worker_id,
            )

            storage_keys.append(
                ObjectKey(
                    chunk_hash=ipc_key.chunk_hash,
                    model_name=ipc_key.model_name,
                    kv_rank=kv_rank,
                )
            )

    return storage_keys
