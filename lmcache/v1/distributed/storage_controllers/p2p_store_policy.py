# SPDX-License-Identifier: Apache-2.0
"""
Store policy for a server that extends its DRAM with P2P peers.

Objects stay in the local DRAM L1 while it has room. Once its usage reaches a
threshold, a newly written object is stored into one peer's L1 instead and its
local copy is dropped. L2 adapters that are not P2P adapters are treated as by
the default policy.
"""

# First Party
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.store_policy import (
    StorePolicy,
    register_store_policy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor

# The L1 whose objects a P2P adapter can transfer: the local DRAM L1.
_LOCAL_DRAM = MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)


def _peer_write_config(adapter: L2AdapterDescriptor) -> P2PL2AdapterConfig | None:
    """Return the adapter's config when it stores objects into a peer."""
    config = adapter.config
    if isinstance(config, P2PL2AdapterConfig) and config.write_enabled:
        return config
    return None


class P2PStorePolicy(StorePolicy):
    """
    Keep objects local below a DRAM usage threshold, store them into a peer
    above it.

    A store task goes to at most one peer: the write-enabled P2P adapters take
    turns. A P2P adapter with writes disabled never gets a task.

    Args:
        l1_managers: Every L1 manager of the storage manager, in
            write-placement order. The one with local DRAM memory tags is the
            L1 whose usage is compared with the threshold.
    """

    def __init__(self, l1_managers: list[L1ManagerInterface]) -> None:
        super().__init__(l1_managers)
        self._local_dram_l1 = next(
            (m for m in l1_managers if m.get_memory_tags() == _LOCAL_DRAM), None
        )
        self._next_peer = 0

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[L2AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Send the keys to every non-P2P adapter and, above the usage
        threshold, to one write-enabled peer.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to the keys to store there. A peer is
            included only when the local DRAM L1's used fraction is at or
            above that peer adapter's ``write_l1_usage_threshold``.
        """
        targets = {
            ad.index: list(keys)
            for ad in adapters
            if not isinstance(ad.config, P2PL2AdapterConfig)
        }
        peers = [
            (ad.index, config)
            for ad in adapters
            if (config := _peer_write_config(ad)) is not None
        ]
        if not peers or self._local_dram_l1 is None:
            return targets

        peer_index, peer_config = peers[self._next_peer % len(peers)]
        used_bytes, total_bytes = self._local_dram_l1.get_memory_usage()
        if used_bytes >= peer_config.write_l1_usage_threshold * total_bytes:
            self._next_peer += 1
            targets[peer_index] = list(keys)
        return targets

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
        adapter: L2AdapterDescriptor,
    ) -> list[ObjectKey]:
        """
        Drop the local copies of keys that now live on a peer.

        Args:
            keys: Keys that were successfully stored to L2.
            adapter: Descriptor of the L2 adapter that stored them.

        Returns:
            All keys when ``adapter`` stores into a peer, otherwise none.
        """
        if _peer_write_config(adapter) is None:
            return []
        return list(keys)


register_store_policy("p2p", P2PStorePolicy)
