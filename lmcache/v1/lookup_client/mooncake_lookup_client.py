# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class MooncakeLookupClient(LookupClientInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        master_addr: str,
    ):
        # Third Party
        from mooncake.store import MooncakeDistributedStore

        self.store = MooncakeDistributedStore()
        self.store.setup(
            "localhost",
            "P2PHANDSHAKE",
            0,
            16 * 1024 * 1024,
            "tcp",
            "",
            master_addr,
        )

        # Initialize token database for processing tokens
        assert isinstance(
            config, LMCacheEngineConfig
        ), "LMCache v1 configuration is should be passed."

        # First Party
        from lmcache.v1.token_database import ChunkedTokenDatabase

        assert (
            not config.enable_blending
        ), "LMCache v1 blending is not supported in MooncakeLookupClient yet."
        self.token_database = ChunkedTokenDatabase(config, metadata)

        # Cache lookup results per request to avoid repeated lookups.
        # Maps lookup_id (req_id) -> number of hit tokens.
        self.reqs_status: dict[str, int] = {}

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        Return cached lookup result for the given lookup ID.

        Returns:
            -1 means not found (first time lookup needed);
            int >= 0 means cached number of hit tokens.
        """
        return self.reqs_status.get(lookup_id, -1)

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Clear cached lookup status for a given lookup ID."""
        self.reqs_status.pop(lookup_id, None)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: Optional[str] = None,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        # process token_ids to cacheengine keys
        keys = []
        ends = []
        for start, end, key in self.token_database.process_tokens(token_ids):
            assert isinstance(key, CacheEngineKey)
            keys.append(key.to_string())
            ends.append(end)

        # Use batch_is_exist to check all keys at once
        # rets is list of int: 1 = found, 0 = not found, -1 = error
        rets = self.store.batch_is_exist(keys)

        # Find the first key that doesn't exist (ret != 1)
        # This follows the same logic as cache engine's lookup method
        num_hit_tokens = 0
        for i, ret in enumerate(rets):
            if ret != 1:  # Not found or error
                num_hit_tokens = ends[i - 1] if i > 0 else 0
                break
        else:
            # All keys were found, return the last end position
            num_hit_tokens = ends[-1] if ends else 0

        # Cache the result so subsequent calls via lookup_cache() return it
        if lookup_id is not None:
            self.reqs_status[lookup_id] = num_hit_tokens

        return num_hit_tokens

    def supports_producer_reuse(self) -> bool:
        """Return True as MooncakeLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        # nothing here
        pass
