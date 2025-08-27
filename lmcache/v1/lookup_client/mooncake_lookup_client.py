# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class MooncakeLookupClient(LookupClientInterface):
    def __init__(
        self,
        vllm_config: "VllmConfig",
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
        # First Party
        from lmcache.integration.vllm.utils import create_lmcache_metadata

        metadata, config = create_lmcache_metadata(vllm_config)

        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed."
        )

        # First Party
        from lmcache.v1.token_database import ChunkedTokenDatabase

        assert not config.enable_blending, (
            "LMCache v1 blending is not supported in MooncakeLookupClient yet."
        )
        self.token_database = ChunkedTokenDatabase(config, metadata)

    def lookup(
        self,
        token_ids: torch.Tensor,
        lookup_id: Optional[str] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
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
        for i, ret in enumerate(rets):
            if ret != 1:  # Not found or error
                # Return the end position of the previous chunk
                # If i == 0, no chunks were found, return 0
                return ends[i - 1] if i > 0 else 0

        # All keys were found, return the last end position
        return ends[-1] if ends else 0

    def supports_producer_reuse(self) -> bool:
        """Return True as MooncakeLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        # nothing here
        pass
