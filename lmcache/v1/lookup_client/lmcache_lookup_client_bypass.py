# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import create_lmcache_metadata
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class LMCacheBypassLookupClient(LookupClientInterface):
    """
    Bypass lookup client that directly calls LMCacheEngine.lookup()
    instead of using ZMQ communication. This is particularly useful
    for MLA scenarios where only rank 0 needs to perform lookups.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        lmcache_engine: LMCacheEngine,
    ):
        """
        Initialize the bypass lookup client.

        Args:
            vllm_config: The vLLM configuration
            lmcache_engine: The LMCacheEngine instance to use for lookups
        """
        metadata, config = create_lmcache_metadata(vllm_config)

        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration should be passed."
        )

        self.lmcache_engine = lmcache_engine
        self.config = config

        # Initialize token database for processing tokens
        self.enable_blending = config.enable_blending
        self.token_database: TokenDatabase
        if self.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

        logger.info("LMCacheBypassLookupClient initialized")

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Perform lookup using the LMCacheEngine directly.

        Args:
            token_ids: The token IDs to lookup
            lookup_id: The lookup ID to associate with the lookup
            request_configs: The configs of the request

        Returns:
            The number of tokens that can be loaded from cache
        """
        try:
            if not self.enable_blending:
                # Process tokens to get hashes and offsets
                hashes = []
                offsets = []
                for start, end, key in self.token_database.process_tokens(
                    token_ids, make_key=False
                ):
                    hashes.append(key)
                    offsets.append(end - start)

                # Call LMCacheEngine lookup with hashes and offsets
                result = self.lmcache_engine.lookup(
                    hashes=hashes,
                    offsets=offsets,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )
            else:
                # For blending mode, pass tokens directly
                result = self.lmcache_engine.lookup(
                    tokens=token_ids,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )

            return result

        except Exception as e:
            logger.error(f"Error in bypass lookup: {e}")
            return 0

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheBypassLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        """Close the lookup client and clean up resources."""
        # No resources to clean up for bypass client
        logger.info("LMCacheBypassLookupClient closed")
