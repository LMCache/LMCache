# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

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
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        lmcache_engine: LMCacheEngine,
    ):
        """
        Initialize the bypass lookup client.

        Args:
            vllm_config: The vLLM configuration
            config: The LMCacheEngine configuration
            metadata: The LMCacheEngine metadata
            lmcache_engine: The LMCacheEngine instance to use for lookups
        """
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration should be passed."
        )

        self.lmcache_engine = lmcache_engine
        self.config = config

        # Use the token database from the provided LMCacheEngine
        self.token_database = self.lmcache_engine.token_database
        self.enable_blending = self.config.enable_blending

        logger.info("LMCacheBypassLookupClient initialized")

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        num_computed_tokens: int = 0,
    ) -> Optional[int]:
        try:
            if not self.enable_blending:
                # Process tokens to get hashes and offsets
                hashes = []
                offsets = []
                # We already have hashes here so we can skip the chunks that are already
                # in GPU cache. Don't pass num_computed_tokens to engine.
                aligned_computed_tokens = num_computed_tokens  # pre-aligned in adapter
                result = aligned_computed_tokens
                for start, end, key in self.token_database.process_tokens(
                    token_ids, make_key=False
                ):
                    if end <= aligned_computed_tokens:
                        continue
                    hashes.append(key)
                    offsets.append(end - start)
                # Return aligned_computed_tokens immediately if there is no token to
                # lookup
                if not hashes:
                    return result

                # Call LMCacheEngine lookup with hashes and offsets
                result += self.lmcache_engine.lookup(
                    hashes=hashes,
                    offsets=offsets,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                    num_computed_tokens=0,
                )
            else:
                # For blending mode, pass tokens directly
                result = self.lmcache_engine.lookup(
                    tokens=token_ids,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                    num_computed_tokens=num_computed_tokens,
                )

            return result

        except Exception as e:
            logger.error(f"Error in bypass lookup: {e}")
            return 0

    def supports_producer_reuse(self) -> bool:
        return True

    def close(self):
        # No resources to clean up for bypass client
        logger.info("LMCacheBypassLookupClient closed")
