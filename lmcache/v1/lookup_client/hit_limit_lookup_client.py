# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


"""
HitLimitLookupClient now is used for test, when lookup is called, cal the cache hit,
- if the cache hit < hit_limit_upper, direct return the result
- if the cache hit > hit_limit_upper, re-compute the result by hit_limit_upper
"""


class HitLimitLookupClient(LookupClientInterface):
    def __init__(
        self, actual_lookup_client: LookupClientInterface, config: LMCacheEngineConfig
    ):
        assert config.hit_limit_upper is not None and 0 <= config.hit_limit_upper <= 1
        self.actual_lookup_client = actual_lookup_client
        self.hit_limit_upper = config.hit_limit_upper
        self.chunk_size = config.chunk_size
        logger.info(
            f"create HitLimitLookupClient succeed, the hit limit upper "
            f"is {self.hit_limit_upper}, chunk size is {self.chunk_size}"
        )

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        # get real hit tokens
        result = self.actual_lookup_client.lookup(token_ids, lookup_id, request_configs)
        if result is not None:
            total_tokens_length = len(token_ids)
            assert result <= total_tokens_length
            current_hit_rate = result / total_tokens_length
            # limit the hit tokens
            if current_hit_rate > self.hit_limit_upper:
                origin_result = result
                # align to chunk size
                new_result = (
                    int(total_tokens_length * self.hit_limit_upper)
                    // self.chunk_size
                    * self.chunk_size
                )
                # check again
                result = min(result, new_result)
                logger.debug(
                    f"hit limit upper: {self.hit_limit_upper} is smaller than "
                    f"the real hit rate {current_hit_rate}, "
                    f"the origin result is {origin_result}, "
                    f"the new result is {new_result}, the final result is {result}"
                )
        return result

    def supports_producer_reuse(self) -> bool:
        return self.actual_lookup_client.supports_producer_reuse()

    def close(self) -> None:
        self.actual_lookup_client.close()
