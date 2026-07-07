# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving Q tensors.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.sdk.kvcache import LMCacheKVCacheContext
from lmcache.sdk.kvcache import retrieve as _retrieve

logger = init_logger(__name__)


def q_model_name(model_name: str) -> str:
    """Replace model name with a query-specific model name for Q cache.
    The Q cache is stored in the same LMCache server, differentiated by
    the model name (see vllm_multi_process_adapter.py).

    Args:
        model_name: The original model name.

    Returns:
        The model name used to identify the Query from KV.
    """
    return f"__lmc_query__{model_name}"


def connect(
    url: str, http_url: str, model_name: str, timeout: float = 60.0
) -> LMCacheKVCacheContext:
    """Connect to the LMCache server and return a context for Q cache.
    The technique to get the query tensors are the same as KV cache,
    however, specific model name prefix needs to be used.

    Args:
        url: The MQ URL of the LMCache server.
        http_url: The HTTP URL of the LMCache server.
        model_name: The original model name.
        timeout: The timeout for the connection.
    """
    ctx = LMCacheKVCacheContext(
        url=url,
        http_url=http_url,
        model_name=q_model_name(model_name),
        timeout=timeout,
    )
    ctx.register_kv_caches()
    return ctx


def retrieve_query(
    ctx: LMCacheKVCacheContext, tokens: Sequence[int], cache_salt: str = ""
) -> torch.Tensor | None:
    """Return the stored query for `tokens`. The underlying machinery is
    similar to KV cache, hence only needs to wrap the `retrieve` function
    with the Q cache context and model name.

    Args:
        ctx: The context for Q cache.
        tokens: The tokens to retrieve the query for.
        cache_salt: The cache salt to use for the retrieval.
    """
    return _retrieve(ctx, tokens, cache_salt=cache_salt)
