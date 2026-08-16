# SPDX-License-Identifier: Apache-2.0
"""Shared data types for lazy-offload policies."""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


@dataclass
class PendingStoreItem:
    """Store metadata selected for one request epoch by any drain policy."""

    request_id: str
    epoch: int = 0
    metadatas: list[tuple["LMCacheMPRequestMetadata", dict[int, bytes]]] = field(
        default_factory=list
    )
