# SPDX-License-Identifier: Apache-2.0
"""Metadata shared by the SGLang unified LMCache integration."""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

# Third Party
import torch

if TYPE_CHECKING:
    # Third Party
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_radix_cache import NodeId


@dataclass
class LMCacheLookupOperation:
    request_id: str
    token_ids: list[int]
    local_hit_tokens: int
    cache_salt: str
    submission_future: Any = None
    completion_future: Any = None
    total_hit_tokens: Optional[int] = None
    locks_held: bool = False
    lock_start: int = 0


@dataclass(frozen=True)
class SGLangKVComponentGroup:
    """One SGLang KV address space exposed as one LMCache engine group."""

    name: str
    kv_tensors: tuple[torch.Tensor, ...]
    sliding_window_size: int = -1
    tokens_per_block: int = 0
    slots_per_block: int = 0
    tensor_rows_per_block: tuple[int, ...] = ()
    recurrent_state: bool = False


@dataclass
class LMCacheLoadOperation:
    request_id: str
    token_ids: list[int]
    start: int
    end: int
    local_hit_tokens: int
    device_indices: torch.Tensor
    future: Any
    lookup: LMCacheLookupOperation
    result: Optional[bool] = None

    def query(self) -> bool:
        return self.result is not None or bool(self.future.query())


@dataclass
class LMCacheStoreOperation:
    request_id: str
    start: int
    end: int
    future: Any
    result: Optional[bool] = None

    def query(self) -> bool:
        return self.result is not None or bool(self.future.query())


@dataclass
class LMCacheExternalFlow:
    key: "RadixKey"
    lookup: LMCacheLookupOperation
    total_hit: Optional[int] = None
    local_hit_tokens: Optional[int] = None
    load: Optional[LMCacheLoadOperation] = None
    anchor_node: Optional["NodeId"] = None
    anchor_lock: Optional["DecLockRefParams"] = None
    mamba_value: Optional[torch.Tensor] = None
    request_mamba_value: Optional[torch.Tensor] = None
    allocated_request_mamba_for_load: bool = False
    load_req: Optional["Req"] = None
    free_mamba_after_load: bool = False
    loaded_skip_tokens: int = 0
    released_skip_tokens: int = 0
    prefix_published: bool = False
    load_completed: bool = False
    retire_requested: bool = False
    cancelled: bool = False


@dataclass
class LMCachePendingStore:
    operation: LMCacheStoreOperation
    node_id: "NodeId"
    lock_params: "DecLockRefParams"
