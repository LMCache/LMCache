# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional, Tuple, Union
import time

try:
    # Third Party
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)

DEFAULT_STORAGE_TIER_COSTS: Dict[str, float] = {
    "GPU": 1.0,
    "RAM": 10.0,
    "CPU": 10.0,
    "DISK": 100.0,
    "S3": 1000.0,
    "REMOTE": 1000.0,
}


class _ChunkMetadata:
    """Internal metadata holder for tracked cache chunks."""

    __slots__ = ("chunk_length", "storage_tier_cost", "last_access_time")

    def __init__(
        self,
        chunk_length: float,
        storage_tier_cost: float,
        last_access_time: float,
    ) -> None:
        self.chunk_length = chunk_length
        self.storage_tier_cost = storage_tier_cost
        self.last_access_time = last_access_time


class CostAwareEvictionPolicy(BaseCachePolicy[KeyType, dict[KeyType, Any]]):
    """
    Cost-Aware Cache Eviction Policy for LMCache.

    Weighs cache entries by their recomputation compute cost (chunk length)
    and storage layer retrieval latency, balanced against recency decay:

        Score = (w1 * chunk_length) + (w2 * storage_tier_cost) - (w3 * time_since_last_access)

    Where:
    - chunk_length: Number of tokens in the chunk (proxy for GPU prefill cost).
    - storage_tier_cost: Retrieval latency weight of the storage tier.
    - time_since_last_access: Elapsed time since the chunk was last accessed.

    The chunk with the absolute LOWEST score is selected for eviction first.
    """

    def __init__(
        self,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        storage_tier_cost: Optional[Dict[str, float]] = None,
        default_chunk_length: int = 256,
        default_storage_tier: str = "CPU",
        default_tier_cost: float = 10.0,
    ) -> None:
        """
        Initialize CostAwareEvictionPolicy.

        Args:
            w1: Weight for compute cost (chunk token length).
            w2: Weight for storage tier retrieval latency cost.
            w3: Weight for recency decay (time since last access penalty).
            storage_tier_cost: Optional mapping of tier name (e.g. "GPU", "DISK")
                to tier retrieval cost float.
            default_chunk_length: Default token length when unassigned.
            default_storage_tier: Default tier name when unassigned.
            default_tier_cost: Default fallback tier cost when tier is unknown.
        """
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)

        self.tier_cost_map: Dict[str, float] = dict(DEFAULT_STORAGE_TIER_COSTS)
        if storage_tier_cost is not None:
            self.tier_cost_map.update(storage_tier_cost)

        self.default_chunk_length = float(default_chunk_length)
        self.default_storage_tier = default_storage_tier
        self.default_tier_cost = float(default_tier_cost)

        self.metadata: Dict[KeyType, _ChunkMetadata] = {}

        logger.info(
            "Initializing CostAwareEvictionPolicy (w1=%.2f, w2=%.2f, w3=%.2f)",
            self.w1,
            self.w2,
            self.w3,
        )

    def init_mutable_mapping(self) -> dict[KeyType, Any]:
        """
        Initialize a mutable mapping for cache storage.

        Returns:
            An empty dictionary for cache storage.
        """
        return {}

    def _resolve_tier_cost(
        self,
        storage_tier: Optional[str] = None,
        storage_tier_cost: Optional[float] = None,
    ) -> float:
        if storage_tier_cost is not None:
            return float(storage_tier_cost)
        if storage_tier is not None:
            tier_upper = str(storage_tier).upper()
            return self.tier_cost_map.get(tier_upper, self.default_tier_cost)
        return self.default_tier_cost

    def _extract_metadata(
        self,
        key: KeyType,
        value: Any = None,
        chunk_length: Optional[int] = None,
        storage_tier: Optional[str] = None,
        storage_tier_cost: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Extract chunk length and tier cost from arguments or object properties.
        """
        resolved_length: Optional[float] = (
            float(chunk_length) if chunk_length is not None else None
        )

        if resolved_length is None and value is not None:
            for attr in ("chunk_length", "length", "num_tokens", "seq_len", "size"):
                if hasattr(value, attr):
                    val = getattr(value, attr)
                    if isinstance(val, (int, float)):
                        resolved_length = float(val)
                        break
                    elif callable(val):
                        try:
                            resolved_length = float(val())
                            break
                        except Exception:
                            pass
                elif isinstance(value, dict) and attr in value:
                    val = value[attr]
                    if isinstance(val, (int, float)):
                        resolved_length = float(val)
                        break

        if resolved_length is None and key is not None:
            for attr in ("chunk_length", "length"):
                if hasattr(key, attr):
                    val = getattr(key, attr)
                    if isinstance(val, (int, float)):
                        resolved_length = float(val)
                        break

        if resolved_length is None:
            resolved_length = self.default_chunk_length

        resolved_cost: Optional[float] = None
        if storage_tier_cost is not None:
            resolved_cost = float(storage_tier_cost)
        elif storage_tier is not None:
            resolved_cost = self._resolve_tier_cost(storage_tier=storage_tier)

        if resolved_cost is None and value is not None:
            for attr in ("storage_tier_cost", "tier_cost"):
                if hasattr(value, attr):
                    resolved_cost = float(getattr(value, attr))
                    break
                elif isinstance(value, dict) and attr in value:
                    resolved_cost = float(value[attr])
                    break
            if resolved_cost is None:
                for attr in ("storage_tier", "tier", "backend"):
                    if hasattr(value, attr):
                        tier_str = str(getattr(value, attr))
                        resolved_cost = self._resolve_tier_cost(storage_tier=tier_str)
                        break
                    elif isinstance(value, dict) and attr in value:
                        tier_str = str(value[attr])
                        resolved_cost = self._resolve_tier_cost(storage_tier=tier_str)
                        break

        if resolved_cost is None:
            resolved_cost = self._resolve_tier_cost(
                storage_tier=self.default_storage_tier
            )

        return resolved_length, resolved_cost

    def calculate_score(
        self,
        key: KeyType,
        current_time: Optional[float] = None,
    ) -> float:
        """
        Calculate the cost-aware score for a single chunk key.

        Args:
            key: The key of the chunk.
            current_time: Optional current monotonic timestamp.

        Returns:
            The calculated float score, or float("-inf") if the key is untracked.
        """
        meta = self.metadata.get(key)
        if meta is None:
            return float("-inf")
        if current_time is None:
            current_time = time.monotonic()
        time_since_last_access = current_time - meta.last_access_time
        return (
            self.w1 * meta.chunk_length
            + self.w2 * meta.storage_tier_cost
            - self.w3 * time_since_last_access
        )

    def _compute_scores_vectorized(
        self,
        candidate_keys: List[KeyType],
        current_time: float,
    ) -> Union[List[float], Any]:
        """
        Compute cost-aware scores for candidate keys using NumPy vectorization if available,
        or optimized Python list comprehension.
        """
        n = len(candidate_keys)
        if n == 0:
            return np.array([], dtype=np.float64) if HAS_NUMPY else []

        if HAS_NUMPY:
            lengths = np.empty(n, dtype=np.float64)
            tier_costs = np.empty(n, dtype=np.float64)
            last_accesses = np.empty(n, dtype=np.float64)

            for i, k in enumerate(candidate_keys):
                meta = self.metadata.get(k)
                if meta is not None:
                    lengths[i] = meta.chunk_length
                    tier_costs[i] = meta.storage_tier_cost
                    last_accesses[i] = meta.last_access_time
                else:
                    lengths[i] = self.default_chunk_length
                    tier_costs[i] = self.default_tier_cost
                    last_accesses[i] = current_time

            time_since_access = current_time - last_accesses
            scores = (
                self.w1 * lengths
                + self.w2 * tier_costs
                - self.w3 * time_since_access
            )
            return scores
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3
            def_len, def_cost = self.default_chunk_length, self.default_tier_cost
            scores_list = []
            for k in candidate_keys:
                meta = self.metadata.get(k)
                if meta is not None:
                    c_len = meta.chunk_length
                    t_cost = meta.storage_tier_cost
                    tsa = current_time - meta.last_access_time
                else:
                    c_len = def_len
                    t_cost = def_cost
                    tsa = 0.0
                scores_list.append(w1 * c_len + w2 * t_cost - w3 * tsa)
            return scores_list

    def put(
        self,
        key: KeyType,
        value: Any = None,
        chunk_length: Optional[int] = None,
        storage_tier: Optional[str] = None,
        storage_tier_cost: Optional[float] = None,
    ) -> None:
        """
        Store or update metadata for a cache entry.

        Args:
            key: Cache key.
            value: Cache value object (optional).
            chunk_length: Number of tokens in chunk (optional override).
            storage_tier: Name of storage tier (optional override).
            storage_tier_cost: Retrieval cost of storage tier (optional override).
        """
        length, tier_cost = self._extract_metadata(
            key,
            value=value,
            chunk_length=chunk_length,
            storage_tier=storage_tier,
            storage_tier_cost=storage_tier_cost,
        )
        self.metadata[key] = _ChunkMetadata(
            chunk_length=length,
            storage_tier_cost=tier_cost,
            last_access_time=time.monotonic(),
        )

    def access(self, key: KeyType, cache_dict: Any = None) -> None:
        """
        Record access to a chunk key, refreshing its last_access_time timestamp.

        Args:
            key: Cache key accessed.
            cache_dict: Optional cache storage dictionary.
        """
        if key in self.metadata:
            self.metadata[key].last_access_time = time.monotonic()
        else:
            value = cache_dict.get(key) if isinstance(cache_dict, dict) else None
            self.put(key, value=value)

    def update_on_put(self, key: KeyType) -> None:
        """
        BaseCachePolicy interface method called when a cache entry is stored.

        Args:
            key: Cache key stored.
        """
        if key not in self.metadata:
            self.put(key)
        else:
            self.metadata[key].last_access_time = time.monotonic()

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: dict[KeyType, Any],
    ) -> None:
        """
        BaseCachePolicy interface method called on cache hit.

        Args:
            key: Cache key hit.
            cache_dict: Current cache dict.
        """
        self.access(key, cache_dict=cache_dict)

    def update_on_force_evict(self, key: KeyType) -> None:
        """
        BaseCachePolicy interface method called when key is evicted.

        Args:
            key: Key force evicted.
        """
        self.metadata.pop(key, None)

    def get_evict_candidates(
        self,
        cache_dict: dict[KeyType, Any],
        num_candidates: int = 1,
    ) -> List[KeyType]:
        """
        Select candidate keys with the absolute lowest scores for eviction.

        Args:
            cache_dict: Dictionary of current cache entries.
            num_candidates: Number of candidate keys to return.

        Returns:
            List of candidate keys to evict ordered by score ascending (lowest score first).
        """
        if not cache_dict or num_candidates <= 0:
            return []

        candidate_keys: List[KeyType] = []
        for k, cache_obj in cache_dict.items():
            if hasattr(cache_obj, "can_evict") and not cache_obj.can_evict:
                continue
            candidate_keys.append(k)

        if not candidate_keys:
            return []

        current_time = time.monotonic()
        scores = self._compute_scores_vectorized(candidate_keys, current_time)

        k_num = min(num_candidates, len(candidate_keys))

        if HAS_NUMPY:
            if k_num == 1:
                min_idx = int(np.argmin(scores))
                return [candidate_keys[min_idx]]
            else:
                partition_indices = np.argpartition(scores, k_num - 1)[:k_num]
                sorted_subset = partition_indices[np.argsort(scores[partition_indices])]
                return [candidate_keys[idx] for idx in sorted_subset]
        else:
            indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])
            return [candidate_keys[idx] for idx, _ in indexed_scores[:k_num]]

    def evict(
        self,
        cache_dict: Optional[dict[KeyType, Any]] = None,
        num_candidates: int = 1,
    ) -> Union[List[KeyType], KeyType, None]:
        """
        Evict chunk(s) with the lowest score(s).

        Args:
            cache_dict: Optional dictionary of cache entries. If None, evaluates tracked metadata keys.
            num_candidates: Number of keys to evict.

        Returns:
            List of evicted keys if num_candidates > 1, or single key if num_candidates == 1.
        """
        target_dict = (
            cache_dict if cache_dict is not None else {k: None for k in self.metadata}
        )
        evict_keys = self.get_evict_candidates(
            target_dict, num_candidates=num_candidates
        )
        for key in evict_keys:
            self.metadata.pop(key, None)

        if num_candidates == 1:
            return evict_keys[0] if evict_keys else None
        return evict_keys

    def remove_next(
        self,
        cache_dict: Optional[dict[KeyType, Any]] = None,
    ) -> Optional[KeyType]:
        """
        Select, remove, and return the single chunk with the lowest score.

        Args:
            cache_dict: Optional dictionary of cache entries.

        Returns:
            Key of evicted entry, or None if empty.
        """
        res = self.evict(cache_dict=cache_dict, num_candidates=1)
        if isinstance(res, list):
            return res[0] if res else None
        return res
