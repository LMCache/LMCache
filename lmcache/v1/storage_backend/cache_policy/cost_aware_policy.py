# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)


class _ChunkMetadata:
    """Internal metadata holder for tracked cache chunks."""

    __slots__ = (
        "estimated_recompute_tokens",
        "memory_size_bytes",
        "last_access_time",
        "insertion_index",
        "observation_count",
    )

    def __init__(
        self,
        estimated_recompute_tokens: Optional[float],
        memory_size_bytes: Optional[float],
        last_access_time: float,
        insertion_index: int,
        observation_count: int = 0,
    ) -> None:
        self.estimated_recompute_tokens = estimated_recompute_tokens
        self.memory_size_bytes = memory_size_bytes
        self.last_access_time = last_access_time
        self.insertion_index = insertion_index
        self.observation_count = observation_count


class CostAwareEvictionPolicy(BaseCachePolicy[KeyType, dict[KeyType, Any]]):
    """
    Compute-Cost-Aware Eviction Policy with Recency Decay for LMCache.

    Calculates chunk eviction score according to:
        1. observed_recompute_tokens = total_request_tokens - chunk_start
        2. estimated_recompute_tokens = EWMA(observed_recompute_tokens, alpha)
        3. cost_density = estimated_recompute_tokens / memory_size_bytes
        4. score = cost_density / (1.0 + age_seconds / half_life_seconds)

    The candidate with the absolute LOWEST score is selected for eviction first.

    Known Limitations:
    The recomputation estimate is derived from request position, but the policy
    does not enforce structural prefix dependencies or anti-orphan constraints.
    Cross-tier placement and tier demotion are handled outside this local policy.
    """

    def __init__(
        self,
        half_life_seconds: float = 60.0,
        cost_ewma_alpha: float = 0.2,
    ) -> None:
        """
        Initialize CostAwareEvictionPolicy.

        Args:
            half_life_seconds: Time in seconds after which cost density decays by half.
            cost_ewma_alpha: EWMA smoothing factor for updating recompute estimates (0 < alpha <= 1).

        Raises:
            ValueError: If parameters are non-positive, out of range, or non-finite.
        """
        if (
            not isinstance(half_life_seconds, (int, float))
            or math.isnan(half_life_seconds)
            or math.isinf(half_life_seconds)
            or half_life_seconds <= 0
        ):
            raise ValueError(
                f"half_life_seconds must be a finite positive float, got {half_life_seconds!r}"
            )

        if (
            not isinstance(cost_ewma_alpha, (int, float))
            or math.isnan(cost_ewma_alpha)
            or math.isinf(cost_ewma_alpha)
            or not (0 < cost_ewma_alpha <= 1.0)
        ):
            raise ValueError(
                f"cost_ewma_alpha must be a finite float in range (0, 1], got {cost_ewma_alpha!r}"
            )

        self.half_life_seconds = float(half_life_seconds)
        self.cost_ewma_alpha = float(cost_ewma_alpha)

        self.metadata: Dict[KeyType, _ChunkMetadata] = {}
        self._next_insertion_index: int = 0

        logger.info(
            "Initializing CostAwareEvictionPolicy (half_life_seconds=%.2f, cost_ewma_alpha=%.2f)",
            self.half_life_seconds,
            self.cost_ewma_alpha,
        )

    def init_mutable_mapping(self) -> dict[KeyType, Any]:
        """
        Initialize a mutable mapping for cache storage.

        Returns:
            An empty dictionary for cache storage.
        """
        return {}

    def _extract_memory_bytes(
        self,
        value: Any = None,
        explicit_memory_bytes: Optional[int] = None,
    ) -> Optional[float]:
        """
        Extract physical/allocated memory size in bytes from object methods or explicit args.

        Extraction order:
        1. Explicit memory_size_bytes argument.
        2. value.get_physical_size(), when available and valid (> 0).
        3. value.get_size(), when available and valid (> 0).
        4. Explicit metadata fields (memory_size_bytes, memory_bytes, size_bytes).
        5. Missing-metadata fallback (None).
        """
        if explicit_memory_bytes is not None:
            if (
                not isinstance(explicit_memory_bytes, (int, float))
                or math.isnan(explicit_memory_bytes)
                or math.isinf(explicit_memory_bytes)
                or explicit_memory_bytes <= 0
            ):
                raise ValueError(
                    f"memory_size_bytes must be a finite positive number, got {explicit_memory_bytes!r}"
                )
            return float(explicit_memory_bytes)

        if value is None:
            return None

        # Prefer get_physical_size()
        if hasattr(value, "get_physical_size"):
            try:
                psize = value.get_physical_size()
                if (
                    isinstance(psize, (int, float))
                    and psize > 0
                    and not math.isnan(psize)
                    and not math.isinf(psize)
                ):
                    return float(psize)
            except (AttributeError, TypeError, ValueError):
                pass

        # Fallback to get_size()
        if hasattr(value, "get_size"):
            try:
                gsize = value.get_size()
                if (
                    isinstance(gsize, (int, float))
                    and gsize > 0
                    and not math.isnan(gsize)
                    and not math.isinf(gsize)
                ):
                    return float(gsize)
            except (AttributeError, TypeError, ValueError):
                pass

        # Explicit typed metadata fields
        for attr in ("memory_size_bytes", "memory_bytes", "size_bytes"):
            if hasattr(value, attr):
                try:
                    val = getattr(value, attr)
                    if (
                        isinstance(val, (int, float))
                        and val > 0
                        and not math.isnan(val)
                        and not math.isinf(val)
                    ):
                        return float(val)
                except (AttributeError, TypeError, ValueError):
                    pass
            elif isinstance(value, dict) and attr in value:
                try:
                    val = value[attr]
                    if (
                        isinstance(val, (int, float))
                        and val > 0
                        and not math.isnan(val)
                        and not math.isinf(val)
                    ):
                        return float(val)
                except (AttributeError, TypeError, ValueError):
                    pass

        return None

    def _update_recompute_ewma(
        self,
        key: KeyType,
        observed_cost: float,
    ) -> float:
        """
        Update recompute estimate for key using EWMA.
        """
        if (
            not isinstance(observed_cost, (int, float))
            or math.isnan(observed_cost)
            or math.isinf(observed_cost)
            or observed_cost <= 0
        ):
            raise ValueError(
                f"observed_cost must be a finite positive float, got {observed_cost!r}"
            )

        obs_float = float(observed_cost)
        meta = self.metadata.get(key)
        if meta is None or meta.estimated_recompute_tokens is None:
            return obs_float
        else:
            old_est = meta.estimated_recompute_tokens
            return (
                self.cost_ewma_alpha * obs_float
                + (1.0 - self.cost_ewma_alpha) * old_est
            )

    def calculate_score(
        self,
        key: KeyType,
        current_time: Optional[float] = None,
    ) -> float:
        """
        Calculate the cost-aware score for a single chunk key.

        Returns:
            Calculated float score, or float("-inf") if untracked or missing metadata.
        """
        meta = self.metadata.get(key)
        if meta is None:
            return float("-inf")
        if meta.estimated_recompute_tokens is None or meta.memory_size_bytes is None:
            return float("-inf")

        if current_time is None:
            current_time = time.monotonic()

        age_seconds = max(0.0, current_time - meta.last_access_time)
        cost_density = meta.estimated_recompute_tokens / meta.memory_size_bytes
        time_decay = 1.0 + (age_seconds / self.half_life_seconds)

        return cost_density / time_decay

    def put(
        self,
        key: KeyType,
        value: Any = None,
        total_request_tokens: Optional[int] = None,
        chunk_start: Optional[int] = None,
        memory_size_bytes: Optional[int] = None,
        estimated_recompute_tokens: Optional[float] = None,
        observed_recompute_tokens: Optional[float] = None,
    ) -> None:
        """
        Store or update metadata for a cache entry.
        """
        self.update_on_put_with_metadata(
            key,
            cache_obj=value,
            total_request_tokens=total_request_tokens,
            chunk_start=chunk_start,
            memory_size_bytes=memory_size_bytes,
            estimated_recompute_tokens=estimated_recompute_tokens,
            observed_recompute_tokens=observed_recompute_tokens,
        )

    def update_on_put(self, key: KeyType) -> None:
        """
        BaseCachePolicy interface callback when a cache chunk is stored.
        """
        self.update_on_put_with_metadata(key)

    def update_on_put_with_metadata(
        self,
        key: KeyType,
        cache_obj: Any = None,
        **metadata: Any,
    ) -> None:
        """
        Update internal policy metadata when a cache object is stored.
        """
        now = time.monotonic()

        # Validate explicit position parameters if provided
        total_req = metadata.get("total_request_tokens")
        c_start = metadata.get("chunk_start")

        if total_req is not None:
            if (
                not isinstance(total_req, (int, float))
                or math.isnan(total_req)
                or math.isinf(total_req)
                or total_req < 0
            ):
                raise ValueError(
                    f"total_request_tokens must be a non-negative finite integer, got {total_req!r}"
                )

        if c_start is not None:
            if (
                not isinstance(c_start, (int, float))
                or math.isnan(c_start)
                or math.isinf(c_start)
                or c_start < 0
            ):
                raise ValueError(
                    f"chunk_start must be a non-negative finite integer, got {c_start!r}"
                )

        if total_req is not None and c_start is not None:
            if c_start >= total_req and total_req > 0:
                raise ValueError(
                    f"chunk_start ({c_start}) must be strictly less than total_request_tokens ({total_req})"
                )

        # Determine observed recompute tokens
        obs_recompute = metadata.get("observed_recompute_tokens")
        if obs_recompute is None and total_req is not None and c_start is not None:
            obs_recompute = float(max(1, total_req - c_start))

        if obs_recompute is not None:
            if (
                not isinstance(obs_recompute, (int, float))
                or math.isnan(obs_recompute)
                or math.isinf(obs_recompute)
                or obs_recompute <= 0
            ):
                raise ValueError(
                    f"observed_recompute_tokens must be a finite positive float, got {obs_recompute!r}"
                )

        explicit_est_recompute = metadata.get("estimated_recompute_tokens")
        if explicit_est_recompute is not None:
            if (
                not isinstance(explicit_est_recompute, (int, float))
                or math.isnan(explicit_est_recompute)
                or math.isinf(explicit_est_recompute)
                or explicit_est_recompute <= 0
            ):
                raise ValueError(
                    f"estimated_recompute_tokens must be a finite positive float, got {explicit_est_recompute!r}"
                )

        # Resolve recompute estimate
        new_recompute: Optional[float] = None
        if explicit_est_recompute is not None:
            new_recompute = float(explicit_est_recompute)
        elif obs_recompute is not None:
            new_recompute = self._update_recompute_ewma(key, float(obs_recompute))
        elif key in self.metadata:
            new_recompute = self.metadata[key].estimated_recompute_tokens

        # Resolve memory size
        explicit_mem = metadata.get("memory_size_bytes")
        extracted_mem = self._extract_memory_bytes(
            value=cache_obj, explicit_memory_bytes=explicit_mem
        )
        new_mem = extracted_mem if extracted_mem is not None else (
            self.metadata[key].memory_size_bytes if key in self.metadata else None
        )

        obs_count = (
            self.metadata[key].observation_count + (1 if obs_recompute is not None else 0)
            if key in self.metadata
            else (1 if obs_recompute is not None else 0)
        )

        if key in self.metadata:
            meta = self.metadata[key]
            meta.estimated_recompute_tokens = new_recompute
            meta.memory_size_bytes = new_mem
            meta.last_access_time = now
            meta.observation_count = obs_count
        else:
            self._next_insertion_index += 1
            self.metadata[key] = _ChunkMetadata(
                estimated_recompute_tokens=new_recompute,
                memory_size_bytes=new_mem,
                last_access_time=now,
                insertion_index=self._next_insertion_index,
                observation_count=obs_count,
            )

    def update_cost_observation(
        self,
        key: KeyType,
        observed_recompute_tokens: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        """
        Record a cost observation for key using EWMA without altering recency or last_access_time.
        """
        obs_recompute = observed_recompute_tokens
        if obs_recompute is None:
            obs_recompute = metadata.get("observed_recompute_tokens")

        if obs_recompute is not None and key in self.metadata:
            obs_float = float(obs_recompute)
            if (
                math.isnan(obs_float)
                or math.isinf(obs_float)
                or obs_float <= 0
            ):
                raise ValueError(
                    f"observed_recompute_tokens must be a finite positive float, got {obs_recompute!r}"
                )

            meta = self.metadata[key]
            if meta.estimated_recompute_tokens is None:
                meta.estimated_recompute_tokens = obs_float
            else:
                meta.estimated_recompute_tokens = (
                    self.cost_ewma_alpha * obs_float
                    + (1.0 - self.cost_ewma_alpha) * meta.estimated_recompute_tokens
                )
            meta.observation_count += 1

    def access(
        self,
        key: KeyType,
        cache_dict: Any = None,
        cache_obj: Any = None,
        observed_recompute_tokens: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        """
        Record access/hit for a key. Refreshes recency and optional memory size.
        Does NOT update estimated_recompute_tokens unless a real observed_recompute_tokens is explicitly provided.
        """
        now = time.monotonic()
        val = cache_obj
        if val is None and isinstance(cache_dict, dict):
            val = cache_dict.get(key)

        if key in self.metadata:
            meta = self.metadata[key]
            meta.last_access_time = now

            # Refresh memory size if available
            extracted_mem = self._extract_memory_bytes(value=val)
            if extracted_mem is not None:
                meta.memory_size_bytes = extracted_mem

            # Only update cost EWMA if real observation is explicitly passed
            if observed_recompute_tokens is not None:
                self.update_cost_observation(
                    key, observed_recompute_tokens=observed_recompute_tokens
                )
        else:
            self.update_on_put_with_metadata(
                key,
                cache_obj=val,
                observed_recompute_tokens=observed_recompute_tokens,
                **metadata,
            )

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: dict[KeyType, Any],
    ) -> None:
        """
        BaseCachePolicy interface callback on cache hit.
        """
        self.access(key, cache_dict=cache_dict)

    def update_on_force_evict(self, key: KeyType) -> None:
        """
        BaseCachePolicy interface callback when a key is force-evicted by backend.
        """
        self.metadata.pop(key, None)

    def get_evict_candidates(
        self,
        cache_dict: dict[KeyType, Any],
        num_candidates: int = 1,
    ) -> List[KeyType]:
        """
        Select candidate keys for eviction.

        Ordering Rules:
        1. Candidates with can_evict == False are skipped.
        2. Candidates without valid cost metadata (untrusted) are ranked BEFORE fully scored candidates.
        3. Among missing-metadata candidates, order by oldest last_access_time first.
        4. Among fully scored candidates, order by score ascending.
        5. Tie-breaker for equal scores: older last_access_time first.
        6. Final tie-breaker: insertion_index ascending.
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

        def candidate_sort_key(k: KeyType) -> Tuple[int, float, float, int]:
            meta = self.metadata.get(k)
            if (
                meta is None
                or meta.estimated_recompute_tokens is None
                or meta.memory_size_bytes is None
            ):
                # Category 0: Missing metadata (untrusted -> rank first for eviction)
                last_access = meta.last_access_time if meta is not None else 0.0
                insert_idx = meta.insertion_index if meta is not None else 0
                return (0, 0.0, last_access, insert_idx)
            else:
                # Category 1: Valid metadata
                age_seconds = max(0.0, current_time - meta.last_access_time)
                cost_density = meta.estimated_recompute_tokens / meta.memory_size_bytes
                time_decay = 1.0 + (age_seconds / self.half_life_seconds)
                score = cost_density / time_decay
                return (1, score, meta.last_access_time, meta.insertion_index)

        sorted_candidates = sorted(candidate_keys, key=candidate_sort_key)
        return sorted_candidates[: min(num_candidates, len(sorted_candidates))]

    def evict(
        self,
        cache_dict: Optional[dict[KeyType, Any]] = None,
        num_candidates: int = 1,
    ) -> Union[List[KeyType], KeyType, None]:
        """
        Select candidate key(s) for eviction.
        Does NOT remove metadata directly; backend calls update_on_force_evict(key) after eviction.
        """
        target_dict = (
            cache_dict if cache_dict is not None else {k: None for k in self.metadata}
        )
        candidates = self.get_evict_candidates(
            target_dict, num_candidates=num_candidates
        )
        if num_candidates == 1:
            return candidates[0] if candidates else None
        return candidates

    def remove_next(
        self,
        cache_dict: Optional[dict[KeyType, Any]] = None,
    ) -> Optional[KeyType]:
        """
        Select and return the single candidate chunk key with the lowest score.
        """
        res = self.evict(cache_dict=cache_dict, num_candidates=1)
        if isinstance(res, list):
            return res[0] if res else None
        return res
