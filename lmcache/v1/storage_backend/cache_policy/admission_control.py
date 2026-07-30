# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import (
    BaseCachePolicy,
    KeyType,
    MapType,
)

logger = init_logger(__name__)


class _FrequencySketch:
    """
    Approximate, decaying per-key request-frequency counter.

    Not a real Count-Min Sketch (no hashing/collisions modeled) -- an
    exact dict counter with periodic halving to bound memory and let old
    popularity decay over time, matching the design validated in
    ``benchmarks/cache_policy/experiments/admission_control.py``.
    """

    def __init__(self, halve_every: int = 20_000) -> None:
        """
        Initialize the frequency sketch.

        Args:
            halve_every: Number of increments between halving passes over
                all tracked counts.

        Raises:
            ValueError: If halve_every is non-positive.
        """
        if halve_every <= 0:
            raise ValueError(f"halve_every must be positive, got {halve_every!r}")

        self._counts: Dict[Any, int] = {}
        self._halve_every = halve_every
        self._increments = 0

    def increment(self, key: Any) -> None:
        """Record one observed request for key."""
        self._counts[key] = self._counts.get(key, 0) + 1
        self._increments += 1
        if self._increments % self._halve_every == 0:
            for tracked_key in list(self._counts):
                halved = self._counts[tracked_key] // 2
                if halved > 0:
                    self._counts[tracked_key] = halved
                else:
                    del self._counts[tracked_key]

    def estimate(self, key: Any) -> int:
        """Return the current approximate request-frequency count for key."""
        return self._counts.get(key, 0)


class AdmissionControlledPolicy(BaseCachePolicy[KeyType, MapType]):
    """
    TinyLFU-style admission control, wrapping any existing cache policy.

    Delegates all standard eviction-ranking behavior (`get_evict_candidates`,
    `update_on_hit`, `update_on_put`, `update_on_force_evict`) to an inner
    policy unchanged, and adds one new capability: `should_admit`, which
    tracks an approximate global request-frequency estimate per key and
    refuses to admit a new key unless its estimated frequency exceeds the
    *coldest currently-resident key's*, by that same estimate. This
    targets a different failure mode than eviction *ranking* does -- under
    high-fan-out, mostly-one-shot traffic, the dominant cost is admitting
    low-value entries at all, each displacing something that might have
    been reused (see
    ``docs/design/v1/storage_backend/cache_policy/admission-control-policy.md``
    for the evaluation that motivated this).

    `should_admit` deliberately does **not** ask the inner policy for its
    eviction candidate to compare against, even though that would seem
    like the more natural "would you evict this key" question to ask.
    `get_evict_candidates` is documented and used everywhere else in this
    codebase as a call that's always immediately followed by actually
    evicting the returned key(s) -- some implementations (e.g.
    `LFUCachePolicy`) rely on that and mutate their own bookkeeping
    (`key_to_freq`/`freq_to_keys`) as a side effect of the call itself, not
    of a separate evict step. Calling it speculatively, as this class's
    first version did, silently corrupts that policy's internal state
    whenever `should_admit` decides to reject: the peeked key is purged
    from the inner policy's bookkeeping but never actually removed from
    `cache_dict`, so a later hit on it crashes with a `KeyError`. Comparing
    against our own frequency sketch instead is a purely additive,
    side-effect-free read that never touches the inner policy.

    Integration status: this class is a complete, correct
    ``BaseCachePolicy`` implementation usable today via
    ``get_cache_policy("ADMISSION_<INNER>")``, and behaves identically to
    ``<INNER>`` alone for every method except `should_admit`. No shipped
    storage backend calls `should_admit` yet (wiring one in, at an
    insertion call site that knows the incoming key before allocating
    space, is a separate follow-up -- see the design doc), so the
    admission-rejection behavior does not yet take effect in production
    request handling; it is fully exercised by
    `lmcache.tools.cache_policy_bench.runner`.
    """

    def __init__(
        self,
        inner_policy: BaseCachePolicy[KeyType, MapType],
        halve_every: int = 20_000,
    ) -> None:
        """
        Initialize the admission-controlled policy.

        Args:
            inner_policy: The wrapped policy responsible for eviction
                ranking and mutable-mapping construction.
            halve_every: Passed through to the internal frequency sketch.
        """
        self.inner_policy = inner_policy
        self.halve_every = halve_every
        self._sketch = _FrequencySketch(halve_every=halve_every)

        logger.info(
            "Initializing AdmissionControlledPolicy(inner=%s)",
            type(inner_policy).__name__,
        )

    def init_mutable_mapping(self) -> MapType:
        """
        Initialize a mutable mapping for cache storage.

        Return:
            The inner policy's mutable mapping, unmodified.
        """
        return self.inner_policy.init_mutable_mapping()

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: MapType,
    ) -> None:
        """
        Record the observed request and delegate to the inner policy.

        Input:
            key: an object of KeyType
            cache_dict: a dict consists of current cache
        """
        self._sketch.increment(key)
        self.inner_policy.update_on_hit(key, cache_dict)

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        """
        Record the observed request and delegate to the inner policy.

        Input:
            key: an object of KeyType
        """
        self._sketch.increment(key)
        self.inner_policy.update_on_put(key)

    def update_on_put_with_metadata(
        self,
        key: KeyType,
        cache_obj: Any = None,
        **metadata: Any,
    ) -> None:
        """
        Record the observed request and delegate to the inner policy.

        Input:
            key: an object of KeyType
            cache_obj: optional cache object (e.g. MemoryObj)
            metadata: additional metadata key-value pairs
        """
        self._sketch.increment(key)
        self.inner_policy.update_on_put_with_metadata(key, cache_obj, **metadata)

    def update_cost_observation(
        self,
        key: KeyType,
        **metadata: Any,
    ) -> None:
        """
        Delegate a cost observation to the inner policy unchanged.

        Input:
            key: an object of KeyType
            metadata: additional cost observation metadata
        """
        self.inner_policy.update_cost_observation(key, **metadata)

    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        """
        Delegate to the inner policy unchanged.

        Input:
            key: an object of KeyType
        """
        self.inner_policy.update_on_force_evict(key)

    def get_evict_candidates(
        self,
        cache_dict: MapType,
        num_candidates: int = 1,
    ) -> list[KeyType]:
        """
        Delegate eviction ranking to the inner policy unchanged.

        Input:
            cache_dict: a dict consists of current cache
            num_candidates: number of candidates to be evicted

        Return:
            The inner policy's eviction candidates, unmodified.
        """
        return self.inner_policy.get_evict_candidates(cache_dict, num_candidates)

    def should_admit(
        self,
        key: KeyType,
        cache_dict: MapType,
    ) -> bool:
        """
        Decide whether to admit a new key, per the class docstring.

        Precondition: only meaningful when the cache is at/over capacity
        -- see `BaseCachePolicy.should_admit`.

        Records this call as an observed request for `key`, regardless of
        the outcome: a key that keeps losing admission must still be able
        to accumulate frequency across repeated attempts, or it would be
        locked out forever the moment it's first rejected (since a
        rejected key never reaches `update_on_put_with_metadata`, the
        only other place frequency is recorded on the miss path).

        Deliberately does not call `get_evict_candidates` -- see the class
        docstring for why that would be unsafe for inner policies (e.g.
        `LFUCachePolicy`) whose eviction-candidate lookup mutates their own
        state as a side effect. Instead compares against the coldest
        currently-resident key by this class's own frequency estimate.

        Input:
            key: an object of KeyType for the candidate new entry
            cache_dict: a dict consists of current cache

        Return:
            True if `cache_dict` is empty (nothing to displace), or if
            key's estimated request frequency exceeds the coldest
            resident key's; False otherwise.
        """
        self._sketch.increment(key)
        if not cache_dict:
            return True
        coldest_estimate = min(self._sketch.estimate(k) for k in cache_dict)
        return self._sketch.estimate(key) > coldest_estimate
