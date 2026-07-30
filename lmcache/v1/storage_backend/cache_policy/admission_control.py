# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any, Dict, Optional

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


class WindowedAdmissionControlledPolicy(BaseCachePolicy[KeyType, MapType]):
    """
    Windowed TinyLFU (W-TinyLFU) admission control, wrapping any existing
    cache policy.

    A second, independent admission-control design, kept separate from
    ``AdmissionControlledPolicy`` (not a replacement for it) so the two can
    be directly compared -- see
    ``docs/design/v1/storage_backend/cache_policy/admission-control-policy.md``,
    "Does windowing fix Findings 5-6?" for why this exists and how the two
    compare empirically.

    ``AdmissionControlledPolicy.should_admit`` uses a strict comparison
    against the coldest resident key, which always favors incumbents on a
    tie. Under purely one-shot traffic (every key touched exactly once)
    that comparison is *always* a tie, so the cache admits nothing new for
    the rest of the run once full -- a permanent, silent freeze. Under
    generously-sized caches with light eviction pressure it also causes a
    real hit-rate regression, for the same tie-breaking reason. This class
    fixes both by construction, the way Caffeine's W-TinyLFU does: new
    keys always land in a small **window** region (plain LRU, unconditional
    admission -- ties are structurally impossible here, since nothing is
    ever rejected), and only when the window itself overflows is its
    oldest member evaluated for **promotion** into the frequency-gated
    **main** region (everything not currently in the window). An
    unpromoted window entry is queued for a real eviction the next time
    one is needed -- unlike the un-windowed design's silent rejection, so
    turnover can never stop.

    **Bug 3, caught while evaluating the first version of this class**: an
    earlier draft computed ``window_capacity`` on every call as
    ``len(cache_dict) * window_fraction`` and only ever pruned the window
    from inside ``get_evict_candidates``. That is unconditionally
    unbounded in practice: every insertion into ``cache_dict`` also
    inserts into the window (see ``update_on_put_with_metadata`` below),
    and ``get_evict_candidates`` is only ever called once per insertion
    (exactly one eviction is needed to make room for exactly one new
    key). So each cycle removed at most one key from the window *and*
    added exactly one back -- a mathematical 1-in-1-out invariant that
    left the window's size wherever it happened to be, forever. During
    the initial cache fill (before any eviction happens at all, so the
    window-pruning code never runs even once), the window grows
    unboundedly to the *entire* cache -- and then stays there permanently,
    since the invariant above never lets it shrink. The result: window
    "capacity" was never actually enforced, main never accumulated
    anything, and the whole class degenerated to exactly its
    ``inner_policy``'s behavior with zero measurable effect -- caught not
    by code review but by noticing a benchmark sweep produced eviction
    counts *identical* to the digit for the plain, non-windowed baseline,
    which is a near-impossible coincidence for a genuinely different
    algorithm.

    The fix: window capacity is an **absolute integer**
    (``window_capacity``, not a fraction of anything computed at call
    time) and is enforced **immediately at insertion**
    (``update_on_put``/``update_on_put_with_metadata``), independent of
    whether the caller happens to need an eviction right now. An insertion
    that pushes the window over capacity evaluates its oldest member on
    the spot: if frequent enough, it's promoted (untracked from the
    window, left resident in ``cache_dict`` as main -- no eviction needed
    for this, since it's already accounted for in the cache's total size);
    otherwise it's pushed onto ``self._pending_discards``, a FIFO of keys
    still physically resident in ``cache_dict`` that are owed a real
    eviction. ``get_evict_candidates`` drains that queue first (a real
    hit before the queued eviction happens rescues the key -- see
    ``update_on_hit``) before falling back to ``inner_policy`` for normal
    main-region eviction. This keeps the window genuinely, continuously
    bounded, in both the fill phase and steady state, without ever
    needing to know the cache's total capacity.

    ``inner_policy`` continues to own **main**-region eviction ranking, so
    ``WINDOWED_ADMISSION_LRU`` vs. ``WINDOWED_ADMISSION_LFU`` vs.
    ``WINDOWED_ADMISSION_COST_AWARE`` remain meaningfully different from
    each other, the same property ``AdmissionControlledPolicy`` has.
    Unlike that class, this one *does* call
    ``inner_policy.get_evict_candidates`` (for main-region eviction) --
    safely, because every call here is immediately and unconditionally
    honored (whatever it returns gets evicted, full stop). It is never
    used as a discardable "peek": the promotion/discard decision is made
    first, using only this class's own frequency sketch, and
    ``inner_policy`` is only consulted when a main-region eviction is
    already certain to happen. This is the lesson from
    ``AdmissionControlledPolicy``'s "Bug 2" (a speculative,
    possibly-discarded call corrupted ``LFUCachePolicy``'s internal
    bookkeeping) applied from the start rather than retrofitted.

    Integration status: same as ``AdmissionControlledPolicy`` -- a
    complete, correct ``BaseCachePolicy`` implementation usable today via
    ``get_cache_policy("WINDOWED_ADMISSION_<INNER>")`` and fully exercised
    by ``lmcache.tools.cache_policy_bench.runner``, but no shipped storage
    backend calls ``should_admit`` yet (moot for this class in practice,
    since it always returns ``True`` -- see below).
    """

    def __init__(
        self,
        inner_policy: BaseCachePolicy[KeyType, MapType],
        halve_every: int = 20_000,
        window_capacity: int = 20,
        promotion_threshold: int = 2,
    ) -> None:
        """
        Initialize the windowed admission-controlled policy.

        Args:
            inner_policy: The wrapped policy responsible for main-region
                eviction ranking and mutable-mapping construction.
            halve_every: Passed through to the internal frequency sketch.
            window_capacity: Absolute maximum number of keys the window
                region may hold at once, enforced at insertion time (see
                class docstring for why this is an absolute count, not a
                fraction of cache size).
            promotion_threshold: Minimum sketch-estimated frequency a
                window entry must reach to be promoted into main instead
                of queued for discard when the window overflows.

        Raises:
            ValueError: If ``window_capacity`` or ``promotion_threshold``
                is non-positive.
        """
        if window_capacity <= 0:
            raise ValueError(
                f"window_capacity must be positive, got {window_capacity!r}"
            )
        if promotion_threshold <= 0:
            raise ValueError(
                f"promotion_threshold must be positive, got {promotion_threshold!r}"
            )

        self.inner_policy = inner_policy
        self.halve_every = halve_every
        self.window_capacity = window_capacity
        self.promotion_threshold = promotion_threshold
        self._sketch = _FrequencySketch(halve_every=halve_every)
        self._window_keys: "OrderedDict[KeyType, None]" = OrderedDict()
        self._pending_discards: "OrderedDict[KeyType, None]" = OrderedDict()

        logger.info(
            "Initializing WindowedAdmissionControlledPolicy(inner=%s, "
            "window_capacity=%d, promotion_threshold=%d)",
            type(inner_policy).__name__,
            window_capacity,
            promotion_threshold,
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
        Record the observed request, delegate to the inner policy, refresh
        the key's window-LRU position if it's a window member, and rescue
        it from the pending-discard queue if it's there (a hit before its
        queued eviction happens means it has since proved its worth).

        Input:
            key: an object of KeyType
            cache_dict: a dict consists of current cache
        """
        self._sketch.increment(key)
        self.inner_policy.update_on_hit(key, cache_dict)
        if key in self._window_keys:
            self._window_keys.move_to_end(key)
        self._pending_discards.pop(key, None)

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        """
        Record the observed request, delegate to the inner policy, and
        register the key as a window member (new arrivals always start in
        the window), enforcing ``window_capacity`` immediately.

        Input:
            key: an object of KeyType
        """
        self._sketch.increment(key)
        self.inner_policy.update_on_put(key)
        self._admit_to_window(key)

    def update_on_put_with_metadata(
        self,
        key: KeyType,
        cache_obj: Any = None,
        **metadata: Any,
    ) -> None:
        """
        Record the observed request, delegate to the inner policy, and
        register the key as a window member (new arrivals always start in
        the window), enforcing ``window_capacity`` immediately.

        Input:
            key: an object of KeyType
            cache_obj: optional cache object (e.g. MemoryObj)
            metadata: additional metadata key-value pairs
        """
        self._sketch.increment(key)
        self.inner_policy.update_on_put_with_metadata(key, cache_obj, **metadata)
        self._admit_to_window(key)

    def _admit_to_window(self, key: KeyType) -> None:
        """
        Register ``key`` as the newest window member and enforce
        ``window_capacity`` on the spot: if the window is now over
        capacity, its oldest member is either promoted (frequent enough --
        left resident in ``cache_dict``, now implicitly main, no eviction
        needed) or queued in ``self._pending_discards`` for a real
        eviction at the next opportunity (see ``get_evict_candidates``).

        This is enforced here, at insertion time, rather than inside
        ``get_evict_candidates`` -- see the class docstring's "Bug 3" note
        for why computing/enforcing window size only during eviction
        leaves it unbounded.
        """
        self._window_keys[key] = None
        self._window_keys.move_to_end(key)
        if len(self._window_keys) <= self.window_capacity:
            return
        overflow_key, _ = self._window_keys.popitem(last=False)
        if self._sketch.estimate(overflow_key) < self.promotion_threshold:
            self._pending_discards[overflow_key] = None
        # Otherwise: promoted. Still resident in cache_dict, no longer
        # tracked as a window member -- implicitly main from now on.

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
        Delegate to the inner policy and drop the key from window and
        pending-discard tracking if present.

        Input:
            key: an object of KeyType
        """
        self.inner_policy.update_on_force_evict(key)
        self._window_keys.pop(key, None)
        self._pending_discards.pop(key, None)

    def get_evict_candidates(
        self,
        cache_dict: MapType,
        num_candidates: int = 1,
    ) -> list[KeyType]:
        """
        Select eviction candidates, draining ``self._pending_discards``
        (real evictions already decided at window-overflow time -- see
        ``_admit_to_window``) before falling back to the inner policy's
        normal main-region ranking.

        Input:
            cache_dict: a dict consists of current cache
            num_candidates: number of candidates to be evicted

        Return:
            Up to ``num_candidates`` keys to evict. May return fewer than
            requested if the cache has too few evictable entries.
        """
        evicted: list[KeyType] = []
        for _ in range(num_candidates):
            victim = self._next_evict_candidate(cache_dict, evicted)
            if victim is None:
                break
            evicted.append(victim)
            self._window_keys.pop(victim, None)
            self._pending_discards.pop(victim, None)
        return evicted

    def _next_evict_candidate(
        self,
        cache_dict: MapType,
        already_chosen: list[KeyType],
    ) -> Optional[KeyType]:
        for pending_key in self._pending_discards:
            if pending_key in cache_dict and pending_key not in already_chosen:
                return pending_key

        # No pending discard owed: defer to the inner policy's normal
        # eviction choice over the whole cache. Always committed (see
        # class docstring) -- safe regardless of the inner policy.
        candidates = self.inner_policy.get_evict_candidates(cache_dict, 1)
        if candidates and candidates[0] not in already_chosen:
            return candidates[0]
        return None

    def should_admit(
        self,
        key: KeyType,
        cache_dict: MapType,
    ) -> bool:
        """
        Always admits. New keys always enter the window unconditionally
        -- see the class docstring for why this is the core property that
        makes a permanent freeze structurally impossible. The actual
        frequency gate is applied later, at window-overflow time, in
        `get_evict_candidates`.

        Input:
            key: an object of KeyType for the candidate new entry
            cache_dict: a dict consists of current cache

        Return:
            Always ``True``.
        """
        return True
