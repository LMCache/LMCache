# L1 byte-budget eviction

`L1EvictionController` computes a per-cycle target from allocated physical
bytes times `EvictionConfig.eviction_ratio`. It passes an `EvictionByteBudget`
to the configured policy. LRU, ARC and isolated LRU preserve their replacement
order and choose one batch of whole objects until this target is covered or
eligible keys run out. A final object may exceed the target. ARC's adaptive
partition and ghost capacity remain entry-based; this changes batch accounting,
not ARC's replacement algorithm. `noop` continues to return no actions.

Existing L2 and coordinator calls omit the optional keyword and retain their
key-count semantics and rounding. An isolated policy still requires an explicit
cache salt; the budget does not introduce a global path for that policy.

## Physical sizes and eligibility

`L1Manager.get_evictable_size(key)` estimates the physical bytes that an ordinary
non-forced `delete` can free, including allocator padding:

- An unlocked resident counts only if no live staging writer protects the key.
- Expired staging allocations count even when a reader or another writer keeps
  the resident allocation in place.
- Live staging allocations and read-locked resident allocations do not count.

This accessor takes no manager lock because the policy already holds its own
lock while selecting. Manager lifecycle notifications take the manager lock
before entering the policy; acquiring the manager lock inside selection would
invert that order. It snapshots staging entries and retains object references
while inspecting their sizes and lock state.

The result is advisory. `delete` rechecks locks under the manager lock, and its
existing lifecycle callbacks update the policy only for confirmed removals.
Concurrent stores, reads or TTL expiration can change the bytes ultimately
reclaimed. A cycle does not retry individual victims or spin until total usage
falls by the target: such retries could chase concurrent allocations and scan
the same locked prefix repeatedly. Remaining pressure is reconsidered on the
next ordinary controller tick.

The policy looks up sizes only for the candidate prefix it visits; it does not
sum every resident allocation or maintain a second persistent size index.
Selection and deletion remain batched, with no per-object delete RPC or extra
manager-lock acquisition during selection.

## Validation

The normal CPU MP-service reproduction is in
[issue #5837](https://github.com/LMCache/LMCache/issues/5837). It uses eight objects
and 64 KiB allocated in an 80 KiB pool, watermark 0.65, and ratio 0.25. Seven
4 KiB objects followed by one 36 KiB object require four old objects to meet the
16 KiB byte target; eight uniform 8 KiB objects require two. Both the key-count
and byte targets are integral. The native regression also uses padded logical
sizes to verify physical accounting. Policy tests cover count-mode compatibility,
bounded size inspection, eligibility, partial budgets and ARC ghost updates.
