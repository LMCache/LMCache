# External MP storage selectors

Tracking: [#5780](https://github.com/LMCache/LMCache/issues/5780), with the
loader scoped by [#5781](https://github.com/LMCache/LMCache/issues/5781).

## Problem

MP store and prefetch selectors are extensible through in-process registries,
but an installed external package cannot be selected by the standard server
CLI without arranging an import before argument construction. `KVLoadPolicy`
already has an explicit `package.module:Class` mechanism. The storage selectors
need the same deployment capability.

## Loading contract

`--l2-store-policy` and `--l2-prefetch-policy` accept either an existing
registered name or `package.module:Class`. Registered names take precedence.
Only an explicitly selected external class is imported. Installing a package
does not enable it, and built-in defaults stay unchanged.

The external class subclasses `StorePolicy` or `PrefetchPolicy`, respectively,
implements its abstract methods and has a no-argument constructor, matching
the existing registry factory contract. Import, attribute and constructor
errors are startup errors. Malformed paths, non-class objects, wrong base
classes and abstract classes are rejected with a diagnostic naming the
selection. The framework does not catch these errors and silently select a
different policy.

The loader does not discover entry points, modify the registry or install
files inside LMCache. These are trusted operator-installed Python classes,
not a sandbox for untrusted code.

## Execution contract

The existing controllers continue to own execution, buffers and locks.
Store selection happens after L1 writes, so refusing an L2 target cannot
avoid the preceding GPU-to-L1 copy. Prefetch selection happens after adapter
lookups and locks; it cannot reduce lookup fan-out. Plugins follow the
existing policy contracts and must not perform blocking I/O in callbacks.

This change does not add runtime hot-swapping, a new constructor context,
new cost inputs, new eviction behavior or new safety guarantees for an
arbitrary plugin's plan. It does not replace request-level `KVLoadPolicy`.

## Subsequent experiments

[#5782](https://github.com/LMCache/LMCache/issues/5782) qualifies bounded
observations and a local filesystem L2 admission example that retains L1.
[#5783](https://github.com/LMCache/LMCache/issues/5783) evaluates prefix-closed
GPU admission through the existing `lmcache.max_offload_tokens` request field,
separately in eager and lazy modes. Neither is a performance result of loading
an external class.

Conditional add-only warm prefetch in
[#5784](https://github.com/LMCache/LMCache/issues/5784) reuses EventGate,
checkpoint/replay, directory and usage views, and PrefetchService. It does not
reintroduce inventory reconciliation, a serving-path coordinator dependency,
source trimming or minimum-replica guarantees. Existing P2P ownership issues
[#5454](https://github.com/LMCache/LMCache/issues/5454) and
[#5263](https://github.com/LMCache/LMCache/issues/5263) remain prerequisites.

## Verification

Exercise both public factories, actual CLI/config parsing, and StorageManager
construction with independently importable policies. Cover built-ins,
malformed selections, import/attribute errors, wrong and abstract classes,
constructor errors and an external selection that does not register itself.
Controller integration checks establish actual store/load behavior; GPU
data-path checks are reported separately from CPU loading tests. No pooling,
model serving speedup or RDMA qualification follows from these checks.
