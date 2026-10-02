L1 Write Overflow and Ownership
===============================

This change adds two foundations for multi-L1 support: ordered write overflow
and an owner tag on each L1 memory object. It does **not** enable multi-L1
serving through the CLI.

What changes for users?
-----------------------

Keep using the existing single-L1 configuration. No CLI migration is needed.
The DRAM, Device-DAX, and GDS options in :doc:`configuration` remain unchanged,
including the legacy ``--gds-l1-path`` options. There is no new public multi-L1
configuration or L1--L2 affinity setting.

The internal construction path accepts an ordered set of existing L1 managers
for allocation and ownership tests. It requires no L2 adapters and ``noop``
eviction. Serving reads and prefetch are rejected with multiple managers.
``memcheck()``, ``get_l1_usage()``, ``report_status()``, and
``publish_capacity()`` also raise ``ValueError`` in this internal mode rather
than report only the primary L1.
Multi-L1 read selection, cancellation, eviction, and L2 integration are deferred.

How overflow works
------------------

``StorageManager.reserve_write()`` tries the primary L1 first. It retries only
keys that returned ``OUT_OF_MEMORY`` on the next eligible L1, in explicit order.
Calls are synchronous. Each candidate is visited at most once per reservation.
The candidate order is validated and captured at construction, so later changes
to the supplied policy object do not change an existing manager's write order.

.. code-block:: text

    Reserve a batch
         |
         v
    Primary L1 ---- OUT_OF_MEMORY keys ----> Fallback L1 ----> Next L1
         |                                       |
         +-- keep successes                      +-- keep successes
         +-- stop on other errors                +-- stop on other errors

Successful keys stay on their selected L1. They are not allocated again on a
fallback. Conflicts and exceptions do not trigger overflow. If every candidate
is full, failed keys are omitted from the returned object mapping, as before.
This does not add cross-L1 deduplication.

Existing allocation batches are preserved. Overflow is **not perfect packing**.
For example, a new two-object batch may fail when each L1 has space for only one
object, even though their combined free space is sufficient. The allocator frees
partial allocations before reporting that batch as out of memory; overflow
passes the failed subset to the next L1 without splitting it into single-key
allocations.

Why each object has an owner
----------------------------

Each L1 stamps its objects with a process-local integer identity. Objects
created outside L1 start without an owner. The identity belongs to the manager,
not its position in the placement order. It is separate from the writer tag
and is not part of keys, hashes, or serialized KV metadata.

Completion captures owner identities from the objects returned by reservation.
It never repeats placement or searches for another copy of the key.

.. code-block:: text

    Reserved objects --> group keys by object owner
                                  |
                          GPU copy completes
                                  |
                                  v
                         Stream-ordered callback
                                  |
                                  v
                     Finish on each captured owner

The internal callback carries only owner integers and keys, not memory objects,
tensors, or Python pointers. Existing L1 staging keeps allocations alive. An
unknown or unset owner is an error; multi-manager completion does not guess the
primary L1. Existing single-L1 callers can still finish writes using keys alone.
Single-L1 trace recording also keeps the existing key-only completion format,
so traces can be replayed against a fresh manager with a different owner ID.

The owner tag does not replace writer validation, read locks, reference counts,
or GPU completion ordering. It is not a write generation or a crash-recovery
token. Two copies of the same key can have different owners; finishing one
does not publish the other.

Not included
------------

Backend rewiring, GPU L1, public multi-L1 configuration, L1--L2 affinity,
per-L1 reporting, concurrent lookup, migration, and shared-CXL services remain
follow-up work. CPU allocation tests do not qualify multi-L1 serving, GPU
transfers, or cross-host sharing.
