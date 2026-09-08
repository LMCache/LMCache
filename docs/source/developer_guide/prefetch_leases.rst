Prefetch leases and generations
===============================

The distributed ``PrefetchRequestSpec`` API supports logical sparse
prefetches without exposing a serving engine's physical block identifiers.
Callers provide :class:`ObjectKey` chunks and may set
``policy=TrimPolicy.SPARSE``.  The result bitmap always uses the order of the
original logical key list.

Request generations
-------------------

``PrefetchRequestSpec.generation`` is an opaque, non-negative value owned by
the caller.  A serving adapter should advance it whenever a request slot is
reused or reordered.  The value is copied to ``PrefetchHandle`` and is
used as a guard when cancelling an asynchronous request; it does not affect
key lookup or sparse bitmap selection.

The generation is intentionally separate from ``ObjectKey``.  A key can be
shared by multiple requests while a stale prefetch operation is still safely
cancelled.

Lease lifecycle
---------------

For ``policy=TrimPolicy.SPARSE``, ``submit_prefetch_task`` retains the L1 read
locks for the keys reported by the handle and coordinates any L2 lookup/load
locks in the controller.  A caller has two compatible choices:

* Existing integrations can call ``query_prefetch_status(handle)``, read the
  retained objects, and finish them with ``finish_read_prefetched(keys)``.
* Lease-oriented integrations can call ``consume_prefetch_task(handle)`` after
  they have finished using the retained objects and the result bitmap is
  available.  This releases the retained read locks for that handle.  If the
  result is not ready, the method returns ``None`` and does not change
  ownership.

When a sparse prediction is no longer valid, call
``release_prefetch_task(handle)``.  It is idempotent and cancels the controller
request with the handle's generation.  If the caller has already consumed the
result bitmap through the legacy query API, it may pass the consumed keys to
``release_prefetch_task(handle, keys)`` so those locks are released as well.
The existing ``PREFIX`` and ``WARM`` paths keep their historical cleanup
semantics.

Cancellation is completed on the controller thread.  An in-flight adapter
operation is allowed to return before its L2 locks, L1 write reservation, and
any retained L1 read locks are released.  This ordering prevents eviction or
reuse from racing an asynchronous load.  Repeated cancellation/release calls
are safe, and a generation mismatch cannot cancel a newer operation using the
same logical request slot.

Minimal example
---------------

.. code-block:: python

   spec = PrefetchRequestSpec(
       keys=logical_keys,
       group_layout_descs={0: layout},
       policy=TrimPolicy.SPARSE,
       generation=request_generation,
   )
   handle = storage_manager.submit_prefetch_task(spec)

   if not storage_manager.wait_prefetch_status(handle, timeout=1.0):
       storage_manager.release_prefetch_task(handle)
   else:
       found = storage_manager.query_prefetch_status(handle)
       retained_keys = found.gather(logical_keys)
       with storage_manager.read_prefetched_results(retained_keys) as objects:
           consume(objects)
       storage_manager.release_prefetch_task(handle, retained_keys)

LMCache accepts only logical keys/chunks in this contract.  An adapter that
uses physical page tables must perform that mapping at its own boundary and
must not pass physical block IDs as ``ObjectKey`` values.  The contract does
not require SGLang or any other serving engine as a dependency.
