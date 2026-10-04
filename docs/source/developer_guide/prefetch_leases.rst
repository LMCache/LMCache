Logical sparse prefetch leases
==============================

The distributed ``PrefetchTaskSpec`` API accepts ``GroupedObjectKeys`` rows.
A sparse transfer uses ``fetching_policy="full"`` and one row per
serving object-group/rank, with logical chunks as columns. Its reported
indices retain the original key order,
including holes. Physical serving-engine page IDs stay outside this lookup
contract.

Request identity
----------------

The multiprocess transfer module owns the identity
``(instance_id, request_id, generation, layer_id)``. Generations are
non-negative and must advance when a serving request row is reused.
``PrefetchHandle`` remains the controller's normal handle; it does not carry
the serving generation. A late result cannot replace a newer sparse job.

Lease lifecycle
---------------

``submit_prefetch_lease(spec)`` requires ``PrefetchLockMode.LOCK`` and retains
the grouped specification and result until explicit release. Use
``query_prefetch_lease`` or ``wait_prefetch_lease`` for these handles. Ordinary
``submit_prefetch_task`` and ``query_prefetch_status`` callers retain their
existing explicit ``finish_read_prefetched`` contract.

``release_prefetch_task(handle)`` and ``cancel_prefetch_task(handle)`` drain
already accepted controller I/O to completion before releasing that lease's
L1 read locks. They do not abort an adapter's accepted load. Repeated cleanup
is safe. A cleanup exception leaves the lease reachable for retry. Optional
release keys validate ownership and cannot broaden the owned key set.

A sparse GPU retrieval uses ``read_prefetched_results(...,
release_on_error=False)``. If a later copy fails, its owner first synchronizes
the copy stream, then releases the lease. Failed synchronization retains the
job for retry. The default read context still releases on errors and retains
locks after successful reads, as before.

Minimal example
---------------

.. code-block:: python

   spec = PrefetchTaskSpec(
       key_groups=[
           GroupedObjectKeys(logical_keys, logical_keys[0].object_group_id, layout)
       ],
       fetching_policy="full",
       lock_mode=PrefetchLockMode.LOCK,
   )
   handle = storage_manager.submit_prefetch_lease(spec)
   try:
       if storage_manager.wait_prefetch_lease(handle, timeout=1.0):
           found = storage_manager.query_prefetch_lease(handle)
           retained = [
               key
               for row, hits in zip(spec.key_groups, found.hit_cells)
               for key in hits.gather(row.keys)
           ]
           with storage_manager.read_prefetched_results(
               retained, release_on_error=False
           ) as objects:
               consume_synchronously(objects)
   finally:
       storage_manager.release_prefetch_task(handle)

The six sparse RPCs use operation names on ZeroMQ and explicit protobuf
messages on gRPC. Frozen legacy numeric operation IDs are unchanged. The
serving adapter maps logical chunks and supplies GPU destinations only to its
registered transfer context. Storage and transport remain independent of
SGLang.
