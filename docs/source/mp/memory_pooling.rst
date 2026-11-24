Private DAX memory pooling
==========================

``MemoryPoolController`` in ``lmcache.v1.memory_orchestrator.pooling`` coordinates
a provider such as Pangaea with an existing MP worker's private L1 or L2 DAX
hotplug API. Each worker keeps its existing allocator and KV index. No shared
mapping, shared-object service, tenant budgets, or new gRPC messages are needed.

The deployment supplies a ``MemoryPoolProvider`` adapter. Its
``acquire(consumer_id, request_id, size_bytes)`` returns a ``MemoryAllocation``
with an allocation ID, worker-local device path, byte offset and byte size.
``release(allocation_id)`` returns that exclusive range to the physical pool.
Pangaea owns physical provisioning and placement; LMCache owns its mappings.
The provider must retain grants across failures, make exact requests idempotent,
reject reuse of released request IDs, and never revoke mapped memory on a TTL.
No vendor SDK is bundled.

Call the controller from the deployment's orchestration loop:

.. code-block:: python

   import httpx
   from deployment.pangaea import create_pool  # Your provider adapter.
   from lmcache.v1.memory_orchestrator.pooling import MemoryPoolController

   pool = create_pool()
   with httpx.Client(base_url="http://mp-worker:8080", timeout=30) as worker:
       controller = MemoryPoolController(pool, worker, "node-a/mp-a", "l1")
       allocation = controller.grow("request-001", 1 << 30)
       # Later, repeat while False: L1 may still hold live allocations.
       released = controller.shrink("request-001")

The HTTP client belongs to the caller; configure its credentials using the
worker's existing authentication scheme. Use ``"l2"`` and ``adapter_index=N``
to select the Nth DAX L2 adapter. Its ``hotplug_enabled`` setting must be true.
Grow provisions first, then calls the existing hotplug-add endpoint. Shrink
waits for confirmed backend removal before calling the provider's release.
L1 drains added arenas and cannot return its initial primary arena. L2 uses
``evict`` with ``force=False``; this discards cached KV after locks and borrows
permit removal. Capacity reporting already follows these backend operations.

Safety and recovery
-------------------

Workers advertise ``X-LMCache-Instance`` on reconfiguration responses. The
controller sends ``X-LMCache-Expected-Instance`` on subsequent requests; a
replacement worker rejects the old incarnation before touching its backend.
Legacy hotplug callers may omit the header, but pool controllers require it.
Use one controller per worker/backend and do not concurrently reconfigure its
owned ranges through another caller.

An uncertain HTTP operation retains the provider lease and blocks automatic
reuse of that request. Never release backing as rollback for a failed add,
timeout, worker restart, or merely ``draining`` response. Provider release may
be retried after confirmed detach without calling the backend again.
Controller state is in memory: after controller loss, stop/fence the original
worker and reconcile its provider grants offline before releasing anything.
This small API adds no scheduling policy, background loop, or persistent
recovery service. It is separate from the shared-region ``lmcache memory``
object service. Hardware/Pangaea qualification remains deployment-specific.
