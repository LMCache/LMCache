# Store batch context

`LMCACHE_MP_TRACE_CONTEXT=1` captures a writer's W3C headers when its L1 keys
are reserved for writing. A later completion callback can run without that
writer's Python context. `StoreListener.pop_pending_batch()` returns keys and
their captured contributors atomically to the controller.

The controller submits the batch under `mp.l2.store.schedule`, a root span
with links to contributing writers. It measures scheduling, not asynchronous
backend I/O. Failed submission ends the span with ERROR status and no error
description, events, keys, or payload attributes. The original exception is
re-raised and the worker context is restored.

Tracking is bounded to 10,000 keys, 8 contributors per key, and 128 distinct
batch links. L1 deletion, prefetch-only write completion, and listener close
remove retained contexts. Eviction affects links only; keys remain pending.
An entirely unsampled batch does not create a new scheduling span.
It installs a valid, independent, unsampled context while the handler runs.
This preserves the sampling decision for tracers using a parent-based sampler
without selecting a writer as the batch parent. A custom sampler can override
that decision. The caller's context is restored after success or failure.

L2 events capture this context, but the current L2 subscribers do not create
I/O spans from it. Backend completion tracing and native queues are separate
work; this change covers the controller's synchronous submission boundary.

Tests use real controller queues, Linux poll/eventfd, CPU buffers, an L1 test
fixture, and the existing MockL2Adapter. They do not establish compatibility
with a native storage backend or GPU execution.
