Lazy KV Cache Offload
=====================

Lazy offload lets the vLLM ``LMCacheMPConnector`` defer GPU-to-LMCache store
operations and submit finished requests in FIFO batches. It moves store work
out of the normal per-step path, which can improve serving throughput when
immediate cache availability is less important than reducing per-request
offload overhead.

This feature is disabled by default and is available only with vLLM and
``LMCacheMPConnector``. The LMCache server needs no additional flags.

Enable Lazy Offload
-------------------

Pass the lazy-offload options in vLLM's ``kv_connector_extra_config``. For
example, the following configuration starts offloading after 20 requests have
finished and selects up to 10 requests per FIFO batch:

.. code-block:: bash

   vllm serve Qwen/Qwen3-14B \
       --kv-transfer-config \
       '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"localhost","lmcache.mp.port":5555,"lmcache.mp.lazy_offload":true,"lmcache.mp.lazy_offload_policy":"FIFO","lmcache.mp.lazy_offload_threshold":20,"lmcache.mp.lazy_offload_select_count":10}}'

The options are:

.. list-table::
   :header-rows: 1
   :widths: 36 14 50

   * - Key
     - Default
     - Description
   * - ``lmcache.mp.lazy_offload``
     - ``false``
     - Enable deferred, batched store submission.
   * - ``lmcache.mp.lazy_offload_policy``
     - ``FIFO``
     - Select finished requests in insertion order. ``FIFO`` is currently the
       only supported policy.
   * - ``lmcache.mp.lazy_offload_threshold``
     - ``100``
     - Number of finished pending requests required to make a batch eligible
       for offload.
   * - ``lmcache.mp.lazy_offload_select_count``
     - ``10``
     - Maximum number of finished requests selected each time the threshold is
       met.

How It Works
------------

Lazy offload changes when stores are submitted, but it uses the same LMCache
server store path as the default eager behavior:

.. code-block:: text

   new KV blocks
       |
       v
   buffer store metadata and GPU block hashes
       |
       v
   request finishes -> add it to the FIFO-ready count
       |
       | ready count >= threshold, on a non-empty scheduling step
       v
   select at most lazy_offload_select_count requests
       |
       v
   verify block hashes -> protect valid blocks -> submit asynchronous stores
       |
       v
   all vLLM workers report completion -> release protected GPU blocks

Before a request is selected, lazy offload does not pin its GPU blocks. vLLM
may reuse those blocks for other requests. The connector records each block's
hash when it buffers the store metadata and checks the hash again before
offload. If a block has been reused, the connector logs a warning and skips the
stale store metadata instead of caching incorrect KV data. A later lookup for
the missing data therefore becomes a cache miss and vLLM recomputes the tokens;
inference correctness is unaffected.

The trigger is evaluated only on scheduling steps that process at least one
token. Pending requests below the threshold, or the remaining tail after
traffic becomes idle, are not flushed automatically. Lazy offload is therefore
best suited to continuous serving workloads rather than jobs that require
every request's KV cache to be persisted immediately.

Selected requests are removed from the ready count. For example, with a
threshold of 20 and a select count of 10, the first trigger submits 10 requests
and leaves 10 ready. Another 10 requests must finish before the count reaches
20 and triggers the next batch. Configure both values as positive integers.

Tuning
------

- Lower ``lmcache.mp.lazy_offload_threshold`` when caches must become reusable
  sooner or when traffic arrives in short bursts. This causes more frequent
  offload batches.
- Raise the threshold to defer more store work, accepting a longer delay and a
  greater chance that vLLM reuses a pending GPU block before it is offloaded.
- Raise ``lmcache.mp.lazy_offload_select_count`` to drain more requests per
  trigger. Larger values create larger store bursts; smaller values spread the
  work across more threshold crossings.
- Use ``lmcache.mp.lazy_offload_threshold=1`` for the earliest eligibility,
  while remembering that submission still needs a later non-empty scheduling
  step.

To confirm that the connector enabled lazy offload, check the vLLM log for a
message similar to:

.. code-block:: text

   lazy offload enabled with FIFO policy, offload threshold: 20
