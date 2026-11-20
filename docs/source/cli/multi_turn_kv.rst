Mixed-length multi-turn KV benchmark
====================================

The checkout example ``examples/benchmarks/multi_turn_kv/benchmark.py`` runs
short and long conversations concurrently using the engine-benchmark library.
It is a qualification workload for growing KV prefixes, not a new cache
policy. Existing ``lmcache bench engine`` workloads and defaults are unchanged.

Each next prompt preserves the exact previous prompt and server-generated
token IDs, then appends new input IDs. This avoids text re-tokenization and
chat-template changes at the prefix boundary. Sessions remain sequential;
different sessions overlap, with a configurable global concurrency limit
and think time after each response.

Requirements
------------

Use a dedicated single-GPU vLLM deployment, the matching model tokenizer,
and an installed LMCache checkout with CLI dependencies. The runner uses
the vLLM ``/v1/completions`` extension ``return_token_ids`` (protocol checked
against v0.27.1). A server that omits IDs, changes the prompt, or reports
inconsistent token counts fails the request instead of approximating a
continuation. SDK retries are disabled for these requests.

The prompts are synthetic raw token continuations. They do not exercise a
model's chat template or measure answer accuracy. A deterministic seed fixes
initial contexts, per-turn additions and sampling seeds; it does not promise
identical model output across hardware, serving revisions or cache policies.

Serving setup
-------------

Follow the :doc:`../getting_started/quickstart` MP setup. For an eager-offload
experiment, an illustrative configuration is:

.. code-block:: bash

   # Use a new empty directory on the SSD being evaluated for each cold run.
   lmcache server --host 127.0.0.1 --port 5555 --http-port 8080 \
       --l1-size-gb 0.5 --eviction-policy LRU --chunk-size 256 \
       --l2-adapter '{"type":"fs","base_path":"/mnt/nvme/kv-bench/run-001"}'

In a separate terminal:

.. code-block:: bash

   vllm serve Qwen/Qwen3-0.6B --port 8000 \
       --max-model-len 16384 --kv-cache-memory-bytes 1073741824 \
       --enable-prefix-caching \
       --kv-transfer-config \
       '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5555,"lmcache.mp.lazy_offload":false}}'

The example GPU KV budget must fit the selected model's maximum sequence.
Check the actual bytes per token and available memory before adopting these
values. Host L1 and GPU KV-pool budgets are independent; workload token
counts do not set either budget. Keep model/dtype, GPU KV bytes, chunk size,
host L1 size, SSD, and serving versions fixed within a comparison.

Keep vLLM prefix caching enabled in every comparison arm, including the
baseline without external offload. Lazy offload requires it. Do not clear
GPU caches during a measured run; if all continuation hits remain on the
GPU, increase the independent session working set rather than assuming that
large individual prompts force external-cache retrieval.

Run and inspect
---------------

From the checkout root:

.. code-block:: bash

   PYTHONPATH=. python examples/benchmarks/multi_turn_kv/benchmark.py \
       --engine-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B \
       --metrics-url http://127.0.0.1:8080/metrics --require-l2 \
       --num-sessions 32 --max-inflight 4 --turns 4 \
       --short-context 1024 --long-context 8192 --turn-tokens 128 \
       --output-tokens 128 --max-context-tokens 16384 \
       --think-seconds 0.1 --seed 42 --ignore-eos \
       --ttft-slo-ms 1000 --label eager-pressure --output eager-pressure.json

``--model`` must resolve the server tokenizer locally or through HuggingFace.
The client context limit must match the server's configured limit. Validation
reserves worst-case growth through the last output before making requests.
``--ignore-eos`` requests the full generation length; omit it to retain normal
EOS stopping. Generated special-token IDs are still preserved in either case.

The report includes:

* Raw per-turn request measurements, cohort/session/turn identifiers, and
  client queueing time. ``prefix_overlap_tokens`` counts identical prior input,
  including the previous output; it is not a measured cache-hit count.
* p50/p95/p99 TTFT and client queueing, grouped by short/long cohort,
  individual session, and initial/continuation phase.
* Attempts meeting the optional queue-plus-TTFT target. Failed attempts count
  as failures; turns skipped after a failed conversation are listed separately
  in ``unattempted_turns``. One failed session does not stop healthy peers.
* Raw before/after MP counter snapshots and deltas for lookup-hit tokens,
  L2-hit tokens, completed L2-load chunks and L2 prefetch-failure chunks.
  L1/L2 eviction-chunk deltas are included when the server publishes them.

L2 activity is ``observed`` only when all three hit/load deltas are positive.
This proves observed L2-to-L1 activity on the isolated server, not that every
request retrieved KV or that latency improved. ``not_observed`` means the
available deltas did not meet that test; missing final counters, a reset, or
an unreachable endpoint produce ``unavailable``. Counter labels are summed,
so unrelated traffic invalidates attribution. Do not restart the server
between scrapes. A counter first published during the run starts at zero.

The default one-second ``--metrics-settle-seconds`` pause lets asynchronous
metric subscribers drain after requests finish. Increase it if necessary and
inspect subscriber lag; a pause alone cannot prove that no events were dropped.
This pause is outside individual request timings.

Exit status is 1 for failed inference requests, or 2 when ``--require-l2``
cannot establish L2 activity. The report is written in both cases. Omit
``--metrics-url`` and ``--require-l2`` for the no-external-offload control.

Comparison matrix
-----------------

Run each arm against a pinned serving revision, restarting serving processes
and using a fresh SSD directory between cold runs. Keep the seed and workload
configuration unchanged across policies. Initial turns and continuation turns
are reported separately; an ``initial`` label alone does not establish that
the underlying cache was cold.

.. list-table::
   :header-rows: 1
   :widths: 23 27 50

   * - Arm
     - Deployment change
     - Purpose
   * - No external offload
     - Omit ``--kv-transfer-config``; retain vLLM prefix caching
     - Local GPU-cache control; no L2 activity is expected.
   * - Eager offload
     - ``lmcache.mp.lazy_offload=false``
     - Compare with immediate offload using the same GPU and host budgets.
   * - Eviction-aware lazy offload
     - ``lmcache.mp.lazy_offload=true`` and policy ``EVICTION_AWARE``
     - Compare existing deferred offload. See :doc:`../mp/lazy_offload`.
   * - Ample host capacity
     - Increase only L1 capacity enough for the measured working set
     - Capacity control; do not require L2 activity in this arm.

For each arm, sweep ``--max-inflight`` through 1, 4 and 8 while keeping
``--num-sessions`` at 32. Then sweep paired initial lengths, for example
512/4096 and 1024/8192, at a fixed concurrency. Check that all arms fit the
configured maximum context. Repeat runs and retain each report instead of
presenting one run as a performance result. Include serving commands, commit
IDs, model/tokenizer revisions, hardware and memory budgets with the reports.

The current ``dev`` already includes byte-aware admission from PR #5889.
Evaluate any further admission experiments on separately pinned revisions;
the workload does not depend on unmerged controller changes or alter defaults.

Measurement limits
------------------

Per-token inter-token latency, actual L2 transferred bytes, per-request
fallback reasons, GPU KV-pool usage, GPU transfer/staging usage, and host L1
allocation/staging peaks require additional instrumentation. The report lists
them as unavailable. A throughput histogram cannot recover transferred bytes,
and token counts are not measurements of allocated GPU memory.

No request timeout or fault injection is added here. Server I/O lifetime,
locks and charged buffers remain owned by existing serving code. Do not infer
correct cancellation behavior from a client fallback or a completed benchmark.

Local validation uses CPU-only deterministic tokenizer fixtures and mock HTTP
streams through the real OpenAI SDK. It establishes workload/protocol/report
behavior, not model-serving correctness, achieved L2 pressure or performance.
Those claims require real GPU runs and their saved reports.
