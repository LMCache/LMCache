lmcache bench engine
====================

The ``lmcache bench engine`` command runs sustained performance benchmarks
against an inference engine (e.g., vLLM). It supports multiple workload types
that exercise different caching patterns and reports TTFT, decoding speed, and
throughput metrics.

.. code-block:: bash

   lmcache bench engine [options]

There are three ways to configure the benchmark:

1. **CLI arguments** -- pass all options on the command line.
2. **Interactive mode** -- run ``lmcache bench engine`` without required args
   and follow the step-by-step prompts.
3. **Config file** -- save a configuration to JSON and replay it with
   ``--config``.


Quick Start
-----------

**Minimal (with all required arguments):**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-qa \
       --lmcache-url http://localhost:8080

**Interactive mode (guided setup):**

.. code-block:: bash

   lmcache bench engine

The interactive mode walks you through each required setting, then asks
whether you want to configure general and workload-specific options or use
defaults.

**From a saved config file:**

.. code-block:: bash

   lmcache bench engine --engine-url http://localhost:8000 \
       --config my_bench.json

Config files contain benchmark parameters (workload, KV cache settings, etc.)
but not the engine URL, so you can reuse the same config against different
engines.

**Export a config without running the benchmark:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-qa \
       --lmcache-url http://localhost:8080 \
       --export-config my_bench.json

This resolves all auto-detected values (model name, tokens per GB) and saves
them to a portable JSON file that works without an LMCache server.

**Non-interactive mode (for scripts and CI):**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-qa \
       --lmcache-url http://localhost:8080 \
       --no-interactive

Errors immediately if any required argument is missing, instead of entering
interactive mode. Useful in automated pipelines.

If you don't have an LMCache server, you can pass ``--tokens-per-gb-kvcache``
directly instead of ``--lmcache-url``
(see :ref:`bench-tokens-per-gb` for how to find this value).


General Options
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Flag
     - Required
     - Description
   * - ``--config FILE``
     - No
     - Load configuration from a JSON file. Skips interactive mode.
       CLI flags override values in the file. The engine URL is not
       stored in config files and must be provided separately.
   * - ``--export-config FILE``
     - No
     - Export resolved configuration to a JSON file and exit. Does not
       run the benchmark. Auto-detected values (model, tokens per GB)
       are resolved and saved so the config is portable. Environment-
       specific values (engine URL, LMCache URL) are excluded.
   * - ``--no-interactive``
     - No
     - Disable interactive mode. Errors if required arguments are
       missing instead of prompting. Useful for scripts and CI.
   * - ``--engine-url URL``
     - Yes
     - Inference engine URL (e.g., ``http://localhost:8000``).
       Set ``OPENAI_API_KEY`` env var if authentication is needed.
   * - ``--workload TYPE``
     - Yes
     - Workload type: ``long-doc-qa``, ``multi-round-chat``,
       ``long-doc-permutator``, ``prefix-suffix-tuner``, or
       ``random-prefill``.
   * - ``--tokens-per-gb-kvcache N``
     - \*
     - Tokens per GB of KV cache. Required unless ``--lmcache-url`` is set.
       See :ref:`bench-tokens-per-gb` for how to find this value.
   * - ``--lmcache-url URL``
     - No
     - LMCache HTTP server URL. When provided, ``--tokens-per-gb-kvcache``
       is auto-detected from the server.
   * - ``--model NAME``
     - No
     - Model name. Auto-detected from the engine if omitted.
   * - ``--kv-cache-volume GB``
     - No
     - Target active KV cache volume in GB (default: 100).
   * - ``--seed N``
     - No
     - Random seed (default: 42).
   * - ``--output-dir DIR``
     - No
     - Directory for CSV and JSON output files (default: current directory).
   * - ``--no-csv``
     - No
     - Skip CSV export.
   * - ``--json``
     - No
     - Export a JSON summary file.
   * - ``-q`` / ``--quiet``
     - No
     - Suppress the real-time progress display.


.. _bench-tokens-per-gb:

Finding ``--tokens-per-gb-kvcache``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have an LMCache server running, the easiest approach is to pass
``--lmcache-url`` and let the tool auto-detect the value.

If you are using **vLLM without LMCache**, look for these lines in vLLM's
startup log:

.. code-block:: text

   INFO: Available KV cache memory: 12.34 GiB
   INFO: GPU KV cache size: 567,890 tokens

Then compute::

   tokens_per_gb = 567890 / 12.34 = 46,020


Workloads
---------

long-doc-qa
~~~~~~~~~~~~

Simulates repeated Q&A over long documents. Warmup sends each document once
to populate the KV cache, then benchmark queries are dispatched with
semaphore-controlled concurrency.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--ldqa-document-length``
     - 10000
     - Token length of each synthetic document.
   * - ``--ldqa-query-per-document``
     - 2
     - Number of questions asked per document.
   * - ``--ldqa-shuffle-policy``
     - random
     - Request ordering: ``random`` (shuffled) or ``tile`` (round-by-round).
   * - ``--ldqa-num-inflight-requests``
     - 3
     - Maximum concurrent in-flight requests.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-qa \
       --lmcache-url http://localhost:8080 \
       --kv-cache-volume 50 \
       --ldqa-document-length 8000 \
       --ldqa-query-per-document 4 \
       --ldqa-shuffle-policy tile


multi-round-chat
~~~~~~~~~~~~~~~~~

Simulates multi-round chat with stateful sessions. Creates concurrent user
sessions, dispatches requests at a fixed QPS rate, and records responses in
session history so each subsequent query includes prior context.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--mrc-shared-prompt-length``
     - 2000
     - System prompt token length per session.
   * - ``--mrc-chat-history-length``
     - 10000
     - Pre-filled chat history token length.
   * - ``--mrc-user-input-length``
     - 50
     - Tokens per user query.
   * - ``--mrc-output-length``
     - 200
     - Max tokens to generate per response.
   * - ``--mrc-qps``
     - 1.0
     - Target queries per second.
   * - ``--mrc-duration``
     - 60.0
     - Benchmark duration in seconds.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload multi-round-chat \
       --lmcache-url http://localhost:8080 \
       --mrc-qps 2.0 \
       --mrc-duration 120


long-doc-permutator
~~~~~~~~~~~~~~~~~~~~

Stress-tests blended KV cache reuse by sending permutations of a set of context
documents. Each request concatenates all context documents in a different order:

.. code-block:: text

   [System Prompt] + [Doc_i1] + [Doc_i2] + ... + [Doc_iN]

A single dummy warmup request is sent before the benchmark phase. Requests are
dispatched with semaphore-controlled concurrency.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--ldp-num-contexts``
     - 5
     - Number of unique context documents.
   * - ``--ldp-context-length``
     - 5000
     - Token length of each context document.
   * - ``--ldp-system-prompt-length``
     - 1000
     - Token length of the shared system prompt. Use ``0`` for no system prompt.
   * - ``--ldp-num-permutations``
     - 10
     - Number of distinct permutations to send. Capped at N! where
       N = ``--ldp-num-contexts``.
   * - ``--ldp-num-inflight-requests``
     - 1
     - Maximum concurrent in-flight requests.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-permutator \
       --lmcache-url http://localhost:8080 \
       --ldp-num-contexts 4 \
       --ldp-context-length 8000 \
       --ldp-num-permutations 24 \
       --ldp-num-inflight-requests 2


prefix-suffix-tuner
~~~~~~~~~~~~~~~~~~~

A two-pass sequential workload designed to be run **unchanged** across
three LMCache configurations to demonstrate the value of each cache tier
(L0 HBM, L1 DRAM, L2 disk):

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - Baseline
     - LMCache config
     - Targeted overflow
     - Expected pass-2 hits
   * - 1
     - vanilla vLLM (L0 only)
     - L0 (HBM)
     - none -- every request a cold prefill
   * - 2
     - vLLM + LMCache L1 + L2
     - L1 (DRAM)
     - L2 prefix hits (suffix recomputed)
   * - 3
     - vLLM + LMCache L1 + L2 + CacheBlend
     - L1 (DRAM)
     - L2 prefix hits + CacheBlend suffix hits

Set ``--kv-cache-volume`` to the size in GB of the tier you want to overflow
(L0 size for Baseline 1, L1 size for Baselines 2 and 3). The workload itself
is identical across baselines.

Each request has the layout::

   [prefix_i with unique-ID][random breaker][shared suffix]

- ``num_prefixes`` distinct prefixes, each starting with ``PREFIX_<8-hex>``
  so the prefix's tokenized hash differs across the pool.
- A fresh random 32-token breaker per request, defeating ordinary prefix
  caching past the prefix boundary.
- A single shared suffix used by every request -- the only entry CacheBlend
  can reuse.

Pass 1 (warmup) sends each prefix once to populate the cache; its stats are
discarded. Pass 2 sends them again in identical order. Because LRU evicts
the next-needed prefix on each pass-2 access, even a 1.05x overflow of the
targeted tier is enough to make every pass-2 request miss that tier and
fall through to the next one.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--psf-context-length``
     - 8000
     - Total tokens per request (prefix + breaker + suffix).
   * - ``--psf-prefix-ratio``
     - 0.8
     - Fraction of context-length used by the prefix. Must be in (0.0, 1.0).
       The remainder (minus a 32-token breaker) is the shared suffix.
   * - ``--psf-thrash``
     - 20.0
     - **Size in GB of the KV-cache tier to overflow.** Use the L0 (HBM)
       size for vanilla vLLM, or the L1 (LMCache DRAM) size for tiered
       baselines. The workload sizes its prefix pool to slightly more than
       this (5% overflow internally), enough to drive every pass-2 request
       to a miss of that tier under sequential dispatch + LRU.

The number of pass-2 (measured) requests equals the prefix pool size,
computed as
``floor(psf_thrash * 1.05 * tokens_per_gb / prefix_tokens)``.
``--kv-cache-volume`` is unused by this workload — sizing is driven solely
by ``--psf-thrash``.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload prefix-suffix-tuner \
       --lmcache-url http://localhost:8080 \
       --psf-context-length 8000 \
       --psf-prefix-ratio 0.8 \
       --psf-thrash 100

.. note::
   For the analytical-model claim "thrash ≈ L1 size → ~0% LMCache hit rate"
   to hold empirically, the LMCache server must be started with
   ``--eviction-ratio 0.99`` (default ``0.20`` only clears 20% per cycle,
   leaving ~60% of pass-1 content in cache through pass 2):

   .. code-block:: bash

      lmcache server --l1-size-gb <SIZE> --eviction-policy LRU \
          --eviction-trigger-watermark 0.80 \
          --eviction-ratio 0.99

   The workload itself sleeps 5 seconds between pass 1 (warmup) and pass 2
   (measured), so LMCache's 1Hz batched-eviction polling thread has time
   to actually run.  Without that sleep, fast benchmarks complete before
   any eviction fires.


random-prefill
~~~~~~~~~~~~~~~

Fires all requests simultaneously with ``max_tokens=1`` to measure pure
prefill performance. No warmup phase.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--rp-request-length``
     - 10000
     - Token length per prefill request.
   * - ``--rp-num-requests``
     - 50
     - Number of requests to fire.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload random-prefill \
       --lmcache-url http://localhost:8080 \
       --rp-request-length 15000 \
       --rp-num-requests 100


Interactive Mode
----------------

.. image:: /_static/bench_interactive_demo.gif
   :alt: Interactive mode demo
   :width: 100%

When ``--engine-url`` or ``--workload`` is not provided (and
``--no-interactive`` is not set), the tool enters interactive mode. It guides
you through four phases:

1. **Required settings** -- engine URL, workload type, LMCache server
   (or tokens per GB).
2. **General settings** (optional gate) -- model name, KV cache volume.
3. **Workload settings** (optional gate) -- workload-specific parameters.
4. **Summary and action** -- review configuration, then start the benchmark
   or export to a JSON file.

Each prompt focuses on a single setting. Selection prompts use arrow keys;
text and number prompts accept typed input with defaults shown in brackets.

.. code-block:: text

   ══════════════════════════════════════════════════
    lmcache bench engine -- Interactive Setup
   ══════════════════════════════════════════════════

   Engine URL
     URL of the inference engine.
     [default: http://localhost:8000] >

   Workload
     The type of benchmark workload to run.
     Use up/down to navigate, Enter to select.

     * long-doc-qa           Repeated Q&A over long documents
       multi-round-chat       Multi-turn chat with stateful sessions
       long-doc-permutator    Permutations of context documents
       prefix-suffix-tuner    Two-pass tiered KV-cache demonstrator
       random-prefill         Prefill-only requests fired simultaneously

   LMCache Server
     Do you have a running LMCache server?
     It can auto-detect KV cache size information.
     [default: Y] (Y/n) >

   ...

   ──────────────────────────────────────────────────
    Configuration Summary
   ──────────────────────────────────────────────────
     Workload:             long-doc-qa
     Model:                Qwen/Qwen3-14B
     Tokens per GB:        6553
     ...
   ──────────────────────────────────────────────────

   What would you like to do?
     * Start benchmark
       Export configuration for later use and exit

When you choose "Export configuration", all auto-detected values (model name,
tokens per GB) are resolved and saved to a portable JSON file.


Config File
-----------

Config files store benchmark parameters but **not** environment-specific
values like engine URL or LMCache URL. This lets you reuse the same config
across different environments.

You can create a config file in three ways:

1. **Interactive mode** -- choose "Export configuration" at the summary step.
2. **``--export-config``** -- resolve and export from CLI without running.
3. **Manually** -- write JSON with keys matching CLI arg names (dashes
   replaced by underscores).

Example config file:

.. code-block:: json

   {
     "model": "Qwen/Qwen3-14B",
     "workload": "long-doc-qa",
     "tokens_per_gb_kvcache": 6553,
     "kv_cache_volume": 100.0,
     "ldqa_document_length": 10000,
     "ldqa_query_per_document": 2,
     "ldqa_shuffle_policy": "random",
     "ldqa_num_inflight_requests": 3
   }

Load it with ``--config`` (engine URL must be provided separately):

.. code-block:: bash

   lmcache bench engine --engine-url http://localhost:8000 \
       --config my_bench.json

CLI arguments override config file values, so you can use a base config and
tweak individual settings:

.. code-block:: bash

   # Use saved config but override KV cache volume
   lmcache bench engine --engine-url http://localhost:8000 \
       --config my_bench.json --kv-cache-volume 200


Output
------

Terminal (real-time progress)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During the benchmark, a live progress display shows in-flight requests,
average TTFT, decode speed, and throughput. Suppress it with ``-q``.

Terminal (final summary)
~~~~~~~~~~~~~~~~~~~~~~~~~

After completion, a summary table is printed:

.. code-block:: text

   ======= Engine Benchmark Result (long-doc-qa) ========
   ---------------------- Configuration ------------------
   Engine URL:                       http://localhost:8000
   Model:                            Qwen/Qwen3-14B
   Workload:                         long-doc-qa
   ------------------------- Results ---------------------
   Successful requests:              20
   Failed requests:                  0
   Benchmark duration (s):           31.34
   Total input tokens:               200000
   Total output tokens:              2560
   Input throughput (tok/s):         6381.62
   Output throughput (tok/s):        81.69
   --------------- Time to First Token -------------------
   Mean TTFT (ms):                   313.41
   P50 TTFT (ms):                    272.83
   P90 TTFT (ms):                    587.21
   P99 TTFT (ms):                    837.32
   ------------------ Decoding Speed ---------------------
   Mean decode (tok/s):              48.23
   P50 decode (tok/s):               47.91
   P90 decode (tok/s):               42.10
   P99 decode (tok/s):               38.55
   ======================================================

CSV and JSON
~~~~~~~~~~~~~

- ``bench_results.csv`` -- per-request metrics (TTFT, latency, decode speed,
  token counts). Written by default; skip with ``--no-csv``.
- ``bench_summary.json`` -- aggregate statistics with percentiles and config
  metadata. Opt-in with ``--json``.

Both files are written to ``--output-dir`` (default: current directory).


Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - All requests succeeded.
   * - ``1``
     - One or more requests failed.