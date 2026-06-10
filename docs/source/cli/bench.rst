.. _lmcache-bench:

lmcache bench
=============

The ``lmcache bench`` command runs sustained performance benchmarks. It has
three sub-commands, each targeting a different layer of the stack:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Sub-command
     - Description
   * - ``engine``
     - Benchmark an inference engine (e.g. vLLM) with workloads that
       exercise different KV-cache reuse patterns.
   * - ``server``
     - End-to-end sanity test against a running LMCache MP cache server
       (ZMQ + HTTP). Requires the full ``lmcache`` install and a GPU.
   * - ``l2``
     - Throughput / latency benchmark against an L2 cache adapter
       (store / lookup / load).

.. code-block:: bash

   lmcache bench {engine,server,l2} [options]


.. _lmcache-bench-engine:

engine
------

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
~~~~~~~~~~~

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
~~~~~~~~~~~~~~~

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
~~~~~~~~~

long-doc-qa
^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^

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
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~

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
~~~~~~

Terminal (real-time progress)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During the benchmark, a live progress display shows in-flight requests,
average TTFT, decode speed, and throughput. Suppress it with ``-q``.

Terminal (final summary)
^^^^^^^^^^^^^^^^^^^^^^^^

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
   P99 decode (tok/s):               38.55
   ======================================================

CSV and JSON
^^^^^^^^^^^^

- ``bench_results.csv`` -- per-request metrics (TTFT, latency, decode speed,
  token counts). Written by default; skip with ``--no-csv``.
- ``bench_summary.json`` -- aggregate statistics with percentiles and config
  metadata. Opt-in with ``--json``.

Both files are written to ``--output-dir`` (default: current directory).


Exit Codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - All requests succeeded.
   * - ``1``
     - One or more requests failed.


.. _lmcache-bench-server:

server
------

The ``lmcache bench server`` command is an end-to-end sanity test for the
LMCache Multi-Process (MP) cache server. It connects to a running server
over ZMQ and exercises the full KV-cache data path for a sequence of
synthetic requests, then optionally verifies per-chunk checksums through
the HTTP API.

.. code-block:: bash

   lmcache bench server [options]

Unlike :ref:`lmcache bench engine <lmcache-bench-engine>`, this command does
**not** require an inference engine. It only needs a running LMCache MP
server (ZMQ + HTTP). GPU mode additionally requires a CUDA-capable device.
It also requires the full ``lmcache`` install (not the lightweight
``lmcache-cli`` package).


What it does
~~~~~~~~~~~~

For each sequence in ``[--start, --end)``, the tool runs two passes:

1. **Cold pass** -- ``LOOKUP`` is expected to miss, so the generated KV
   tensors are ``STORE``\ d on the server.
2. **Warm pass** -- ``LOOKUP`` is expected to hit; the tool issues
   ``RETRIEVE`` and compares the retrieved KV chunks' checksums to the
   originals.

The full RPC path exercised is::

   REGISTER_KV_CACHE → GET_CHUNK_SIZE → LOOKUP
     → QUERY_PREFETCH_STATUS → RETRIEVE → STORE
     → END_SESSION

When ``--url`` points to the server's HTTP endpoint, per-chunk checksums
are additionally cross-checked against the server-side computation, so a
mismatch between producer and consumer surfaces as a loud
``CHECKSUM MISMATCH`` log line.


Quick start
~~~~~~~~~~~

Start the MP server in one terminal:

.. code-block:: bash

   lmcache server \
       --host localhost --port 15556 \
       --chunk-size 256 --l1-size-gb 5 \
       --eviction-policy LRU --max-workers 1

Then in another terminal:

.. code-block:: bash

   lmcache bench server \
       --rpc-url tcp://localhost:15556 \
       --url http://localhost:8080

By default the tool runs forever (``--end`` unset); stop it with
``Ctrl-C`` at any time. Pass ``--end N`` for a bounded run.


Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--rpc-url URL``
     - ``tcp://localhost:5555``
     - ZMQ endpoint of the MP cache server.
   * - ``--url URL``
     - ``http://localhost:8080``
     - HTTP base URL of the server's checksum API. Used to
       verify per-chunk checksums end-to-end.
   * - ``--mode {gpu,cpu}``
     - ``gpu``
     - Run mode. ``gpu`` allocates real CUDA tensors and uses CUDA IPC
       (handle path). ``cpu`` allocates POSIX-SHM-backed tensors and
       uses the data-transfer path (gather/scatter via slot descriptors).
   * - ``--transfer-mode {auto,handle,data}``
     - ``auto``
     - Transport routing for STORE/RETRIEVE. ``handle`` forces the
       single-shot path (``REGISTER_KV_CACHE`` + ``STORE``/``RETRIEVE``).
       ``data`` forces the two-phase gather/scatter path
       (``REGISTER_KV_CACHE_NON_GPU_CONTEXT`` + ``PREPARE``/``COMMIT``).
       ``auto`` maps gpu→handle and cpu→data.
   * - ``--num-tokens N``
     - ``512``
     - Tokens per synthetic request.
   * - ``--num-blocks N``
     - ``1024``
     - Number of paged blocks allocated on the GPU.
   * - ``--block-size N``
     - ``16``
     - Tokens per paged block.
   * - ``--start N``
     - ``0``
     - First sequence number to run.
   * - ``--end N``
     - *(unset)*
     - Exclusive upper bound on sequence numbers. When omitted the
       loop runs forever.
   * - ``--interval SECS``
     - ``0.5``
     - Delay between successive sub-passes.
   * - ``--kvcache-shape-spec SPEC``
     - ``(2,1024,16,8,128):float16:32``
     - KV cache shape spec (see below).


CPU mode (no GPU)
~~~~~~~~~~~~~~~~~

``--mode cpu`` runs the same end-to-end path without a GPU. The server
runs on a CPU-only host (``StubCPUDevice``); the bench tool allocates
POSIX-SHM-backed KV tensors and exercises the full RPC path.

By default ``--mode cpu`` uses the data-transfer path (``auto`` →
``cpu→data``). To use the zero-copy SHM handle path instead, pass
``--transfer-mode handle``:

.. code-block:: bash

   # Terminal 1 -- start the LMCache server (no GPU required)
   lmcache server \
       --host localhost --port 5555 \
       --l1-size-gb 2 --eviction-policy LRU

   # Terminal 2 -- run bench in CPU + handle mode
   lmcache bench server \
       --rpc-url tcp://localhost:5555 \
       --url http://localhost:8080 \
       --mode cpu --transfer-mode handle \
       --start 0 --end 2


KV cache shape spec
~~~~~~~~~~~~~~~~~~~

The ``--kvcache-shape-spec`` flag describes how KV tensors are laid out on
the GPU. A spec is one or more groups separated by ``;``:

.. code-block:: text

   (kv_size,NB,BS,NH,HS):dtype:layers[;(...):dtype:layers...]

Fields:

* ``kv_size`` -- 2 for classical attention (separate K/V), 1 for MLA.
* ``NB`` -- number of paged blocks.
* ``BS`` -- block size (tokens per block).
* ``NH`` -- number of attention heads per layer.
* ``HS`` -- head size (in elements).
* ``dtype`` -- element dtype (e.g. ``float16``, ``bfloat16``, ``float32``,
  ``uint8``). The full set matches the keys of ``DTYPE_MAP`` in
  ``lmcache/v1/kv_layer_groups.py``.
* ``layers`` -- number of layers in this group.

Multi-group specs let you model heterogeneous layers (for example, MLA
layers + classical attention layers in the same model):

.. code-block:: bash

   lmcache bench server \
       --rpc-url tcp://localhost:15556 \
       --kvcache-shape-spec "(1,1024,16,1,128):float16:4;(2,1024,16,8,128):float16:28"

All groups must share the same ``NB`` and ``BS`` (this is a physical
constraint of paged KV). Layer counts across groups sum to the total
layer count registered with the server.

See ``parse_kvcache_shape_spec`` in ``lmcache/v1/kv_layer_groups.py``
for the authoritative parsing rules and validation errors.


Example output
~~~~~~~~~~~~~~

.. code-block:: text

   Connecting to LMCache MP Server at tcp://localhost:15556 (mode=gpu, transfer=auto) ...
   Server chunk_size = 256
   Resolved KV shape spec: (2,1024,16,8,128):float16:32
   [seq=0] LOOKUP cold:  0/2 chunks hit (1.82 ms)
   [seq=0] STORE:        2 chunks stored (1.74 ms)
   [seq=0] LOOKUP warm:  2/2 chunks hit (1.31 ms)
   [seq=0] RETRIEVE:     2 chunks retrieved (1.48 ms)
   [seq=0] CHECKSUM MATCH OK
   [seq=1] ...

Any ``CHECKSUM MISMATCH``, ``ERROR``, or Python traceback in the log
indicates a real problem worth investigating.


Exit codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Test loop completed (or was interrupted cleanly with Ctrl-C)
       with no checksum mismatches.
   * - ``1``
     - Fatal error (for example, CUDA unavailable in ``--mode gpu``,
       server unreachable, or a checksum mismatch).

.. _lmcache-bench-l2:

l2
--

The ``lmcache bench l2`` command benchmarks an L2 cache adapter
(e.g. the local-filesystem adapter) end-to-end through the same
``parse_args_to_l2_adapters_config`` + ``create_l2_adapter`` pipeline that
LMCache uses in production. Any registered adapter type can be tested
without code changes: you describe the adapter with a single JSON spec
and pick the operations to exercise.

.. code-block:: bash

   lmcache bench l2 [options]

Unlike :ref:`lmcache bench engine <lmcache-bench-engine>`, this command
does **not** require an inference engine or an LMCache MP server. It
only needs the adapter's own backing storage to be reachable (for the
``fs`` adapter, that simply means a writable directory).


What it does
~~~~~~~~~~~~

For each measured operation the tool drives the adapter directly via
its public submit/wait API:

* ``Store``  -- ``submit_store_task`` writes ``num_keys`` MemoryObjs per
  submit and waits for the store eventfd.
* ``Lookup`` -- ``submit_lookup_and_lock_task`` checks key existence
  (no payload transfer) and waits for the lookup eventfd.
* ``Load``   -- ``submit_load_task`` reads ``num_keys`` MemoryObjs per
  submit and waits for the load eventfd.

Each measured **round** issues ``--in-flight`` submits sequentially from
a single producer thread and then waits for all of them to complete; the
round duration is the wall-clock time from the first submit until the
last completion. Warmup rounds run before measurement and their results
are discarded from the final summary.

All three operations share the same key idx universe, so running
``--only store`` followed by ``--only load`` (or ``--only lookup``) with
identical other flags hits exactly the same keys. This makes the
benchmark useful as a quick regression test for adapters that should
support a clean store -> load round-trip.

.. note::

   When ``--only`` is not given, the three operations are run **in a
   single process in the order** ``store -> lookup -> load``. For
   adapters whose backing storage sits behind an OS-level cache --
   most notably the local-filesystem (``fs``) adapter, which is
   subject to the Linux **page cache** -- this means ``lookup`` and
   ``load`` will almost always observe the data that ``store`` just
   wrote still hot in RAM, and the reported numbers reflect
   page-cache throughput rather than the underlying device.

   To benchmark each operation against a cold cache, run them
   separately with ``--only`` and drop the OS caches in between, for
   example::

      lmcache bench l2 --l2-adapter '...' --only store
      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
      lmcache bench l2 --l2-adapter '...' --only lookup
      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
      lmcache bench l2 --l2-adapter '...' --only load

   For adapters that bypass the page cache (e.g. ``fs`` with
   ``"use_odirect": true``) or that talk to a remote service without
   a local cache, the default combined run is usually fine.


Quick start
~~~~~~~~~~~

Benchmark the local filesystem adapter with default parameters:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}'

This runs all three operations (store, lookup, load) with one warmup
round and one measurement round.

Stress the adapter with more in-flight submits and larger payloads:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/data/lmcache-bench","relative_tmp_dir":"tmp"}' \
       --num-keys 32 --in-flight 4 \
       --data-size-kb 512 \
       --rounds 5 --warmup-rounds 1

Run only one operation (useful to isolate store vs. load throughput):

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --only store

Lookup with a controlled hit rate (the benchmark splits the lookup keys
between a potentially-existing range and a guaranteed-non-existent
range):

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --only lookup --lookup-max-hit-rate 0.5

Enable a store -> load round-trip data integrity check on the last
measured round:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --no-skip-verify

If you prefer to keep the JSON spec out of the command line, set the
``L2_ADAPTER_JSON`` environment variable instead of passing
``--l2-adapter``:

.. code-block:: bash

   export L2_ADAPTER_JSON='{"type":"fs","base_path":"/tmp/lmcache-bench"}'
   lmcache bench l2 --num-keys 32 --in-flight 2


Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--l2-adapter JSON``
     - *(unset)*
     - L2 adapter spec as JSON with a ``"type"`` field plus
       adapter-specific configs, e.g.
       ``'{"type":"fs","base_path":"/tmp/bench"}'``. May be passed
       multiple times; only the first spec is benchmarked. If not
       provided, falls back to the ``L2_ADAPTER_JSON`` environment
       variable. Either the flag or the env var is **required**.
   * - ``--num-keys N``
     - ``32``
     - Number of keys per submit.
   * - ``--in-flight N``
     - ``1``
     - In-flight submits per round. Each round issues this many
       submits sequentially from a single producer thread, then waits
       for all of them.
   * - ``--data-size-kb N``
     - ``256``
     - Data size per key, in KiB.
   * - ``--rounds N``
     - ``1``
     - Measurement rounds per operation.
   * - ``--warmup-rounds N``
     - ``1``
     - Warmup rounds run before measurement; their results are
       discarded.
   * - ``--lookup-max-hit-rate F``
     - ``0.0``
     - Upper bound on the lookup hit rate, in ``[0, 1]``. The benchmark
       requests ``floor(N * rate)`` keys from the
       potentially-existing range and ``N - hit`` keys from a
       guaranteed-non-existent range, where ``N`` is the total number
       of lookup keys. The actual hit rate may be lower if those keys
       were never stored in this run.
   * - ``--skip-verify`` / ``--no-skip-verify``
     - ``--skip-verify``
     - Skip the store -> load round-trip data integrity check (the
       default). Pass ``--no-skip-verify`` to enable verification on
       the last measured round; this requires both ``store`` and
       ``load`` to be exercised.
   * - ``--only {lookup,store,load}``
     - *(unset)*
     - Run only the specified operation. When omitted, all three
       operations are run in the order ``store -> lookup -> load``.


Adapter JSON spec
~~~~~~~~~~~~~~~~~

The ``--l2-adapter`` JSON is parsed by
``lmcache.v1.distributed.l2_adapters.config.parse_args_to_l2_adapters_config``,
the same entry point LMCache uses everywhere else. The minimum required
field is ``type``; all remaining fields are forwarded to the adapter
implementation as keyword arguments.

Example for the local-filesystem adapter:

.. code-block:: json

   {
     "type": "fs",
     "base_path": "/data/lmcache-bench",
     "relative_tmp_dir": "tmp",
     "read_ahead_size": null,
     "use_odirect": false
   }

See the source under ``lmcache/v1/distributed/l2_adapters/`` for the
full list of adapter types and their accepted fields.


Example output
~~~~~~~~~~~~~~

Per-round progress (suppressed by ``-q``):

.. code-block:: text

   ============================================================
   L2 Adapter Benchmark
   ============================================================
     Adapter config         : FSL2AdapterConfig
     L2 adapter JSON        : {"type":"fs","base_path":"/data/lmcache-bench","relative_tmp_dir":"tmp"}
     Keys / submit          : 32
     In-flight / round      : 3
     Keys / round           : 96
     Data size / key        : 256 KB
     Data / round           : 24.00 MB
     Rounds                 : 1 (+ 1 warmup)
     Lookup max hit rate    : 0.00%
   ============================================================

   [Init] Creating adapter...
   [Init] Adapter created successfully (FSL2Adapter).

   [Store] Running 1 warmup + 1 measurement rounds...
     [Store] Round 1: 47.83 ms, success_keys=96/96
     [Store] Round 2: 46.19 ms, success_keys=96/96

   [Lookup] Running 1 warmup + 1 measurement rounds...
     [Lookup] Round 1:  5.36 ms, found=96/96
     [Lookup] Round 2:  5.03 ms, found=96/96

   [Load] Running 1 warmup + 1 measurement rounds...
     [Load] Round 1: 18.15 ms, loaded=96/96
     [Load] Round 2: 17.63 ms, loaded=96/96

Final summary (one section per exercised operation):

.. code-block:: text

   ====== L2 Adapter Benchmark Result (FSL2Adapter) =======
   ----------------------- Configuration -------------------
   Adapter:                          FSL2Adapter
   Keys / submit:                    32
   In-flight / round:                3
   Data size / key (KB):             256
   Measurement rounds:               1
   Warmup rounds:                    1
   Lookup max hit rate:              0.0
   --------------------------- Store -----------------------
   Operation:                        Store
   Rounds:                           1
   Keys / round:                     96
   Total keys:                       96
   Total success:                    96
   Duration avg (ms):                46.19
   ...
   Throughput avg (MB/s):            519.62
   Avg ops/s:                        2078.50
   Avg latency / key (ms):           0.481
   --------------------------- Lookup ----------------------
   ...
   ---------------------------- Load -----------------------
   ...
   =========================================================

Each operation section reports per-round duration statistics
(avg / min / max / p50 / p99 / std), aggregate throughput
(``avg_throughput_mbps`` -- 0 for ``Lookup`` since it has no payload),
average key-rate (``avg_ops_per_sec``), and a per-key latency.

For ``Lookup``, three additional fields are reported when
``--lookup-max-hit-rate`` is non-zero or some keys were found:

* ``Expected max hit rate`` -- the configured upper bound.
* ``Expected hit keys`` -- ``floor(total_keys * rate)``, scaled for
  the measured rounds only.
* ``Actual hit rate`` -- the measured hit rate over the kept rounds.


Round-trip verification
~~~~~~~~~~~~~~~~~~~~~~~~

When ``--no-skip-verify`` is passed and both ``store`` and ``load`` were
run, the benchmark compares the load buffers from the last measured
round against the byte pattern that ``store`` wrote (see
``make_memory_objects`` in
``lmcache/cli/commands/bench/l2_adapter_bench/data.py``):

.. code-block:: text

   [Verify] Checking store -> load data integrity for last measured round...
   [Verify] OK

Verification is **off** by default because the stricter byte pattern
also forces every key to allocate its own ``data_size`` buffer
(otherwise the runner is free to reuse a single shared buffer across
keys to keep the memory footprint small).


Exit codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - All requested operations completed and (when enabled) the
       round-trip verification passed.
   * - ``1``
     - Adapter creation failed, round-trip verification failed, or
       an operation hit a fatal error (e.g. all rounds timed out).
   * - ``2``
     - The ``--l2-adapter`` JSON / ``L2_ADAPTER_JSON`` env var was
       missing or could not be parsed.
