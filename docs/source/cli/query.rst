lmcache query
=============

The ``lmcache query`` command runs a single, metrics-first query. It has two
targets:

.. code-block:: bash

   lmcache query {engine,kvcache} [options]

* ``engine`` — send one request to a serving engine's HTTP API.
* ``kvcache`` — report how much of a prompt is already in the KV cache.


query engine
------------

The ``query engine`` subcommand sends one request to the engine API and
reports metrics. ``--prompt`` supports placeholders: ``{lmcache}`` loads
``lmcache/cli/documents/lmcache.txt``, and custom documents can be passed with
``--documents NAME=PATH``. The prompt token count is taken directly from the
usage data reported by the engine (``stream_options: {include_usage: true}``).

.. code-block:: bash

   lmcache query engine --url http://localhost:8000/v1 \
     --prompt "{lmcache} Summarize LMCache usage." \
     --format terminal \
     --max-tokens 128

.. code-block:: text

   ================= Query Engine =================
   Model:                         facebook/opt-125m
   Input tokens:                                618
   --------------- Latency Metrics ----------------
   Output tokens:                                 9
   TTFT (ms):                                 26.88
   TPOT (ms/token):                            0.91
   Total latency (ms):                        35.05
   Throughput (tokens/s):                   1100.64
   ================================================

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Required
     - Description
   * - ``--url URL``
     - Yes
     - Serving engine base URL (e.g. ``http://localhost:8000/v1``).
   * - ``--prompt TEXT``
     - Yes
     - Prompt text with optional ``{name}`` placeholders. ``{lmcache}``
       expands to the bundled sample document.
   * - ``--model ID``
     - No
     - Model ID for the serving engine. Auto-detected from the engine's
       reported usage if omitted.
   * - ``--max-tokens N``
     - No
     - Maximum completion tokens (default: 128).
   * - ``--timeout SECS``
     - No
     - HTTP timeout in seconds (default: 30).
   * - ``--documents NAME=PATH``
     - No
     - Load file text for ``{NAME}`` in ``--prompt``. Accepts one or more
       ``NAME=PATH`` values.
   * - ``--completions``
     - No
     - Use ``POST /v1/completions`` only.
   * - ``--chat-first``
     - No
     - Try ``/v1/chat/completions`` first, then fall back to
       ``/v1/completions``.
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output PATH``
     - No
     - Save metrics to a file (format follows ``--format``).
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout output. Exit code only.


query kvcache
-------------

The ``query kvcache`` subcommand reports how much of a prompt is already stored
in the KV cache. It tokenizes the prompt locally with the model's tokenizer,
posts the token IDs to the controller's ``POST /lookup`` endpoint, and
summarizes the coverage. ``--prompt`` supports the same ``{name}`` placeholders
as ``query engine``: bind one to a file with ``--documents NAME=PATH``, pass a
bare ``--documents PATH`` to fill the next unnamed placeholder (or append to the
prompt if none remain), and ``{lmcache}`` resolves to the bundled sample
document with no ``--documents`` needed.

.. code-block:: bash

   lmcache query kvcache --url http://localhost:5555 \
     --prompt "{ctx} What is the example usage of lmcache?" \
     --documents ctx=lmcache/cli/documents/lmcache.txt \
     --model meta-llama/Llama-3.1-8B-Instruct

.. code-block:: text

   =============== Query KV Cache ================
   Model:            meta-llama/Llama-3.1-8B-Instruct
   Prompt tokens:                             8192
   Cached tokens:                        7680/8192
   Cached chunks:                            30/32
   Cache locations:                  [cpu@inst-0]
   Cache status:                     HIT (partial)
   ==============================================

Cache status is ``HIT`` when the whole prompt is cached, ``MISS`` when nothing
is, and ``HIT (partial)`` otherwise.

.. note::

   Coverage is **prefix-based**: it reports the longest cached prefix and stops
   at the first uncached chunk. ``Cache locations`` lists the instance holding
   that prefix (``location@instance_id``), not a per-tier chunk histogram.
   ``Cached chunks`` is derived from ``--chunk-size`` and is exact only when
   that value matches the server's configured chunk size. Token IDs must come
   from the same tokenizer the engine uses, or coverage will read low.

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Required
     - Description
   * - ``--url URL``
     - Yes
     - Controller HTTP endpoint (e.g. ``http://localhost:5555``).
   * - ``--prompt TEXT``
     - Yes
     - Prompt text with optional ``{name}`` placeholders.
   * - ``--model ID``
     - Yes
     - Tokenizer/model ID used to derive token IDs. For gated models, run
       ``huggingface-cli login`` first.
   * - ``--documents NAME=PATH``
     - No
     - Load file text for ``{NAME}`` in ``--prompt``. Accepts one or more
       ``NAME=PATH`` or bare ``PATH`` values; a bare ``PATH`` fills the next
       unnamed placeholder or is appended to the prompt.
   * - ``--chunk-size N``
     - No
     - Tokens per cache chunk for the chunk-count display (default: 256). Must
       match the server's configured chunk size to be exact.
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output PATH``
     - No
     - Save metrics to a file (format follows ``--format``).
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout output. Exit code only.
