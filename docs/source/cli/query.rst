lmcache query
=============

The ``lmcache query`` command runs a single, read-only query and reports the
result as a metrics report. It has three targets:

.. code-block:: bash

   lmcache query {engine,coordinator,kvcache} [options]

* ``engine`` — send one OpenAI-compatible inference request to a serving
  engine's HTTP API and report token and latency metrics.
* ``coordinator`` — read one of the MP coordinator's read-only HTTP APIs.
* ``kvcache`` — query KV-cache endpoints (not implemented yet).


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


query coordinator
-----------------

The ``query coordinator`` subcommand reads one of the MP coordinator's
read-only HTTP APIs and renders the reply as a metrics report — the same
information ``curl`` returns, but aligned into columns and with byte counts
and ratios already formatted. Pick the API with ``--api``; everything else
is optional.

.. code-block:: bash

   lmcache query coordinator --api NAME [--url URL] [options]

Only reads are exposed. The coordinator's mutating routes are either
server-to-coordinator plumbing (``POST /events``, ``POST /instances``,
heartbeats) or belong to a command that owns the action — quotas are written
with :doc:`lmcache quota <quota>`. See :doc:`/mp/coordinator` for the HTTP
surface itself and the meaning of each field.

The default ``--url`` is ``http://127.0.0.1:9300``, matching the
coordinator's default port. A URL without a scheme is assumed to be
``http://``, and a trailing slash is ignored, so ``coordinator:9300`` and
``http://coordinator:9300/`` are both accepted.

APIs
~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 30 24 30

   * - ``--api``
     - Reads
     - Extra flags
     - Reports
   * - ``usage``
     - ``/instances/usage``, or
       ``/instances/{id}/usage`` with ``--instance``
     - ``--instance`` (optional)
     - Per-compartment occupancy against declared capacity, busiest first.
   * - ``instances``
     - ``/instances``
     - —
     - Registered MP servers with their addresses and P2P URLs.
   * - ``health``
     - ``/healthz``
     - —
     - Coordinator liveness.
   * - ``directory``
     - ``/directory/stats``
     - —
     - Key-directory size and blend-index counts.
   * - ``keys``
     - ``/directory/keys``
     - ``--limit`` (default: 20)
     - A page of directory keys and where each one is placed.
   * - ``quota``
     - ``/quota``, or ``/quota/{salt}``
       with ``--cache-salt``
     - ``--cache-salt`` (optional)
     - Per-salt L2 usage against quota.
   * - ``quota-config``
     - ``/quota/config``
     - —
     - The default limit applied to salts with no explicit quota.
   * - ``prefetch``
     - ``/cache/prefetches/{instance}/{request_id}``
     - ``--instance`` **and** ``--request-id`` (required)
     - One warm-prefetch request's progress.
   * - ``metrics``
     - ``/metrics``
     - —
     - Prometheus text, passed through verbatim.

Fleet memory
~~~~~~~~~~~~

``--api usage`` is the everyday view: one row per memory compartment
(``tier/backend``) per server, sorted so the fullest compartment is first.

.. code-block:: bash

   $ lmcache query coordinator --api usage

   ============== Coordinator: usage ==============
   instance        compartment      used  capacity    ratio
   --------------------------------------------------------
   mp-gpu7         l1/dram      48.00 GB  64.00 GB    75.0%
   mp-gpu8         l1/dram       2.00 GB  64.00 GB     3.1%
   mp-gpu7         l2/fs        12.00 GB        --  unknown
   (fleet-shared)  l2/s3         7.00 GB        --  unknown
   ================================================

Reading the table:

* A capacity of ``--`` and a ratio of ``unknown`` mean the server never
  declared a capacity for that compartment — an unmeasured tier, not an empty
  one. A ``0`` would be misleading, so it is never printed there.
* A compartment shared by the whole fleet (e.g. one S3 bucket behind every
  server) is attributed to ``(fleet-shared)`` rather than to any one server.
* Numeric columns are right-aligned so sizes line up on the decimal point.

Add ``--instance`` to narrow the report to one server. The instance id must be
known to the coordinator — registered, holding bytes, or having declared
capacity — otherwise the coordinator answers ``404`` and the command exits
``1``:

.. code-block:: bash

   $ lmcache query coordinator --api usage --instance mp-gpu7

   ============== Coordinator: usage ==============
   instance  compartment      used  capacity    ratio
   --------------------------------------------------
   mp-gpu7   l1/dram      48.00 GB  64.00 GB    75.0%
   mp-gpu7   l2/fs        12.00 GB        --  unknown
   ================================================

Fleet membership
~~~~~~~~~~~~~~~~

.. code-block:: bash

   $ lmcache query coordinator --api instances

   ============ Coordinator: instances ============
   instance  address        mq port  p2p url
   -----------------------------------------------------
   mp-gpu7   10.0.0.7:8101     8201  tcp://10.0.0.7:8301
   mp-gpu8   10.0.0.8:8101       --  --
   ================================================

``--`` marks a value the server did not advertise (no MQ port, no P2P URL),
kept distinct from ``0`` and from an empty string.

Quota and usage
~~~~~~~~~~~~~~~

With no ``--cache-salt``, the fleet-wide listing:

.. code-block:: bash

   $ lmcache query coordinator --api quota

   ============== Coordinator: quota ==============
   Total usage (GiB):                         19.00
   cache salt  usage GiB  quota GiB  quota set
   -------------------------------------------
   tenant-a        12.50      20.00  yes
   (default)        6.50       0.00  no
   ================================================

``(default)`` is the un-salted (empty-string) tenant. ``quota set`` is ``no``
when no explicit quota exists for that salt — such a salt is governed by
``--api quota-config`` instead, so a ``0.00`` quota column there does not mean
"zero bytes allowed".

With ``--cache-salt``, one tenant:

.. code-block:: bash

   $ lmcache query coordinator --api quota --cache-salt tenant-a

   ============== Coordinator: quota ==============
   Cache salt:                             tenant-a
   Quota (GiB):                               20.00
   Quota set:                                  True
   Usage (GiB):                               12.50
   ================================================

.. note::

   To address the un-salted tenant, pass the sentinel the HTTP API uses:
   ``--cache-salt _default``. An empty ``--cache-salt ''`` builds the path
   ``/quota/``, which does not address the empty salt.

The default limit for salts with no explicit quota:

.. code-block:: bash

   $ lmcache query coordinator --api quota-config

   ========== Coordinator: quota-config ===========
   Default limit (GiB):               none (exempt)
   ================================================

``none (exempt)`` means the default is unset, so unquota'd salts are exempt
from eviction. That is distinct from a default of ``0.0``, which makes every
byte under an unquota'd salt evictable. See :doc:`/mp/coordinator` for why a
freshly restarted coordinator starts out exempt.

Key directory
~~~~~~~~~~~~~

Directory size, and how much of it is fragment-matchable by CacheBlend:

.. code-block:: bash

   $ lmcache query coordinator --api directory

   ============ Coordinator: directory ============
   Keys:                                      18422
   Placements:                                20117
   ----------------- Blend index ------------------
   Contents:                                     91
   Chunks:                                     1740
   Table size:                                 1740
   ================================================

A page of keys and their placements. ``--limit`` sets the page size (default
20; the endpoint accepts 1–10000, and a value outside that range is rejected
by the coordinator and exits ``1``):

.. code-block:: bash

   $ lmcache query coordinator --api keys --limit 2

   ============== Coordinator: keys ===============
   Matching keys:                             18422
   chunk         model                    rank  salt       placements
   ---------------------------------------------------------------------------------------
   abababababab  meta-llama/Llama-3.1-8B     0  tenant-a   mp-gpu7:l1/dram, (shared):l2/s3
   cdcdcdcdcdcd  meta-llama/Llama-3.1-8B     0  (default)  mp-gpu8:l1/dram
   ================================================

``Matching keys`` is the total in the directory, not the number of rows shown.
The chunk hash is truncated to 12 hex characters — enough to correlate with a
log line, without pushing the placements column off the terminal. A placement
with no owning instance (a fleet-shared backend) reads as ``(shared)``. The
directory is a live structure, so successive pages may skip or repeat keys.

Prefetch progress
~~~~~~~~~~~~~~~~~

``--api prefetch`` polls a warm prefetch submitted earlier via
``POST /cache/prefetches``. Both ``--instance`` and ``--request-id`` are
required; omitting either exits ``2`` without issuing a request.

.. code-block:: bash

   $ lmcache query coordinator --api prefetch --instance mp-gpu7 --request-id abc123

   ============ Coordinator: prefetch =============
   Status:                                completed
   Found keys:                                   12
   Total keys:                                   12
   ================================================

While the load is still running, only ``Status: pending`` is reported. The
labels come from the reply itself, so a server that adds fields will show them
without a CLI change.

.. warning::

   The first poll that observes completion drops the job on the MP server.
   Polling the same ``--request-id`` again returns ``404`` and exits ``1``.

Prometheus metrics
~~~~~~~~~~~~~~~~~~

``--api metrics`` is the one API that is not a metrics report: the
coordinator's ``/metrics`` body is written to stdout unchanged, so it can be
piped to other tools. ``--format``, ``--output``, and ``--quiet`` do not apply
to it.

.. code-block:: bash

   lmcache query coordinator --api metrics | promtool check metrics
   lmcache query coordinator --api metrics | grep lmcache_coordinator

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Required
     - Description
   * - ``--api NAME``
     - Yes
     - Which API to read. One of ``usage``, ``instances``, ``health``,
       ``directory``, ``keys``, ``quota``, ``quota-config``, ``prefetch``,
       ``metrics``.
   * - ``--url URL``
     - No
     - Coordinator base URL (default: ``http://127.0.0.1:9300``). The scheme
       may be omitted.
   * - ``--instance ID``
     - No
     - Instance id. Narrows ``--api usage`` to one server; required by
       ``--api prefetch``.
   * - ``--cache-salt SALT``
     - No
     - Narrows ``--api quota`` to one tenant. Use ``_default`` for un-salted
       traffic.
   * - ``--request-id ID``
     - No
     - Prefetch request id. Required by ``--api prefetch``.
   * - ``--limit N``
     - No
     - Rows to request for ``--api keys`` (default: 20).
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``. Ignored by
       ``--api metrics``.
   * - ``--output PATH``
     - No
     - Save the report to a file (format follows ``--format``). Ignored by
       ``--api metrics``.
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout output. Exit code only. Ignored by ``--api metrics``.

JSON output
~~~~~~~~~~~

``--format json`` renders the same report as a JSON object. Table APIs become
a list of row objects under the table's key:

.. code-block:: bash

   $ lmcache query coordinator --api usage --format json
   {
     "title": "Coordinator: usage",
     "metrics": {
       "usage": [
         {
           "instance": "mp-gpu7",
           "compartment": "l1/dram",
           "used": "48.00 GB",
           "capacity": "64.00 GB",
           "ratio": "75.0%"
         },
         {
           "instance": "mp-gpu8",
           "compartment": "l1/dram",
           "used": "2.00 GB",
           "capacity": "64.00 GB",
           "ratio": "3.1%"
         },
         {
           "instance": "mp-gpu7",
           "compartment": "l2/fs",
           "used": "12.00 GB",
           "capacity": "--",
           "ratio": "unknown"
         },
         {
           "instance": "(fleet-shared)",
           "compartment": "l2/s3",
           "used": "7.00 GB",
           "capacity": "--",
           "ratio": "unknown"
         }
       ]
     }
   }

.. warning::

   Table rows carry **display strings**, not raw numbers: ``"48.00 GB"``,
   ``"75.0%"``, ``"12.50"``, ``"yes"``, and the placeholders ``"--"`` /
   ``"unknown"``. They are meant for reading and for grepping, not for
   arithmetic. For machine-readable values, ``curl`` the endpoint directly:

   .. code-block:: bash

      curl -s http://127.0.0.1:9300/instances/usage | jq \
        '.instances[].modules[] | {tier, backend, used_bytes, usage_ratio}'

Exit codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Success.
   * - ``1``
     - The coordinator was unreachable, returned a non-2xx status (e.g. an
       unknown instance id or prefetch request), or replied with something
       that was not JSON.
   * - ``2``
     - Usage error: a required flag for the chosen ``--api`` is missing (e.g.
       ``--api prefetch`` without ``--request-id``). No request is sent.

This makes the command usable as a readiness check:

.. code-block:: bash

   until lmcache query coordinator --api health -q; do sleep 1; done

Common patterns
~~~~~~~~~~~~~~~

**Find the fullest server before pinning work to it:**

.. code-block:: bash

   lmcache query coordinator --api usage | head -5

**Watch a fleet fill up:**

.. code-block:: bash

   watch -n 5 'lmcache query coordinator --api usage'

**Check one tenant against its budget:**

.. code-block:: bash

   lmcache query coordinator --api quota --cache-salt tenant-a
   lmcache query coordinator --api quota-config   # what un-quota'd salts get

**Point at a coordinator in Kubernetes:**

.. code-block:: bash

   lmcache query coordinator --api instances \
     --url http://coordinator.default.svc:9300

Limitations
~~~~~~~~~~~

* Reports are a formatted subset of each reply, not the whole thing. ``--api
  usage`` does not surface the ``registered`` flag, so a server that
  deregistered while its L2 bytes survive appears in the table like any other;
  cross-check with ``--api instances``. ``--api directory`` omits the
  per-instance L1 key counts, and ``--api keys`` omits each key's token count.
* ``--api keys`` requests a page size only. The endpoint's ``tier``,
  ``instance_id``, ``backend``, and ``offset`` filters are not exposed —
  ``curl`` ``/directory/keys`` for those.
* ``--api quota`` reports the L2 tier, which is the tier quotas are enforced
  on. The endpoint's ``tier`` parameter is not exposed.
