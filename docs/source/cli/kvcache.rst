lmcache kvcache
===============

The ``lmcache kvcache`` command manages KV cache state on a running LMCache
server.

.. code-block:: bash

   lmcache kvcache [output-options] <sub-command> [options]

Place the common ``--format``, ``--output``, and ``--quiet`` options before
the sub-command.

.. code-block:: text

   $ lmcache kvcache -h
   usage: lmcache kvcache [-h] [--format FORMAT] [--output PATH] [-q] {clear,list} ...

   Manage KV cache state.

   subcommands:
     clear          Clear all cached KV data in L1 (CPU)
     list           List one page of L2 cache objects

   options:
     -h, --help       show this help message and exit
     --format FORMAT  Stdout output format (default: terminal). Available: terminal, json.
     --output PATH    Save metrics to a file at PATH (format chosen by --format).
     -q, --quiet      Suppress stdout output. Exit code only.

list
----

List one page of object keys and sizes from an L2 storage adapter. This calls
``GET /cache/objects?tier=l2`` on the MP HTTP server. It does not read KV tensor
payloads. The selected adapter must support object listing.

.. code-block:: bash

   lmcache kvcache list --url http://localhost:8080 --adapter fs \
       --model-name org/model --page-size 100

Omitting ``--adapter`` selects the server's primary (first-configured) L2
adapter. Omitting ``--model-name`` lists all models in that adapter. L1 object
listing is not supported by this endpoint.

Each invocation makes one request. If the report contains a non-null
``next_page_token``, pass it unchanged to ``--page-token`` for the next page,
using the same server, adapter, and model filter. Tokens are opaque and should
not be decoded or edited. An empty page is successful; use the continuation
token, not the number of entries, to determine whether more pages exist.

**JSON output and continuation:**

.. code-block:: bash

   lmcache kvcache --format json list --url http://localhost:8080 \
       --adapter fs --page-size 100 > page.json
   TOKEN=$(jq -r '.metrics.next_page_token // empty' page.json)
   if [ -n "$TOKEN" ]; then
       lmcache kvcache --format json list --url http://localhost:8080 \
           --adapter fs --page-size 100 --page-token "$TOKEN"
   fi

The JSON report preserves the server's ``adapter``, ``shared``, ``entries``
and ``next_page_token`` fields under ``metrics``. Each entry retains its full
wire key, including ``kv_rank``, ``object_group_id``, ``cache_salt`` and ``tags``:

.. code-block:: json

   {
     "title": "L2 Cache Objects",
     "metrics": {
       "adapter": "fs",
       "shared": false,
       "next_page_token": null,
       "entries": [
         {
           "key": {
             "chunk_hash_hex": "deadbeef",
             "model_name": "org/model",
             "kv_rank": 0,
             "object_group_id": 0,
             "cache_salt": "",
             "tags": {}
           },
           "size_bytes": 4096
         }
       ]
     }
   }

The default terminal report shows the adapter, sharing flag, continuation
token, and a key/byte-size table. An empty page renders as ``(none)``.
To save a JSON report without stdout output:

.. code-block:: bash

   lmcache kvcache --format json --quiet --output page.json \
       list --url http://localhost:8080

Listing Options
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``--url``
     - Required MP HTTP server URL. A bare host/port is normalized to HTTP.
   * - ``--adapter``
     - L2 adapter type name; defaults to the primary adapter.
   * - ``--model-name``
     - Exact model-name filter; defaults to all models.
   * - ``--page-size``
     - Maximum entries in this page, from 1 to 5000; default 500.
   * - ``--page-token``
     - Opaque token returned by the previous page.

Listing Exit Codes
~~~~~~~~~~~~~~~~~~

``0`` means success, including an empty page; ``1`` means an HTTP, connection,
or JSON decoding error; ``2`` means an argument parsing error. HTTP errors
(such as an invalid cursor, unknown adapter, or unsupported listing) are
reported on stderr even with ``--quiet``. Failed requests do not write a
success report.

clear
-----

Clear all cached KV data in **L1 (CPU memory)** on the target LMCache server.

.. code-block:: bash

   lmcache kvcache clear --url <MP_HTTP_URL>

**Example:**

.. code-block:: bash

   $ lmcache kvcache clear --url http://localhost:8000

   ================ KV Cache Clear ================
   Status:                                       OK
   ================================================

**JSON output** (for scripting with ``jq``):

.. code-block:: bash

   $ lmcache kvcache --format json clear --url http://localhost:8000
   {
     "title": "KV Cache Clear",
     "metrics": {
       "status": "OK"
     }
   }

**Quiet mode** (exit code only, no output):

.. code-block:: bash

   $ lmcache kvcache -q clear --url http://localhost:8000
   $ echo $?
   0

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Flag
     - Required
     - Description
   * - ``--url``
     - Yes
     - URL of the LMCache MP HTTP server (e.g. ``http://localhost:8000``).
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output``
     - No
     - Save output to a file (uses the format chosen by ``--format``).
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout. Useful in scripts where you only need the exit code.

Exit Codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Success.
   * - ``1``
     - Error (connection failure, server error, bad arguments).

Common Patterns
---------------

**Handle temporary server unavailability:**

If the server is temporarily unreachable (e.g. due to network issue), the command
fails with exit code 1. For persistent connectivity issues, use ``lmcache ping``
to diagnose.

.. code-block:: bash

   if lmcache kvcache -q clear --url http://localhost:8000; then
       echo "Cache cleared"
   else
       echo "Clear failed — server temporarily unreachable, retrying later"
   fi

**Clear cache and capture JSON result:**

.. code-block:: bash

   RESULT=$(lmcache kvcache --format json clear --url http://localhost:8000)
   STATUS=$(echo "$RESULT" | jq -r '.metrics.status')
   echo "Clear status: $STATUS"
