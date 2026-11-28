lmcache kvcache
===============

The ``lmcache kvcache`` command manages KV cache state on a running LMCache
server.

.. code-block:: bash

   lmcache kvcache <sub-command> [options]

Like ``lmcache quota``, every sub-command accepts the shared output options
(``--format``, ``--output``, ``-q`` / ``--quiet``) directly.

.. code-block:: text

   $ lmcache kvcache -h
   usage: lmcache kvcache [-h] {clear,list} ...

   Manage KV cache state.

   subcommands:
     clear          Clear all cached KV data in L1 (CPU)
     list           List one page of L2 cache objects

list
----

List one page of object keys and sizes from an L2 storage adapter. This calls
``GET /cache/objects?tier=l2`` on the MP HTTP server. It does not read KV tensor
payloads. The selected adapter must support object listing (currently
only the S3 adapter).

.. code-block:: bash

   lmcache kvcache list --url http://localhost:8080 --adapter s3 \
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

   lmcache kvcache list --format json --url http://localhost:8080 \
       --adapter s3 --page-size 100 > page.json
   TOKEN=$(jq -r '.metrics.next_page_token // empty' page.json)
   if [ -n "$TOKEN" ]; then
       lmcache kvcache list --format json --url http://localhost:8080 \
           --adapter s3 --page-size 100 --page-token "$TOKEN"
   fi

The JSON report preserves the server's ``adapter``, ``shared``, ``entries``
and ``next_page_token`` fields under ``metrics``. Each entry contains the wire
key fields alongside ``size_bytes``, including the full ``chunk_hash_hex`` and
the original ``tags`` object.

.. code-block:: json

   {
     "title": "L2 Cache Objects",
     "metrics": {
       "adapter": "s3",
       "shared": false,
       "next_page_token": null,
       "entries": [
         {
           "chunk_hash_hex": "0123456789abcdef",
           "model_name": "org/model",
           "kv_rank": 0,
           "object_group_id": 0,
           "cache_salt": "",
           "tags": {
             "version": "v2",
             "env": "prod"
           },
           "size_bytes": 4096
         }
       ]
     }
   }

The default terminal report shows the first 12 hex characters followed by
``…`` for longer hashes. Tags appear as comma-separated ``key=value`` pairs
sorted by key, or ``-`` for empty tags. A file saved with
``--format terminal --output page.txt`` uses the same shortened hashes.

.. code-block:: text

   =============== L2 Cache Objects ===============
   Adapter:                                      s3
   Shared:                                    False
   Next page token:                             N/A
   Chunk hash     Model      Rank  Group  Salt  Tags                 Bytes
   -----------------------------------------------------------------------
   0123456789ab…  org/model     0      0        env=prod,version=v2   4096
   ================================================

An empty page displays ``(none)``. Use ``--quiet`` with ``--output`` to save
the JSON report without printing it.

.. code-block:: bash

   lmcache kvcache list --format json --quiet --output page.json \
       --url http://localhost:8080

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
   * - ``--format``
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output``
     - Save output to a file (uses the format chosen by ``--format``).
   * - ``-q`` / ``--quiet``
     - Suppress stdout. Useful in scripts where you only need the exit code.

Listing Exit Codes
~~~~~~~~~~~~~~~~~~

A successful page, including an empty one, exits 0. HTTP, connection, and
JSON decoding errors exit 1 and are reported on stderr even with ``--quiet``.
HTTP errors include the server's ``detail`` when available. If the adapter
does not support listing, the message is ``HTTP 503 from <url>: <detail>``.
Failed requests produce no report. Invalid arguments exit 2.

clear
-----

Clear all cached KV data in **L1 (CPU memory)** on the target LMCache server.

Existing scripts can also pass ``-q`` / ``--quiet``, ``--format``, or
``--output`` before ``clear``, for example
``lmcache kvcache -q clear --url http://localhost:8080``. If an output
option is given in both positions, the value after ``clear`` takes precedence.
A 503 from this command uses the same ``Server unhealthy`` diagnostic as
``describe kvcache``.

.. code-block:: bash

   lmcache kvcache clear --url <MP_HTTP_URL>

**Example:**

.. code-block:: bash

   $ lmcache kvcache clear --url http://localhost:8080

   ================ KV Cache Clear ================
   Status:                                       OK
   ================================================

**JSON output** (for scripting with ``jq``):

.. code-block:: bash

   $ lmcache kvcache clear --format json --url http://localhost:8080
   {
     "title": "KV Cache Clear",
     "metrics": {
       "status": "OK"
     }
   }

**Quiet mode** (exit code only, no output):

.. code-block:: bash

   $ lmcache kvcache clear -q --url http://localhost:8080
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
     - URL of the LMCache MP HTTP server (e.g. ``http://localhost:8080``).
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
     - Error (connection failure, server error, bad arguments). The server's
       reason is reported on stderr when the error response carries one.

Common Patterns
---------------

**Handle temporary server unavailability:**

If the server is temporarily unreachable (e.g. due to network issue), the command
fails with exit code 1. For persistent connectivity issues, use ``lmcache ping``
to diagnose.

.. code-block:: bash

   if lmcache kvcache clear -q --url http://localhost:8080; then
       echo "Cache cleared"
   else
       echo "Clear failed — server temporarily unreachable, retrying later"
   fi

**Clear cache and capture JSON result:**

.. code-block:: bash

   RESULT=$(lmcache kvcache clear --format json --url http://localhost:8080)
   STATUS=$(echo "$RESULT" | jq -r '.metrics.status')
   echo "Clear status: $STATUS"
