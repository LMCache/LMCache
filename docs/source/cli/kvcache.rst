lmcache kvcache
===============

The ``lmcache kvcache`` command manages KV cache state on a running LMCache
server.

.. code-block:: bash

   lmcache kvcache <sub-command> [options]

.. code-block:: text

   $ lmcache kvcache -h
   usage: lmcache kvcache [-h] [--format FORMAT] [--output PATH] [-q] {clear,delete} ...

   Manage KV cache state.

   subcommands:
     clear          Clear all cached KV data in L1 (CPU)
  delete         Delete specific cached objects by key (L1, L2, or both)

   options:
     -h, --help       show this help message and exit
     --format FORMAT  Stdout output format (default: terminal). Available: terminal, json.
     --output PATH    Save metrics to a file at PATH (format chosen by --format).
     -q, --quiet      Suppress stdout output. Exit code only.

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

   $ lmcache kvcache clear --url http://localhost:8000 --format json
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

delete
------

Delete specific cached objects by key, from L1, an L2 adapter, or both. Unlike
``clear``, which drops everything, the caller names the objects.

Keys are object keys: ``chunk_hash_hex`` and ``model_name`` are required,
``kv_rank`` defaults to 0, and ``object_group_id``, ``cache_salt`` and ``tags``
are optional. An L2 adapter lookup needs the same fields the writer used, so a
key copied from a store trace round-trips unchanged.

.. code-block:: bash

   lmcache kvcache delete --url <MP_HTTP_URL> --key '<KEY_JSON>'

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Option
     - Default
     - Description
   * - ``--url``
     - (required)
     - Target MP HTTP endpoint.
   * - ``--key JSON``
     - (required)
     - Object key as JSON. Repeat for several keys.
   * - ``--keys-file PATH``
     - (required)
     - JSON file with an array of keys, or an object holding a ``keys`` array.
       Mutually exclusive with ``--key``.
   * - ``--tier {l1,l2,all}``
     - ``l1``
     - Tier to delete from.
   * - ``--adapter NAME``
     - primary adapter
     - L2 adapter name. Ignored for ``tier=l1``.
   * - ``--force``
     - off
     - Delete L1 keys even while they are read- or write-locked.

The command does not prompt: the keys already have to be spelled out, and the
sibling commands (``clear``, ``quota delete``) do not ask either. Run it from a
script without a stdin dance.

**Example:**

.. code-block:: bash

   KEY='{"chunk_hash_hex":"00...","model_name":"org/model","kv_rank":0}'
   lmcache kvcache delete --url http://localhost:8000 --key "$KEY"

**Delete from every tier and require success:**

.. code-block:: bash

   lmcache kvcache -q delete --url http://localhost:8000 \
       --keys-file /tmp/keys.json --tier all --force

A response with ``ok: false`` (an adapter rejected part of the batch) exits
``1``, so ``delete ... && next-step`` cannot mistake it for success.

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

   RESULT=$(lmcache kvcache clear --url http://localhost:8000 --format json)
   STATUS=$(echo "$RESULT" | jq -r '.metrics.status')
   echo "Clear status: $STATUS"
