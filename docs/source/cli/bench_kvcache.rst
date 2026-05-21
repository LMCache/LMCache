lmcache bench kvcache
=====================

The ``lmcache bench kvcache`` command is an end-to-end sanity test for the
LMCache Multi-Process (MP) cache server. It connects to a running server
over ZMQ and exercises the full KV-cache data path for a sequence of
synthetic requests, then optionally verifies per-chunk checksums through
the HTTP API.

.. code-block:: bash

   lmcache bench kvcache [options]

Unlike :ref:`lmcache bench engine <lmcache-bench-engine>`, this command does
**not** require an inference engine. It only needs a running LMCache MP
server (ZMQ + HTTP) and a GPU.


What it does
------------

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
-----------

Start the MP server in one terminal:

.. code-block:: bash

   python3 -m lmcache.v1.multiprocess.http_server \
       --host localhost --port 15556 \
       --chunk-size 256 --l1-size-gb 5 \
       --eviction-policy LRU --max-workers 1

Then in another terminal:

.. code-block:: bash

   lmcache bench kvcache \
       --rpc-url tcp://localhost:15556 \
       --url http://localhost:8080

By default the tool runs forever (``--end`` unset); stop it with
``Ctrl-C`` at any time. Pass ``--end N`` for a bounded run.


Options
-------

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
   * - ``--mode {gpu}``
     - ``gpu``
     - Run mode. Only ``gpu`` is supported today; CPU mode is a
       planned follow-up.
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


KV cache shape spec
-------------------

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

   lmcache bench kvcache \
       --rpc-url tcp://localhost:15556 \
       --kvcache-shape-spec "(1,1024,16,1,128):float16:4;(2,1024,16,8,128):float16:28"

All groups must share the same ``NB`` and ``BS`` (this is a physical
constraint of paged KV). Layer counts across groups sum to the total
layer count registered with the server.

See ``parse_kvcache_shape_spec`` in ``lmcache/v1/kv_layer_groups.py``
for the authoritative parsing rules and validation errors.


Example output
--------------

.. code-block:: text

   Connecting to LMCache MP Server at tcp://localhost:15556 (mode=gpu) ...
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
----------

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


See also
--------

* :doc:`bench` -- ``lmcache bench engine`` for engine-side workload
  benchmarks.
* :doc:`bench_l2` -- ``lmcache bench l2`` for
  store / lookup / load throughput benchmarks against an L2 cache
  adapter.
* :doc:`kvcache` -- ``lmcache kvcache`` for managing KV cache state on a
  running server (clear, etc.).
