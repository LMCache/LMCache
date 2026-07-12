lmcache tool
============

The ``lmcache tool`` command groups offline analysis utilities bundled with
LMCache.

.. code-block:: bash

   lmcache tool <tool-name> <action> [options]

Three tools are available: ``cache-simulator`` replays lookup logs,
``transfer-channel-benchmark`` measures peer-to-peer read throughput, and
``flamegraph`` profiles a running LMCache process.

.. note::

   ``cache-simulator`` depends on the optional ``plot`` extras
   (``sortedcontainers`` / ``matplotlib``). If they are not installed, the
   sub-command is silently omitted from the CLI. Install the extras to enable
   it.


cache-simulator
---------------

Replay LMCache lookup-hash JSONL logs through an LRU cache to measure the
KV-cache token hit rate. It has three actions:

.. code-block:: bash

   lmcache tool cache-simulator {simulate,sweep,gen-dataset} [options]

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Action
     - Description
   * - ``simulate``
     - Replay logs at a fixed cache capacity; print a text report and save a
       7-panel statistics PNG.
   * - ``sweep``
     - Sweep across a range of cache capacities and save a hit-rate vs.
       capacity PNG.
   * - ``gen-dataset``
     - Generate a ``vllm bench serve`` custom dataset (JSONL) from
       lookup-hash JSONL logs, preserving prefix-sharing structure.

Each action has its own flags. Run the built-in help for the full list:

.. code-block:: bash

   lmcache tool cache-simulator simulate --help
   lmcache tool cache-simulator sweep --help
   lmcache tool cache-simulator gen-dataset --help


transfer-channel-benchmark
--------------------------

Measure read throughput (GB/s) of the LMCache transfer channel
(``lmcache/v1/distributed/transfer_channel/``) for batched peer-to-peer
reads. It allocates the transferred objects through the same
``L1MemoryManager`` production uses, so it exercises the real memory path
rather than raw tensors.

The benchmark runs as **two processes** -- a ``server`` that registers a
source buffer and serves its object catalog, and a ``client`` that reads a
subset of those objects and reports throughput:

.. code-block:: bash

   # Terminal 1: the source
   lmcache tool transfer-channel-benchmark --role server \
       --url 127.0.0.1:7600 --buffer-size 8GB --object-size 10MB

   # Terminal 2: the reader
   lmcache tool transfer-channel-benchmark --role client \
       --url 127.0.0.1:7600 --object-size 10MB --num-objects 100 --iters 5

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Description
   * - ``--role {server,client}``
     - Required. ``server`` registers the source buffer; ``client`` reads
       from it and reports throughput.
   * - ``--transfer-channel-type``
     - Transfer channel implementation to benchmark (default: ``nixl``).
   * - ``--url``
     - Server: bind address. Client: the server's advertise URL to read
       from (default: ``127.0.0.1:7600``).
   * - ``--buffer-size``
     - Server: total registered L1 source buffer, e.g. ``8GB``.
   * - ``--object-size`` / ``--page-size``
     - Size of each transferred object and the alignment; must match on
       both sides.
   * - ``--num-objects`` / ``--iters``
     - Objects read per iteration, and number of measured iterations.
   * - ``--verify``
     - Check the bytes read against the source after transfer.

The ``--url``, ``--object-size``, and ``--page-size`` values must match
between the two processes. Run either side with ``--help`` for the full
list, including the client's own ``--listen-url`` and the catalog
``--control-url``.


flamegraph
----------

Attach a profiler to an already-running process -- an MP cache server, a
vLLM worker with LMCache embedded, or any Python process -- record for a
fixed duration, and render a flame graph. Unlike
``lmcache bench l2 --flamegraph on``, which profiles its own benchmark,
this attaches to a process you did not start.

.. code-block:: bash

   # Which threads of an MP cache server hold the GIL, sampled for 20s
   lmcache tool flamegraph --pid $(pgrep -f 'lmcache server') --mode gil --duration 20

The ``--mode`` values are the same as ``lmcache bench l2`` and are
described in full under
:ref:`lmcache bench l2 <lmcache-bench-l2-profiling>`. In short: to
profile **Python or GIL contention** use ``gil`` / ``wall`` (``py-spy``,
one root per thread, attaches to an **unmodified** target, but only a
CPython one) -- hence the ``gil`` default for a live server. To look at
**CPU/IO time, kernel frames, or a non-Python process** use the perf/bcc
modes ``on-cpu`` (CPU cycles), ``off-cpu`` (blocked time), ``offwake``
(blocked time plus the waker's stack), or ``wakeup`` (the stacks doing
the waking); these name Python functions only when the target was
launched with ``PYTHONPERFSUPPORT=1``, which carries a standing per-call
overhead -- so reserve it for a dedicated profiling session, not
production.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Flag
     - Default
     - Description
   * - ``--pid``
     - *(required)*
     - Process to profile.
   * - ``--mode``
     - ``gil``
     - ``on-cpu``, ``off-cpu``, ``offwake``, ``wakeup``, ``wall``, or
       ``gil``.
   * - ``--duration``
     - ``30``
     - Seconds to record. ``0`` records until interrupted with Ctrl-C.
   * - ``--output``
     - *(auto)*
     - SVG path. Default
       ``/tmp/lmcache_bench_flames/pid<PID>.<mode>.svg``.
   * - ``--flamegraph-scripts-dir``
     - *(auto)*
     - FlameGraph scripts directory; unused by ``wall`` / ``gil``.

Attaching needs permission to trace the target: ``wall`` / ``gil`` require
``kernel.yama.ptrace_scope`` at ``0`` (or root); ``on-cpu`` requires
``kernel.perf_event_paranoid`` at ``2`` or lower; ``off-cpu`` / ``offwake``
/ ``wakeup`` load a BPF program and need ``sudo``. A blocked attach exits
non-zero with the sysctl to run.

Recording a live process is not free: ``on-cpu`` writes a ``perf.data``
that grows with ``--duration`` and thread count, the bcc modes add BPF
probes whose cost rises with how busy the target is, and even ``py-spy``
reads target memory each sample. On a production server keep ``--duration``
short and prefer ``wall`` / ``gil``. See
:ref:`lmcache bench l2 <lmcache-bench-l2-profiling>` for the per-mode
breakdown.
