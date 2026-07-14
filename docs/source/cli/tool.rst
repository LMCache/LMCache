lmcache tool
============

The ``lmcache tool`` command groups offline analysis utilities bundled with
LMCache.

.. code-block:: bash

   lmcache tool <tool-name> <action> [options]

Three tools are available: ``cache-simulator`` replays lookup logs,
``transfer-channel-benchmark`` measures peer-to-peer read throughput, and
``flamegraph`` profiles any running process.

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

The benchmark runs as **two processes**, a ``server`` that registers a
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

Attach a profiler to an already-running process (an MP cache server, a vLLM
worker with LMCache embedded, or any Python process), record for a fixed
duration, and render a flame graph.

It is a thin wrapper over the standard Linux profilers (``py-spy``, ``perf``,
and ``bcc``), rendered as a flame graph with Brendan Gregg's FlameGraph. A
single ``--mode`` selects the tool; the command applies sensible defaults,
resolves Python function names that a raw native profiler leaves as
``[unknown]``, fetches the FlameGraph renderer, and reports a missing tool or
blocked attach with an actionable message. It does not install the profilers
themselves; it names what to run.

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
     - ``on-cpu``, ``off-cpu``, ``offwake``, ``wakeup``, ``wall``, or ``gil``.
   * - ``--duration``
     - ``30``
     - Seconds to record. ``0`` records until interrupted with Ctrl-C.
   * - ``--output``
     - *(auto)*
     - SVG path. Default ``/tmp/lmcache_bench_flames/pid<PID>.<mode>.svg``.
   * - ``--flamegraph-scripts-dir``
     - *(auto)*
     - FlameGraph scripts directory; unused by ``wall`` / ``gil``.


Usage
~~~~~

.. code-block:: bash

   # Which threads of an MP cache server hold the GIL, sampled for 20s
   lmcache tool flamegraph --pid $(pgrep -f 'lmcache server') --mode gil --duration 20

.. _lmcache-flamegraph-entry-points:

LMCache renders flame graphs from three commands; they differ in **where the
load comes from** and **what they attach to**:

.. list-table::
   :header-rows: 1
   :widths: 32 32 36

   * - Command
     - Load driven by
     - Profiles
   * - ``lmcache bench l2 --flamegraph on``
     - The benchmark itself (synthetic, in-process)
     - Its own process: one L2 adapter under a microbenchmark.
   * - ``lmcache bench server --flamegraph on``
     - The benchmark client (synthetic KV over ZMQ)
     - An external server via ``--profile-server-pid``, under that load.
   * - ``lmcache tool flamegraph`` *(this command)*
     - Nothing (it drives no load)
     - Any running process, under whatever **real** work it is already doing.

Reach for this command to profile a **real, unmodified process** (a
production or vLLM-driven server under actual traffic, an idle server, or any
arbitrary PID) without standing up a benchmark harness.


Modes
~~~~~

.. _lmcache-flamegraph-modes:

The six ``--mode`` values fall into two families; pick by what you want to
see.

**Python execution or GIL contention** with ``gil`` / ``wall`` (``py-spy``):
one root frame per thread, attach to an **unmodified** CPython target, but
see no kernel frames.

* **gil**: only threads holding the GIL, so interpreter-lock contention is
  directly visible (the default for a live server).
* **wall**: wall-clock time per thread, blocked threads included.

**CPU/IO time, kernel frames, context switches, or a non-Python process**
with the whole-process modes (``perf`` / bcc): every thread, kernel frames,
scheduler activity, any process, but merged into one chart.

* **on-cpu** (``perf``): where CPU *cycles* go.
* **off-cpu** (``offcputime-bpfcc``): time spent *blocked* (I/O, locks).
* **offwake** (``offwaketime-bpfcc``): like ``off-cpu``, but each blocked
  stack also carries the stack of the thread that *woke* it (upper half,
  drawn inverted); use it when ``off-cpu`` shows a big blocked tower and you
  need the cause.
* **wakeup** (``wakeuptime-bpfcc``): the mirror image, the stacks doing the
  waking.

**How they work.** All run as external processes; what differs is *where the
data is collected*, in the kernel (perf, bcc) or by reading the target's
memory (py-spy):

* **perf** samples the CPU at 99 Hz, recording the running thread's stack, so
  it sees only on-CPU work and walks native stacks via frame pointers.
* **bcc** is *not* sampled: an eBPF program fires on scheduler events, so it
  can measure *blocked* time perf cannot, at a cost that grows with the
  target's context-switch rate. Loading eBPF needs privilege.
* **py-spy** works in user space, reading the target's memory
  (``process_vm_readv``) with ``--nonblocking`` (no pause); it sees only
  Python frames and needs ``ptrace`` permission.

**Equivalent raw commands** for ``--pid P``:

.. code-block:: text

   gil      py-spy record --gil --rate 200 --threads --idle --nonblocking --pid P
   wall     py-spy record       --rate 200 --threads --idle --nonblocking --pid P
   on-cpu   perf record -F 99 -g -p P  ->  perf script | stackcollapse-perf.pl | flamegraph.pl
   off-cpu  sudo offcputime-bpfcc  -df -p P  |  flamegraph.pl --colors io
   offwake  sudo offwaketime-bpfcc -df -p P  |  flamegraph.pl --colors chain
   wakeup   sudo wakeuptime-bpfcc   -f -p P  |  flamegraph.pl --colors wakeup

**Native stacks need frame pointers.** The perf/bcc modes unwind native
stacks through frame pointers (perf via ``-g``, bcc via ``bpf_get_stackid``),
so a target built with ``-fomit-frame-pointer`` shows broken C stacks in
either; rebuild with ``-fno-omit-frame-pointer``. py-spy is unaffected (it
reads the interpreter) but sees only Python.


Requirements
~~~~~~~~~~~~~

Each mode wraps an external tool; a missing one fails fast naming what to
install.

* `py-spy <https://github.com/benfred/py-spy>`__ (``wall`` / ``gil``), by Ben
  Frederickson: reads the interpreter and renders its own SVG.
  ``pip install py-spy``.
* Linux ``perf`` (``on-cpu``), with the FlameGraph scripts below.
* `bcc <https://github.com/iovisor/bcc>`__ (``off-cpu`` / ``offwake`` /
  ``wakeup``), the IO Visor project's BPF Compiler Collection: its
  ``*-bpfcc`` tools need ``sudo``.
* `FlameGraph <https://github.com/brendangregg/FlameGraph>`__, by Brendan
  Gregg: folds and renders the perf/bcc SVGs. Point
  ``--flamegraph-scripts-dir`` (or ``FLAMEGRAPH_DIR``) at a checkout; default
  ``~/FlameGraph``, auto-cloned when absent.

**Naming Python frames in the perf/bcc modes** requires the target to have
been *started* with ``PYTHONPERFSUPPORT=1`` (its perf trampoline map); you
cannot enable it on a running process.

.. warning::

   * **If it was not set:** perf/bcc still record, but Python frames collapse
     to ``[unknown]`` (C/native stack only). Use ``gil`` / ``wall``, or
     restart the target with the variable set.
   * **If it is set:** it adds a standing per-call overhead for the process's
     lifetime, so reserve it for a dedicated profiling session, not
     production.

**Permissions (bare-metal / VM).** Attaching needs permission to trace the
target: ``wall`` / ``gil`` need ``kernel.yama.ptrace_scope`` at ``0`` (or
``CAP_SYS_PTRACE``); ``on-cpu`` needs ``kernel.perf_event_paranoid`` at ``2``
or lower; the bcc modes need ``sudo``. A blocked attach exits non-zero
naming the sysctl.

.. note::

   **Inside a container** it comes down to three separable things: what you
   **install**, which ``docker run`` **flags** you pass, and the **host
   sysctls** those flags bypass.

   **PID namespace:** the profiler must share the target's PID namespace, the
   **same container** or ``--pid=container:<target>`` / ``--pid=host``.

   The official images ship no profilers, and the default container drops the
   privileges each needs; the install goes *inside*, the flags *at launch*:

   .. list-table::
      :header-rows: 1
      :widths: 20 40 40

      * - Mode
        - Install (in the container)
        - ``docker run`` flags (at launch)
      * - ``wall`` / ``gil``
        - ``pip install py-spy``
        - ``--cap-add SYS_PTRACE`` (root alone is not enough).
      * - ``on-cpu``
        - ``linux-tools-generic``; if the wrapper says ``perf not found for
          kernel <ver>`` call ``/usr/lib/linux-tools/*/perf`` directly.
        - ``--security-opt seccomp=unconfined`` **and** ``--cap-add PERFMON``
          (older Docker: ``--cap-add SYS_ADMIN`` / ``--privileged``).
      * - ``off-cpu`` / ``offwake`` / ``wakeup``
        - ``bpfcc-tools`` **plus** the running kernel's headers
          (``/lib/modules/$(uname -r)/build``; BTF alone does not satisfy it).
        - ``seccomp=unconfined``, ``--cap-add BPF --cap-add PERFMON`` (or
          ``SYS_ADMIN`` / ``--privileged``), and
          ``-v /sys/kernel/debug:/sys/kernel/debug``.

   **Host sysctls** (``ptrace_scope``, ``perf_event_paranoid``) are read-only
   inside a container, but the capabilities above bypass them. Verified: at
   ``perf_event_paranoid=3``, ``perf_event_open`` fails with no cap and
   succeeds with ``--cap-add SYS_ADMIN``. They only gate you when you can pass
   *no* flags at all.

   **If you cannot re-launch the container** (RunPod, many Kubernetes
   setups): perf/bcc are effectively unavailable (their gates are not
   grantable from inside). ``wall`` / ``gil`` still work if the pod has
   ``CAP_SYS_PTRACE`` (check with ``py-spy dump --pid <target>``); if not,
   launch the target **under** py-spy, since tracing your own child needs no
   capability:

   .. code-block:: bash

      py-spy record --rate 200 --format flamegraph --threads --idle --gil \
          --subprocesses --duration 30 -o /workspace/flame.svg \
          -- <the target launch command>

   Drop ``--gil`` for ``wall``, or ``--format speedscope`` for an interactive
   chart. ``lmcache tool flamegraph --pid`` is attach-only; run ``py-spy``
   directly.


Cost
~~~~

The mechanisms give each mode a different cost *shape*:

* **perf**: *bounded*, a fixed 99 Hz sample, so overhead is constant; the
  cost is disk, as ``perf.data`` grows with duration × rate × threads ×
  depth (deleted once rendered).
* **bcc**: *unbounded*, firing on every scheduler event ties cost to the
  target's event rate, so a busy service fires the probe constantly. Output
  is tiny.
* **py-spy**: lightest *on the target*, its sampling runs off the target's
  CPUs, whereas perf and bcc steal the target's cycles.

On a production server, prefer ``wall`` / ``gil`` and keep ``--duration``
short; if you need a whole-process mode, ``on-cpu``'s bounded cost is safer
than the bcc modes on a busy target.
