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
fixed duration, and render a flame graph.

.. _lmcache-flamegraph-entry-points:

**Choosing a profiling entry point.** LMCache renders flame graphs from
three commands. They share the same ``--mode`` machinery but differ in
**where the load comes from** and **what they can attach to**:

.. list-table::
   :header-rows: 1
   :widths: 32 32 36

   * - Command
     - Load driven by
     - Profiles
   * - ``lmcache bench l2 --flamegraph on``
     - The benchmark itself (synthetic, in-process)
     - Its own process: a single L2 adapter under a microbenchmark. No
       separate server; profiles its own PID.
   * - ``lmcache bench server --flamegraph on``
     - The benchmark client (synthetic KV blobs over ZMQ)
     - An external server you point ``--profile-server-pid`` at, under that
       synthetic load.
   * - ``lmcache tool flamegraph`` *(this command)*
     - Nothing -- it drives no load
     - Any running process, under whatever **real** work it is already
       doing.

The two ``bench`` commands drive synthetic load; this one drives none.
Reach for it to profile a **real, unmodified process** -- a production or
vLLM-driven server under actual traffic (real ``LMCacheMPConnector`` paths,
not synthetic requests), an idle server, or any arbitrary PID -- without
standing up a benchmark harness.

.. code-block:: bash

   # Which threads of an MP cache server hold the GIL, sampled for 20s
   lmcache tool flamegraph --pid $(pgrep -f 'lmcache server') --mode gil --duration 20

.. _lmcache-flamegraph-modes:

The six ``--mode`` values fall into two families; pick by what you want to
see.

**Python execution or GIL contention** -- the *Python* modes ``gil`` /
``wall`` (``py-spy``). They give one root frame per thread and attribute
the interpreter lock, attach to an **unmodified** target, but see no
kernel frames and work only on a CPython process. ``gil`` is the default
for a live server.

* **gil** -- samples only threads holding the GIL, so interpreter-lock
  contention is directly visible.
* **wall** -- wall-clock time per thread, blocked threads included;
  separates a worker pool that the whole-process modes superimpose.

**CPU/IO time, kernel frames, context switches, or a non-Python process**
-- the *whole-process* modes (``perf`` / bcc). They sample every thread
(including native workers), resolve kernel frames, see scheduler activity
(blocking and wakeups) that the Python modes cannot, and profile any
process, but merge all threads into one chart.

* **on-cpu** (``perf record``) -- where CPU *cycles* go (serialization,
  copies, hashing).
* **off-cpu** (``offcputime-bpfcc``, bcc) -- time spent *blocked* (waiting
  on I/O, locks, eventfds). Often the more informative view for I/O-bound
  work.
* **offwake** (``offwaketime-bpfcc``, bcc) -- like ``off-cpu``, but each
  blocked stack is joined to the stack of the thread that *woke* it, so a
  single stack shows both where a waiter blocked (lower half) and who
  unblocked it (upper half, drawn inverted, coloured separately). Use it
  when ``off-cpu`` shows a big blocked tower and you need the *cause*.
* **wakeup** (``wakeuptime-bpfcc``, bcc) -- the mirror image: the stacks
  that spend time *doing* the waking, attributed by the sleep time they
  end.

**How each mode works.** All three run as external processes attached to
the target; what differs is *where the data is actually collected* --
inside the kernel (``perf``, bcc) or by reading the target's user-space
memory (``py-spy``):

* **perf** (``on-cpu``) -- the *kernel* samples the CPU at a fixed rate
  (99 Hz by default); on each tick it records the stack of whatever thread is
  *currently running* into a buffer the ``perf`` process drains. It
  therefore sees only on-CPU work -- a sleeping or blocked thread
  contributes nothing -- and walks native stacks through frame pointers.
  It is driven by ``perf_event_open``.
* **bcc / eBPF** (``off-cpu`` / ``offwake`` / ``wakeup``) -- *not* sampled.
  A small eBPF program is loaded into the kernel and fires on scheduler
  events: when a thread goes *off* the CPU it records that thread's stack
  and a timestamp, then charges the elapsed time when the thread is woken.
  That event-driven design is why it can measure *blocked* time -- and, for
  ``offwake``, tie a sleeper to its waker -- that ``perf`` cannot see, and
  why its overhead grows with how many context switches the target makes.
  Loading an eBPF program needs privilege (``sudo`` / ``CAP_BPF``).
* **py-spy** (``wall`` / ``gil``) -- does its work entirely in user space,
  in its *own* process: it reads the target's memory (``process_vm_readv``)
  and walks CPython's own interpreter structures to reconstruct Python
  stacks, with no kernel instrumentation and -- because this command always
  runs it with ``--nonblocking`` -- without pausing or modifying the
  target. It samples at an interval, sees only Python frames (never
  kernel/native ones), and for ``gil`` inspects the interpreter-lock state
  to count only GIL-holding threads. Reading another process's memory is
  why it needs ``ptrace`` permission.

This also explains the Python-frame gap below: ``perf`` and bcc walk
*native* stacks, where a chain of CPython calls collapses into one
``_PyEval_EvalFrameDefault`` frame with no per-function name -- hence the
trampoline / ``PYTHONPERFSUPPORT`` requirement. ``py-spy`` reads
interpreter state directly, so it needs none of that.

.. warning::

   The perf/bcc modes name Python functions only when the target was
   **started** with ``PYTHONPERFSUPPORT=1``. You cannot enable it on an
   already-running process:

   * **If it was not set:** these modes still record, but Python frames
     collapse to ``[unknown]`` -- you get the C/native stack only. Use
     ``gil`` / ``wall`` instead (they need no such flag), or restart the
     target with the variable set.
   * **If it is set:** it carries a standing per-call overhead for the
     process's whole lifetime, so reserve it for a dedicated profiling
     session, not production.

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

**On a VM or bare-metal host**, attaching needs permission to trace the
target: ``wall`` / ``gil`` require ``kernel.yama.ptrace_scope`` at ``0`` (or
``CAP_SYS_PTRACE``); ``on-cpu`` requires ``kernel.perf_event_paranoid`` at
``2`` or lower; the bcc modes load a BPF program and need ``sudo``. A
blocked attach exits non-zero naming the sysctl to run. **Inside a
container this works differently -- see below.**

.. note::

   **Profiling inside a container.** Nothing here edits the host kernel. It
   comes down to three separable things -- what you **install** in the
   container, which **flags** you pass to ``docker run``, and the **host
   sysctls** those flags exist to bypass. Keeping them apart:

   **PID visibility -- a** ``docker run`` **flag.** Every mode attaches by
   PID, so the profiler must share the target's PID namespace: run it in the
   **same container** as the target, or launch its container with
   ``--pid=container:<target>`` (or ``--pid=host``). Otherwise the target's
   PID is not visible.

   **Install (in the container) vs.** ``docker run`` **flags.** The official
   images (``lmcache/vllm-openai``) ship none of the profilers, and the
   default container drops the privileges each needs. These are independent:
   the install goes *inside* the container, the flags are set *at launch*.

   .. list-table::
      :header-rows: 1
      :widths: 20 40 40

      * - Mode
        - Install (in the container)
        - ``docker run`` flags (at launch)
      * - ``wall`` / ``gil`` (py-spy)
        - ``pip install py-spy`` -- a self-contained binary, the easy case.
        - ``--cap-add SYS_PTRACE``. Usually all that is needed; root alone is
          *not* enough -- without the cap the attach is denied even to uid 0.
      * - ``on-cpu`` (perf)
        - ``linux-tools-generic``; if the wrapper says ``perf not found for
          kernel <ver>`` call ``/usr/lib/linux-tools/*/perf`` directly (it
          records fine across minor-version gaps).
        - ``--security-opt seccomp=unconfined`` (default seccomp blocks
          ``perf_event_open``) **and** ``--cap-add PERFMON`` (older Docker
          rejects it -- use ``--cap-add SYS_ADMIN`` or ``--privileged``).
      * - ``off-cpu`` / ``offwake`` / ``wakeup`` (bcc)
        - ``bpfcc-tools`` **plus** the running kernel's headers
          (``/lib/modules/$(uname -r)/build``); ``*-bpfcc`` compiles at
          runtime, and on-disk ``/sys/kernel/btf/vmlinux`` alone does *not*
          satisfy it (that serves the separate ``libbpf-tools``).
        - ``--security-opt seccomp=unconfined``, ``--cap-add BPF --cap-add
          PERFMON`` (or ``SYS_ADMIN`` / ``--privileged``), and
          ``-v /sys/kernel/debug:/sys/kernel/debug``. eBPF is host-wide, so
          running these on the host is often simpler.

   **Host sysctls -- read-only inside, but bypassed by the flags.**
   ``kernel.yama.ptrace_scope`` and ``kernel.perf_event_paranoid`` are
   host-wide, not namespaced, and cannot be changed from inside a container.
   You do **not** need to change them if you can pass the capabilities:
   ``CAP_SYS_PTRACE`` bypasses ``ptrace_scope``, and ``CAP_PERFMON`` /
   ``CAP_SYS_ADMIN`` bypasses ``perf_event_paranoid`` (verified: at
   ``perf_event_paranoid=3``, ``perf_event_open`` fails with no cap and
   succeeds with ``--cap-add SYS_ADMIN``). The host sysctl only becomes the
   gate when you can pass *no* flags at all -- see below.

   **If you cannot re-launch the container** (RunPod, many Kubernetes
   setups): the container is started *for* you, so you cannot add
   ``--cap-add`` / ``--privileged`` and cannot change host sysctls
   (``perf_event_paranoid`` / ``ptrace_scope`` are read-only from inside).
   That leaves:

   * ``on-cpu`` (``perf``) and ``off-cpu`` / ``offwake`` / ``wakeup`` (bcc)
     are effectively **unavailable** -- their gate is
     ``perf_event_paranoid`` / ``CAP_BPF`` / ``CAP_PERFMON`` / a mounted
     ``/sys/kernel/debug``, none grantable from inside and none tied to *who
     launched the target*, so no in-container trick helps. (Only exception:
     if the platform runs with a low host ``perf_event_paranoid`` and
     permissive seccomp, ``on-cpu`` may work -- test with ``perf record -F99
     -g -o /tmp/t -- true``; an ``Operation not permitted`` means no.)
   * ``wall`` / ``gil`` (``py-spy``) still work **if** the pod was given
     ``CAP_SYS_PTRACE`` (check with ``py-spy dump --pid <target>``). If it
     was **not**, you can still profile *a process you start yourself*:
     ``ptrace_scope`` only blocks attaching to non-descendants, so tracing
     your own child needs no capability. Launch the target **under** py-spy
     instead of attaching to it:

     .. code-block:: bash

        # equivalent of `--mode gil`, but py-spy launches (and so may trace)
        # the target as its own child; --subprocesses follows forked workers
        py-spy record --rate 200 --format flamegraph --threads --idle --gil \
            --subprocesses --duration 30 -o /workspace/flame.svg \
            -- <the target launch command>

     Drop ``--gil`` for the ``wall`` view, or use ``--format speedscope`` to
     open the run as an interactive flame *chart* at
     https://www.speedscope.app. Note ``lmcache tool flamegraph --pid`` is
     attach-only and cannot use this path; run ``py-spy`` directly.

**Cost of recording.** Those mechanisms give each mode a different cost
*shape*:

* **perf** (``on-cpu``) -- *bounded*. Sampling at a fixed rate makes the
  overhead roughly constant however busy the target is; the main cost is
  disk, as ``perf.data`` grows with ``--duration`` × rate × thread count ×
  stack depth (deep stacks and many threads enlarge it, the workload's
  event rate does not). It is deleted once the SVG renders.
* **bcc** (``off-cpu`` / ``offwake`` / ``wakeup``) -- *unbounded*. Firing on
  every scheduler event ties the cost to the target's *event rate*, not to
  wall-clock: a service doing millions of tiny I/Os or contending on locks
  fires the probe constantly, and ``wakeup`` on a wakeup-heavy workload is
  the heaviest. Its output is tiny (aggregated in-kernel), so disk cost is
  negligible.
* **py-spy** (``wall`` / ``gil``) -- lightest *on the target*: its sampling
  runs in its own process, off the target's CPUs, whereas perf's interrupt
  and bcc's probe run on the target's CPUs and steal its cycles. The target
  pays only for one ``process_vm_readv`` per sample.

On a production server, prefer ``wall`` / ``gil`` and keep ``--duration``
short. If you need a whole-process mode, ``on-cpu``'s bounded cost is safer
on a busy target than the bcc modes, whose cost climbs with the very event
rate a busy server has most of.

**Native stacks need frame pointers.** All four whole-process modes unwind
native stacks through frame pointers -- ``perf`` via ``perf record -g``,
bcc via ``bpf_get_stackid`` -- so a target built with
``-fomit-frame-pointer`` (common in optimized builds and some C extensions)
shows broken or shallow C stacks in *any* of them, and switching perf↔bcc
does not help. Rebuild with ``-fno-omit-frame-pointer`` for full native
stacks. The py-spy modes are unaffected -- they read the interpreter, not
the native stack -- but see only Python.

**Built on.** These modes are thin wrappers around three excellent
open-source profilers, plus Linux ``perf``. When a required tool is
missing, the command fails fast with a message naming what to install.

`py-spy <https://github.com/benfred/py-spy>`__, by Ben Frederickson, powers
the ``wall`` and ``gil`` modes. It reads the interpreter directly and
renders its own SVG (so the FlameGraph scripts below are not needed for
these modes). Install it with ``pip install py-spy``.

`bcc <https://github.com/iovisor/bcc>`__, the IO Visor project's BPF
Compiler Collection, powers the ``off-cpu``, ``offwake``, and ``wakeup``
modes through its ``offcputime-bpfcc``, ``offwaketime-bpfcc``, and
``wakeuptime-bpfcc`` tools. (The ``on-cpu`` mode instead uses Linux
``perf``.)

`FlameGraph <https://github.com/brendangregg/FlameGraph>`__, by Brendan
Gregg, folds and renders the SVG for the ``perf`` and bcc modes. Point
``--flamegraph-scripts-dir`` (or ``FLAMEGRAPH_DIR``) at a checkout; the
default is ``~/FlameGraph``, auto-cloned to a temp directory when absent.
