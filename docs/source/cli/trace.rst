lmcache trace
=============

The ``lmcache trace`` command inspects and replays LMCache trace files
(``.lct``). A file holds one trace *level*: ``storage`` records
``StorageManager`` calls; ``events`` records the cache-event stream a server
emits for the MP coordinator. It has three sub-commands:

.. code-block:: bash

   lmcache trace {info,replay,replay-events} FILE [options]

.. note::

   ``lmcache trace`` needs the full ``lmcache`` package (StorageManager,
   trace codecs, ``TraceReader``). It is not available in the lightweight
   ``lmcache-cli`` install and exits with status ``2`` if those modules are
   missing.

Trace *capture* is not a ``trace`` sub-command — recording is bound to a live
server via ``lmcache server --trace-level {storage,events} [--trace-output ...]``
(see :doc:`server`).


info
----

Print a one-screen summary of a trace file: header metadata plus per-qualname
record counts. Works for both levels.

.. code-block:: bash

   lmcache trace info path/to/trace.lct

A ``storage`` file:

.. code-block:: text

   Trace file: path/to/trace.lct
     level:                      storage
     format_version:             1
     trace_schema_version:       1
     duration:                   12.345s
     sm_config_digest:           a1b2c3d4
     total_records:              2048
     ops:
       StorageManager.store: 1024
       StorageManager.retrieve: 1024

An ``events`` file adds the header metadata the level records and a
summary of the stream: which server wrote it and how many times it
restarted (one incarnation per start), batches by event type, tier, backend
and whether the backend is shared, entries per event type, bytes stored, and
how many store entries carry token ids.

.. code-block:: text

   Trace file: events-node-a.lct
     level:                      events
     format_version:             1
     trace_schema_version:       1
     duration:                   612.480s
     sm_config_digest:           7a10b9a5c8e8344d32b5f8b7a530327a19f189439f5d2d5151065fd5d19ecff8
     cache_event_schema_version: 1
     instance_id:                node-a
     lmcache_version:            0.5.6
     total_records:              8
     ops:
       events.batch: 6
       events.lifecycle: 2
     events:
       instances:                  1
         node-a: incarnations=[1758400000] restarts=0
       batches (type/tier/backend/shared):
         access/l1/-/local: 1
         delete/l1/dram/local: 1
         store/l1/dram/local: 2
         store/l2/fs/local: 1
         store/l2/s3/shared: 1
       entries:
         access: 30
         delete: 12
         store: 128
       store_bytes:                113246208
       entries_with_tokens:        64
       lifecycle:                  start=1, stop=1

``lifecycle: start=1, stop=1`` says the server started once and shut down
cleanly; a missing ``stop`` means the process was killed and the tail may be
truncated. ``restarts`` counts incarnations beyond the first.

The only argument is the positional ``FILE`` (path to a ``.lct`` trace file).


replay
------

Reissue every recorded call against a fresh ``StorageManager``, honoring the
recorded inter-call timings. ``replay`` accepts ``storage`` files only; an
``events`` file is refused with a message saying so, since its records are
coordinator input rather than storage calls (see ``replay-events``).

.. code-block:: bash

   lmcache trace replay path/to/trace.lct \
       --l1-size-gb 10 --eviction-policy LRU

``replay`` accepts the standard storage-manager configuration flags
(``--l1-size-gb``, ``--eviction-policy``, ``--l2-...``); see
``lmcache server --help`` for the full list. The replay-side config may
differ from the config recorded in the trace, which can legitimately cause
retrieve misses.

.. warning::

   A replay environment mismatch may cause retrieve misses. Replay uses the
   replay-side StorageManager config (which may differ from the recorded
   config), runs on a host whose performance may differ from the recording
   host, and StorageManager reads/writes are async. Treat retrieve-miss
   counts as a signal about the replay environment, not a defect in the trace.

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``FILE``
     - Path to a ``.lct`` trace file (positional, required).
   * - ``--verbose``
     - Print one line per replayed record.
   * - ``--jsonl-out PATH``
     - Write one JSON object per replayed record to ``PATH`` (qualname,
       latency_ms, failed).
   * - ``--output-dir DIR``
     - Directory for aggregated CSV / JSON summary output (default: current
       directory).
   * - ``--no-csv``
     - Skip the aggregated CSV summary export.
   * - ``--json``
     - Also export an aggregated JSON summary.
   * - ``-q`` / ``--quiet``
     - Suppress the terminal metrics table (files are still written).

The terminal summary reports overall replay stats (records replayed /
skipped / failed, duration, config-digest match) and per-op latency
percentiles. ``replay`` exits with status ``1`` if any record failed.


replay-events
-------------

Deliver the cache-event stream an ``events`` file holds to a coordinator,
the way the server that wrote it would have. Several files, one per server,
replay as one fleet, merged by wall-clock time:

.. code-block:: bash

   lmcache trace replay-events node-a.lct node-b.lct \
       --coordinator-url http://coordinator:9300

Each ``start`` mark registers its server with the coordinator (at the
address the server recorded, or loopback) and begins heartbeating it every
``--heartbeat-interval`` seconds, each batch is one ``POST /events``
carrying the record unchanged, and each ``stop`` mark deregisters. A file
that ends without a ``stop`` is a server that died or a recording that was
cut: its heartbeats end with its last record, and the coordinator retires
it after its instance timeout, as it would a real crash. By default the
stream goes as fast as the coordinator takes it; ``--speed 1`` replays at
the recorded rate. The servers are not there to be called back, so a replay
exercises ingest and the views (directory, usage, declared capacity), not a
controller's deletes or prefetches; run the coordinator with its eviction
loop off to keep it from addressing them. The summary says how the
coordinator counted the batches: applied, duplicate (a sequence it had
already admitted) and stale (an incarnation it had already moved past).

.. warning::

   Replay into a coordinator that does not have the same fleet live.
   Registering a recorded server replaces a live one's registration, and
   the coordinator would address the live server at the recorded address.

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``FILE``
     - One or more ``events``-level ``.lct`` files (positional, required).
   * - ``--coordinator-url URL``
     - The coordinator to deliver the stream to (required).
   * - ``--speed N``
     - Pacing relative to the recording; ``1`` is the recorded rate, ``0``
       (default) as fast as the coordinator takes it.
   * - ``--heartbeat-interval SECONDS``
     - Heartbeat each registered server this often (default ``5``, below
       the coordinator's 30 s instance timeout); ``0`` sends none.
   * - ``-q`` / ``--quiet``
     - Suppress the terminal summary.

``replay-events`` exits with status ``2`` if a file is not an ``events``
trace this build can replay, and with status ``1`` if the coordinator
cannot be reached or refuses a call; the replay stops there rather than
skip records, since a partly delivered stream would leave views that look
complete and are not. The per-record log says how far it got.
