.. _hpc_shared_filesystem:

Shared filesystem (KV cache paths)
==================================

On HPC clusters the primary storage is a **shared parallel filesystem**
(Lustre / GPFS / BeeGFS) mounted on every node, often alongside small
node-local scratch. This page covers where to put LMCache's disk tier; it is
used by the :ref:`single-node sbatch template <hpc_single_node_submission>`.

LMCache's disk tier is controlled by ``LMCACHE_LOCAL_DISK`` (a ``file://`` URI
or bare path; comma-separate several) and capped by
``LMCACHE_MAX_LOCAL_DISK_SIZE`` (GB). On HPC you must decide **where** that path
lives:

- **Node-local NVMe / scratch** (e.g. ``/local/$SLURM_JOB_ID``): lowest latency,
  gone when the job ends. **Usually the right choice** -- within a job the disk
  tier acts as spill capacity below the CPU-RAM tier.
- **Shared parallel filesystem** (``$PROJECT``): the KV files survive the job
  and the disk tier works correctly on Lustre/GPFS, **but a new process does
  not reuse them**: the disk tier's index lives in memory and is not rebuilt
  from existing files on startup, so a restarted engine treats the old files
  as a miss and re-stores them. Do not choose the shared filesystem expecting
  warm-start across jobs.

.. code-block:: bash

    export PROJECT=<shared_fs_path>/lmcache       # your shared project directory
    export LMCACHE_CHUNK_SIZE=256
    export LMCACHE_LOCAL_CPU=True
    export LMCACHE_MAX_LOCAL_CPU_SIZE=20          # GB of pinned host memory
    export LMCACHE_LOCAL_DISK="file://$PROJECT/kv_cache"
    export LMCACHE_MAX_LOCAL_DISK_SIZE=100        # GB

Multiple GPUs and path sharding
-------------------------------

``LMCACHE_LOCAL_DISK`` accepts a comma-separated list of paths, and
``LMCACHE_LOCAL_DISK_PATH_SHARDING=by_gpu`` (the default and currently the only
strategy) makes each worker select **one of the listed paths** --
``paths[device_id % len(paths)]``, where ``device_id`` is the worker's
node-local GPU index. No per-GPU subdirectories are created: with a single
configured path, every GPU on the node writes into that same directory.

- **Single node, multiple GPUs:** list one path per GPU explicitly, e.g. for a
  4-GPU job on node-local scratch:

  .. code-block:: bash

      export LMCACHE_LOCAL_DISK=/local/$SLURM_JOB_ID/cache0,/local/$SLURM_JOB_ID/cache1,/local/$SLURM_JOB_ID/cache2,/local/$SLURM_JOB_ID/cache3

- **Multiple nodes:** GPU indices restart at 0 on every node, so identical path
  lists on different nodes select the same entries. Node-local paths (as above)
  are naturally isolated per node. If the listed paths are on the shared
  filesystem, ``by_gpu`` provides **no node-level isolation** -- have your
  launcher generate per-node paths (e.g. embed ``$SLURM_NODEID`` or the
  hostname in the path), or use a remote backend for cross-node sharing.
