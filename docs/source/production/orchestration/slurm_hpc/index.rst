.. _hpc_deployment:

HPC / Slurm deployment (Apptainer, rootless, shared filesystem)
===============================================================

This section covers running LMCache on HPC / supercomputing clusters. It is
written as a recipe for the common *restricted* HPC profile below; not every
site imposes every restriction, so skip the parts yours does not:

- **Jobs run under a batch scheduler** (Slurm ``sbatch`` / ``srun``) rather than
  a long-lived container or Kubernetes pod.
- **You do not have root.** Containers run with a rootless runtime
  (Apptainer / Singularity) instead of the Docker daemon.
- **Storage is a shared parallel filesystem** (Lustre / GPFS / BeeGFS) mounted
  on every node, plus possibly small node-local scratch.
- **Compute-node internet access is restricted or absent at many sites.** This
  guide assumes the fully offline case: anything that would normally be
  downloaded at runtime (model weights, tokenizers, pip packages, container
  images) is staged from a login node first.

Each of these dimensions is covered on its own page:

.. toctree::
   :maxdepth: 2

   container_runtime
   shared_filesystem
   offline_staging

.. contents::
   :local:
   :depth: 2

This section assumes you already understand the generic LMCache + engine wiring
in :doc:`/getting_started/quickstart` and :doc:`/mp/deployment`; it only adds
the HPC-specific pieces.

Prerequisites
-------------

- A cluster with Slurm and either **Apptainer** (formerly Singularity) or
  **Singularity CE**. Check with ``apptainer --version`` or
  ``singularity --version`` on a login node.
- A GPU partition with NVIDIA GPUs and a compatible driver on the compute nodes.
- A directory on the shared filesystem you can write to from both login and
  compute nodes (referred to below as ``$PROJECT``; substitute your site's
  shared-filesystem location).

.. note::

   The exact container image tags, module names, and partition names below are
   placeholders. Substitute the values for your site.

Once you have staged an image (:doc:`container_runtime`), staged your models
(:doc:`offline_staging`), and chosen a KV cache path (:doc:`shared_filesystem`),
the pieces come together in a single-node batch job.

.. _hpc_single_node_submission:

Single-node Slurm submission
----------------------------

A minimal ``sbatch`` script that runs the LMCache-enabled engine inside
Apptainer on one GPU node:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=lmcache-serve
    #SBATCH --partition=<gpu_partition>
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=12
    #SBATCH --time=01:00:00

    set -euo pipefail

    export PROJECT=<shared_fs_path>/lmcache

    # Offline model cache (see "Network-less / offline model staging")
    export HF_HOME=$PROJECT/hf_cache
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    # LMCache KV tiers (single-GPU job -> one disk path; for multi-GPU list
    # one path per GPU -- see "Shared filesystem")
    export LMCACHE_CHUNK_SIZE=256
    export LMCACHE_LOCAL_CPU=True
    export LMCACHE_MAX_LOCAL_CPU_SIZE=20
    export LMCACHE_LOCAL_DISK="file://$PROJECT/kv_cache"
    export LMCACHE_MAX_LOCAL_DISK_SIZE=100

    # Short node-local IPC base (see "Rootless container runtime")
    export VLLM_RPC_BASE_PATH=/tmp/lmc_$SLURM_JOB_ID
    mkdir -p "$VLLM_RPC_BASE_PATH"

    apptainer exec --nv \
        --bind "$PROJECT" \
        --bind "$VLLM_RPC_BASE_PATH" \
        $PROJECT/lmcache.sif \
        vllm serve <org/model> \
            --kv-transfer-config \
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Key HPC-specific flags:

- ``apptainer exec --nv`` -- exposes the NVIDIA driver/GPUs to the container
  (the rootless equivalent of ``docker run --gpus all``).
- ``--bind`` -- paths outside the runtime's default bind set must be passed
  explicitly with ``--bind``. Explicit binding also makes the job independent
  of site-specific Apptainer defaults. Bind ``$PROJECT`` (model cache + KV
  cache) and the node-local IPC base.
- ``--gres=gpu:N`` -- request GPUs from Slurm; ``--nv`` then passes them through.
- ``--cpus-per-task`` -- sites often cap CPUs per allocated GPU (e.g. 12); an
  over-ask is rejected at submit time.

This uses the **in-process** connector (``LMCacheConnectorV1``), which needs no
extra server process inside the job.

.. note::

   **Multiprocess (MP) mode under a scheduler is not yet covered here.** For MP
   mode (a separate ``lmcache server`` plus ``LMCacheMPConnector``) see
   :doc:`/mp/deployment`; the container, offline, path, and IPC rules in this
   section apply the same way, but the MP topology under Slurm/Apptainer has not
   been exercised in this guide -- validate it on your cluster before relying on
   it.

Multi-node notes
----------------

.. warning::

   This section is a starting point, not a validated recipe. Multi-node LMCache
   on HPC needs to be confirmed on the target cluster before relying on it.

- **IPC sockets are per-node.** ``VLLM_RPC_BASE_PATH`` must resolve to a
  node-local path on *every* node (``/tmp/...`` or ``/dev/shm/...``), never the
  shared filesystem, both for the 107-byte pathname limit and because
  Unix-domain sockets do not work across nodes.
- **Disk-tier paths give no node-level isolation by themselves.** ``by_gpu``
  path sharding selects from the configured path list by *node-local* GPU
  index, so identical lists on different nodes select the same entries; keep
  the disk tier on node-local scratch, or have the launcher generate per-node
  paths (see :doc:`shared_filesystem`).
- **Cross-node cache sharing** goes through a remote backend
  (``LMCACHE_REMOTE_URL``, e.g. a Redis/Valkey/S3-compatible endpoint reachable
  on the cluster network), not through the node-local tiers.
- **Interconnect:** high-speed fabrics (InfiniBand / Slingshot) affect
  remote-backend and disaggregated-prefill throughput; confirm the container
  can use the fabric (bind the relevant devices/libraries).

Known pitfalls at a glance
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Fix
   * - ``ZMQError: ipc path ... longer than 107 characters``
     - Set ``VLLM_RPC_BASE_PATH`` to a short node-local dir
       (:doc:`container_runtime`, `#3529
       <https://github.com/LMCache/LMCache/issues/3529>`_).
   * - Model download hangs / fails on a compute node
     - Pre-stage into ``HF_HOME`` on shared FS; set ``HF_HUB_OFFLINE=1``
       (:doc:`offline_staging`).
   * - Container cannot see GPUs
     - Use ``apptainer exec --nv`` and request ``--gres=gpu``
       (:ref:`sbatch template <hpc_single_node_submission>`).
   * - ``FileNotFoundError`` for a path that exists on the host
     - The path is outside the container runtime's default bind set; pass it
       explicitly with ``--bind``
       (:ref:`sbatch template <hpc_single_node_submission>`).
   * - Engine dies on the first KV store (e.g.
       ``Unsupported EngineKVFormat``)
     - vLLM/LMCache mismatch inside the image -- pin a mutually compatible
       release tag and validate before staging (:doc:`container_runtime`).
   * - Restarted job ignores the KV cache files from the previous run
     - Expected today: the disk-tier index is in-memory only and is not
       rebuilt from existing files (:doc:`shared_filesystem`).
