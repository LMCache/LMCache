.. _hpc_container_runtime:

Rootless container runtime
==========================

Many shared HPC systems do not grant users root access or expose a Docker
daemon on compute nodes. In that environment, containers commonly run through a
**rootless runtime** such as Apptainer (formerly Singularity) or Singularity
CE, and the runtime's node-local socket paths must stay within the Unix
path-length limit. This page covers obtaining an image and keeping the IPC
socket path short; it is used by the
:ref:`single-node sbatch template <hpc_single_node_submission>`.

.. contents::
   :local:
   :depth: 1

Obtain a container image on a login node
----------------------------------------

In the fully offline compute-node profile assumed by this guide, build or
convert the image on a login node (which has internet) and store the resulting
``.sif`` on shared storage.

**Option A -- convert an official image (simplest):**

.. code-block:: bash

    # On a login node, with internet access.
    export PROJECT=<shared_fs_path>/lmcache              # your shared project directory
    export APPTAINER_CACHEDIR=$PROJECT/apptainer_cache   # keep cache off $HOME
    apptainer build $PROJECT/lmcache.sif \
        docker://lmcache/vllm-openai:v0.5.1     # version validated by this guide

No root and no ``--fakeroot`` is needed for this conversion -- Apptainer
performs it fully rootless (harmless ``setxattr`` warnings during unpacking are
expected). Check the available tags on
`Docker Hub <https://hub.docker.com/r/lmcache/vllm-openai/tags>`_. Replace this
tag only after validating the LMCache/vLLM combination required by your
deployment.

.. warning::

   For reproducible HPC deployments, **pin a mutually compatible
   LMCache/vLLM image tag and validate it before staging to offline compute
   nodes**. Nightly images may change KV-cache layouts or native-kernel
   support between builds, and on a cluster you cannot quickly iterate images
   from a compute node, so a broken image costs a full stage-build-submit
   cycle.

.. note::

   Dated example of the above: the ``latest-nightly`` image pulled on
   2026-07-16 (vLLM 0.23.1-nightly + LMCache 0.5.2.dev52) failed at the first
   KV store with ``RuntimeError: Unsupported EngineKVFormat`` (issue
   `#4111 <https://github.com/LMCache/LMCache/issues/4111>`_, fixed by
   `PR #4128 <https://github.com/LMCache/LMCache/pull/4128>`_ on 2026-07-17),
   while the pinned ``v0.5.1`` release worked end-to-end on the same cluster.
   The failure is specific to that build; the takeaway is the pin-and-validate
   workflow, not any particular tag.

**Option B -- build from a definition file** when you need extra site packages
or a pinned LMCache version:

.. code-block:: bash

    apptainer build --fakeroot $PROJECT/lmcache.sif lmcache.def

.. note::

   ``--fakeroot`` requires uid/gid mappings in ``/etc/subuid`` / ``/etc/subgid``,
   which many HPC sites do not provision. If it is unavailable, build the
   ``.sif`` where you do have privileges (a workstation, or a remote builder)
   and copy the file to ``$PROJECT`` -- or stick with Option A, which needs no
   privileges at all.

.. _hpc_ipc_socket_path:

Avoid the IPC socket path length limit
--------------------------------------

LMCache multiprocess mode uses ZMQ ``ipc://`` Unix-domain sockets for
lookup/offload RPC. A Unix-domain socket pathname is limited to **107 usable
bytes** (``sockaddr_un.sun_path`` is a 108-byte array including the trailing
``NUL``). LMCache builds the path as::

    {base}/engine_{engine_id}_service_{service}_lmcache_rpc_port_{port}

where ``base`` comes from vLLM's ``VLLM_RPC_BASE_PATH``, which **defaults to
the system temp directory and therefore follows ``$TMPDIR``** (LMCache falls
back to ``/tmp/vllm_rpc`` only when vLLM is not importable). HPC schedulers
commonly point ``$TMPDIR`` at a long per-job scratch path -- and with a UUID
engine id the suffix alone is already ~80 bytes, so the socket path
overflows and the worker aborts with::

    zmq.error.ZMQError: ipc path "/p/scratch/.../engine_<uuid>_service_lookup_
    lmcache_rpc_port_1" is longer than 107 characters

(reported from a real supercomputer in issue
`#3529 <https://github.com/LMCache/LMCache/issues/3529>`_). Point
``VLLM_RPC_BASE_PATH`` at a **short, node-local** directory -- IPC socket files
are tiny, so node-local ``/tmp`` or ``/dev/shm`` is ideal and keeps the path
short:

.. code-block:: bash

    export VLLM_RPC_BASE_PATH=/tmp/lmc_$SLURM_JOB_ID
    mkdir -p "$VLLM_RPC_BASE_PATH"

.. note::

   A fix that keeps the generated path within the limit is proposed in
   `PR #3530 <https://github.com/LMCache/LMCache/pull/3530>`_. Until PR #3530
   is merged and included in the LMCache version you deploy, set
   ``VLLM_RPC_BASE_PATH`` to a short node-local directory.
