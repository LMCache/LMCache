Phoenix (PHX)
=============

An L2 adapter backed by `Phoenix <https://github.com/xPU-IO/phoenix>`__, a
GPU-direct I/O stack that turns NVMe SSDs into peer devices of the GPU.  It
issues ``O_DIRECT`` I/O on the file descriptors it is given and DMA's the
data straight from NVMe into GPU (device) memory -- no GPUDirect Storage
(``nvidia-fs``), no special file system, and no RDMA stack required.

The adapter is **asymmetric**:

- **Store (L1 -> L2):** the CPU-resident KV object is written to a file
  under ``base_path`` (via ``phxfs_write_batch`` / POSIX write).  The data
  already lives in CPU memory, so no device DMA is involved.
- **Load (L2 -> L1):** the KV object is DMA'd from NVMe into a
  **device-resident** memory object (via ``phxfs_read``), then scattered
  device-to-device (D2D) into vLLM's paged KV cache on retrieve.  This
  avoids the CPU round-trip that file-based adapters pay on the load path.

**Prerequisites**

- **OS**: Linux x86_64 (tested on Ubuntu 22.04, kernel 6.1).
- **Hardware**: NVIDIA GPUs + NVMe SSDs (local NVMe, NVMe-oF, or NFS all
  work; FUSE is unsupported because its direct-I/O path cannot carry P2P
  DMA).
- **Kernel**: a kernel source / module tree matching your running kernel
  (required to build the ``phoenixfs`` kernel module).
- **CUDA**: a CUDA toolkit (its installer provides the matching NVIDIA
  driver used by the NVIDIA vendor backend).  No ``nvidia-fs`` / GPUDirect
  Storage needed.
- **Build tools**: CMake >= 3.18, a CUDA-capable toolchain, ``liburing``.
- Phoenix kernel module (``phoenixfs``) + user library (``libphoenix.so``).
- ``phxcache`` Python package (pybind11 wrapper around ``libphoenix.so``).

**Installation**

1. Build and install Phoenix (kernel module + user library):

   .. code-block:: bash

       git clone https://github.com/xPU-IO/phoenix.git
       cd phoenix
       mkdir -p build && cd build
       cmake ../
       make -j
       sudo make install        # libphoenix.so → /usr/local/lib
       nvidia-smi               # ensure the NVIDIA driver is loaded first
       sudo make insmod         # loads phoenixfs.ko

   Verify the module is loaded and device nodes exist:

   .. code-block:: bash

       lsmod | grep phoenixfs
       ls /dev/phxfs_dev*       # /dev/phxfs_dev0, /dev/phxfs_dev1, ...

2. Build the ``phxcache`` Python package (from the Phoenix repo):

   .. code-block:: bash

       cd phoenix/adapters/lmcache/phxcache
       bash install.sh          # auto-detects conda env, sets CUDA_HOME etc.

   Phoenix is self-contained: it requires only the device nodes above and
   ``libphoenix.so`` on the library path.  See the Phoenix install guide for
   BAR mapping modes (STAGING vs FULL), vendor backend selection, and
   troubleshooting.

**Configuration**

Add a ``phx`` L2 adapter to your LMCache MP server config:

.. code-block:: bash

    --l2-adapter '{
      "type": "phx",
      "base_path": "your kvcache path",
      "device_ids": [4, 5, 6, 7],
      "buffer_size_mb": 2048
    }'

**Required fields:**

- ``base_path``: Root directory for the KV cache files.

**Optional fields:**

- ``device_ids`` (list[int], default ``None``): Device IDs for the phxfs
  DMA path, one buffer pool per device.  Single device: ``[4]``;
  multi-device: ``[4, 5, 6, 7]``.  When omitted (or empty), the adapter
  falls back to plain POSIX reads.
- ``buffer_size_mb`` (int, default ``2048``): GPU buffer pool size in MiB
  **per device**.
- ``use_direct_io`` (bool, default ``true``): Use ``O_DIRECT`` for I/O.
- ``max_capacity_bytes`` (int, default ``0``): Maximum storage capacity in
  bytes; ``0`` means unlimited.
- ``perf_log_dir`` (str, default ``None``): When set, writes a perf log
  (hit rate + per-phase timing for store/load/lookup) to
  ``perf_log_dir/phx_perf.log``.

**L1 integration (automatic)**

When any L2 adapter has ``"type": "phx"``, L1 automatically switches to
:class:`PhxL1MemoryManager` so that device-resident objects injected by the
adapter are dispatched back to their pool on free.  This is inferred from
the L2 adapter config at parse time -- no extra configuration is needed.

**Fallback behavior**

The adapter degrades gracefully to POSIX reads whenever the Phoenix DMA path
is unavailable:

- ``phxcache`` is not installed, or
- no ``device_ids`` are configured, or
- device initialization fails (e.g. no matching ``/dev/phxfs_dev*`` node).

Stores always go through the CPU path, so only loads are affected by the
fallback.

**Notes**

- ``phxcache`` automatically discovers the ``/dev/phxfs_dev*`` node for each
  CUDA device id in ``device_ids`` (matched by PCI BDF), so no manual
  device-to-phxfs mapping is required.
- Data is written as plain files under ``base_path``, one file per KV
  object, so the on-disk format is inspectable with regular tools.
