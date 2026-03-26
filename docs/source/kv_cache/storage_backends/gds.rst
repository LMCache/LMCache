GDS Backend
==================

.. _gds-overview:

Overview
--------

This backend will work with any file system, whether local, remote, and remote
with GDS-based optimizations. Remote file systems allow for multiple LMCache
instances to share data seamlessly. The GDS (GPU-Direct Storage) optimizations
are used for zero-copy I/O from GPU memory to storage systems. Supports both
NVIDIA cuFile and AMD hipFile for GPU-direct storage.


Ways to configure LMCache GDS Backend
-----------------------------------------

**1. Environment Variables:**

.. code-block:: bash

    # 256 Tokens per KV Chunk
    export LMCACHE_CHUNK_SIZE=256
    # Path to store files
    export LMCACHE_GDS_PATH="/mnt/gds/cache"
    # CuFile Buffer Size in MiB
    export LMCACHE_CUFILE_BUFFER_SIZE="8192"
    # Disabling CPU RAM offload is sometimes recommended as the
    # CPU can get in the way of GPUDirect operations
    export LMCACHE_LOCAL_CPU=False

**2. Configuration File**:

Passed in through ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml``

Example ``config.yaml``:

.. code-block:: yaml

    # 256 Tokens per KV Chunk
    chunk_size: 256
    # Disable local CPU
    local_cpu: false
    # Path to file system, local, remote or GDS-enabled mount
    gds_path: "/mnt/gds/cache"
    # CuFile Buffer Size in MiB
    cufile_buffer_size: 8192


CuFile Buffer Size Explanation
------------------------------

The backend currently pre-registers buffer space to speed up cuFile operations. This buffer space
is registered in VRAM so options like ``--gpu-memory-utilization`` from ``vllm`` should be considered
when setting it. For example, a good rule of thumb for H100 which generally has 80GiBs of VRAM would
be to start with 8GiB and set ``--gpu-memory-utilization 0.85`` and depending on your workflow fine-tune
it from there.


Using AMD hipFile
-----------------

.. note::

   hipFile is alpha software and has been tested on limited hardware.
   For full installation details, see the
   `hipFile install guide <https://github.com/ROCm/hipFile/blob/develop/INSTALL.md>`__.

**Prerequisites:**

- **ROCm >= 7.2** with ``amdgpu-dkms >= 30.20.1``
  (see the `ROCm quick start installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`__)
- **Supported storage:** local NVMe drives only
- **Supported filesystems:** ext4 (mounted with ``data=ordered``) and xfs
- **Kernel:** ``CONFIG_PCI_P2PDMA`` must be enabled

**Quick install (Ubuntu 24.04):**

.. code-block:: bash

    sudo apt install libmount-dev wget

    # Install nightly hipFile packages
    wget https://github.com/ROCm/hipFile/releases/download/nightly/hipfile_0.2.0.70200-nightly.9999.24.04_amd64.deb
    wget https://github.com/ROCm/hipFile/releases/download/nightly/hipfile-dev_0.2.0.70200-nightly.9999.24.04_amd64.deb
    sudo dpkg -i hipfile-dev_0.2.0.70200-nightly.9999.24.04_amd64.deb hipfile_0.2.0.70200-nightly.9999.24.04_amd64.deb

You can verify that the HIP libraries and kernel support AIS (AMD Infinity Storage) by running:

.. code-block:: bash

    /opt/rocm/bin/ais-check

Successful output will show ``True`` for ``Kernel P2PDMA support``, ``HIP runtime``, and ``amdgpu``.

**LMCache configuration:**

To use AMD hipFile instead of NVIDIA cuFile, add the following to your configuration:

**Environment Variables:**

.. code-block:: bash

    export LMCACHE_EXTRA_CONFIG='{"use_hipfile": true}'

**Configuration File:**

.. code-block:: yaml

    extra_config:
        use_hipfile: true

Note: The ``cufile_buffer_size`` configuration is used for both cuFile and hipFile buffers.


Setup Example
-------------

.. _gds-prerequisites:

**Prerequisites:**

- A Machine with at least one GPU. You can adjust the max model length of your vllm instance depending on your GPU memory.

- A mounted file system. A file system supportings GDS will work best.

- vllm and lmcache installed (:doc:`Installation Guide <../../getting_started/installation>`)

- Hugging Face access to ``meta-llama/Llama-3.1-8B-Instruct``

.. code-block:: bash

    export HF_TOKEN=your_hugging_face_token

**Step 1. Create cache directory under your file system mount:**

To find all the types of file systems supporting GDS in your system, use `gdscheck` from NVIDIA:

.. code-block:: bash

    sudo /usr/local/cuda-*/gds/tools/gdscheck -p

Check with your storage vendor on how to mount the remote file system.

(For example, if you want to use a GDS-enabled NFS driver, try the modified [NFS
stack](https://vastnfs.vastdata.com/), which is an open source driver that
works with any standard [NFS
RDMA](https://datatracker.ietf.org/doc/html/rfc5532) server. More
vendor-specific instructions will be added here in the future).

Create a directory under the file systew mount (the name here is arbitrary):

.. code-block:: bash

    mkdir /mnt/gds/cache

**Step 2. Start a vLLM server with file backend enabled:**

Create a an lmcache configuration file called: ``gds-backend.yaml``

.. code-block:: yaml

    local_cpu: false
    chunk_size: 256
    gds_path: "/mnt/gds/cache"
    cufile_buffer_size: 8192

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # LMCACHE_LOCAL_CPU=False \
    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_GDS_PATH="/mnt/gds/cache" \
    # LMCACHE_CUFILE_BUFFER_SIZE=8192 \
    LMCACHE_CONFIG_FILE="gds-backend.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 65536 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'


POSIX fallback
--------------

In some cases, libcufile implements its own internal POSIX fallback without `GdsBackend` being aware.
In others, an error such as `RuntimeError: cuFileHandleRegister failed (cuFile err=5030, cuda_err=0)` may be throwned.
Thus, backend can be configured to fallback to its own POSIX implementation when the usage of the libcufile APIs is not successful.

To force `GdsBackend` not use libcufile APIs for any reason, you can override its behavior via `extra_config`,
e.g:

.. code-block:: yaml

    LMCACHE_EXTRA_CONFIG='{"use_cufile": false}'

Note that under this mode it would still use CUDA APIs to map and do operations the pre-registered GPU memory.
