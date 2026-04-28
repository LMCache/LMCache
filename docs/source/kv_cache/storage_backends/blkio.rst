Blkio (libblkio Block Device I/O)
==================================

.. _blkio-overview:

Overview
--------

The blkio backend provides high-performance block device I/O for KV cache storage
using the `libblkio <https://gitlab.com/libblkio/libblkio>`_ library. libblkio
offers a unified interface to various block device backends; this connector uses
the **io_uring** driver for asynchronous, kernel-bypassing I/O directly to NVMe
drives or other block devices.

Key characteristics:

- **Per-worker io_uring instances**: Each C++ worker thread creates its own ``io_uring`` via libblkio -- no shared-queue contention, true parallelism.
- **Map/IO/Unmap per operation**: Each read or write maps the DRAM buffer, submits I/O, waits for completion, then unmaps. This matches the NIXL ``registerBlkioBuf`` pattern.
- **O_DIRECT support**: Bypasses the page cache for predictable latency and no double-buffering overhead (enabled by default).
- **Dual-mode support**: Works in both non-MP mode (via ``BlkioClient``) and MP mode (via ``NativeConnectorL2Adapter`` as an L2 adapter).

The native C++ source lives in ``csrc/storage_backends/blkio/``. See
:doc:`Adding Native Connectors <../../developer_guide/extending_lmcache/native_connectors>`
for the full architecture.

.. code-block:: text

   Non-MP mode:
     CacheEngine -> RemoteBackend -> BlkioClient -> LMCacheBlkioClient (C++)
                                       (asyncio)       |
                                                  BlkioConnector
                                                    +- worker 0: blkio(io_uring) -> block device
                                                    +- worker 1: blkio(io_uring) -> block device
                                                    +- worker N: blkio(io_uring) -> block device

   MP mode:
     StoreController / PrefetchController
           |
     NativeConnectorL2Adapter (Python bridge)
       +- 3 eventfds (store, lookup, load)
       +- completion demux thread
       +- client-side lock tracking
           |
     LMCacheBlkioClient (C++)
       +- BlkioConnector -> per-worker io_uring instances


Prerequisites
-------------

- LMCache installed from source (``pip install -e . --no-build-isolation``) to compile the C++ extension
- ``libblkio`` development headers and library installed (see `Installing libblkio`_ below)
- A block device for storage (NVMe drive, loopback device, etc.)
- Root access if using ``O_DIRECT`` with a real block device


Installing libblkio
--------------------

**From package manager:**

.. code-block:: bash

    # Ubuntu/Debian
    sudo apt-get install libblkio-dev

    # RHEL/CentOS/Fedora
    sudo dnf install libblkio-devel

**From source:**

.. code-block:: bash

    git clone https://gitlab.com/libblkio/libblkio.git
    cd libblkio
    meson setup build
    ninja -C build
    sudo ninja -C build install
    sudo ldconfig

**Verify the install:**

.. code-block:: bash

    pkg-config --cflags --libs blkio
    # Expected: -I/usr/local/include -lblkio  (paths may vary)

If libblkio is installed in a non-standard location, set ``CFLAGS`` and ``LDFLAGS``
before building LMCache:

.. code-block:: bash

    CFLAGS="-I/path/to/include" LDFLAGS="-L/path/to/lib" \
        pip install -e . --no-build-isolation


Configuration
-------------

.. _blkio-config-params:

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Parameter
     - Type
     - Default
     - Description
   * - ``device_path``
     - str
     - (required)
     - Path to the block device (e.g. ``/dev/nvme0n1``, ``/dev/loop0``)
   * - ``num_workers``
     - int
     - 4
     - Number of C++ worker threads. Each gets its own ``io_uring`` instance
   * - ``direct_io``
     - bool
     - ``true``
     - Enable ``O_DIRECT`` to bypass the page cache


Non-MP Mode (Single Process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In non-MP mode, the blkio connector is used directly via the ``BlkioClient``
asyncio wrapper.

.. code-block:: python

    from lmcache.v1.storage_backend.native_clients.blkio_client import BlkioClient

    client = BlkioClient(
        device_path="/dev/nvme0n1",   # block device path
        num_workers=4,                 # io_uring instances (default 4)
        direct_io=True,                # O_DIRECT (default True)
    )


MP Mode (Multiprocess)
~~~~~~~~~~~~~~~~~~~~~~~

In MP mode, LMCache runs as a separate server process. The blkio connector
serves as an L2 adapter:

**Start the LMCache MP server:**

.. code-block:: bash

    python -m lmcache.v1.multiprocess.server \
        --l1-size-gb 10 \
        --eviction-policy LRU \
        --chunk-size 256 \
        --l2-adapter '{"type": "blkio", "device_path": "/dev/nvme0n1", "num_workers": 4, "direct_io": true}' \
        --port 6555

**Start vLLM with LMCache MP connector:**

.. code-block:: bash

    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --kv-transfer-config '{
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": 6555
            }
        }' \
        --no-enable-prefix-caching \
        --port 8000

L2 Adapter JSON Fields
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Field
     - Type
     - Default
     - Description
   * - ``type``
     - str
     - (required)
     - Must be ``"blkio"``
   * - ``device_path``
     - str
     - (required)
     - Path to the block device
   * - ``num_workers``
     - int
     - 4
     - C++ worker threads for parallel I/O
   * - ``direct_io``
     - bool
     - ``true``
     - Bypass page cache via ``O_DIRECT``


Testing
-------

Unit Tests
~~~~~~~~~~

These test ``BlkioL2AdapterConfig`` parsing, validation, and registry wiring.
No block device or C++ extension required:

.. code-block:: bash

    pytest -xvs tests/v1/distributed/test_blkio_l2_adapter.py

Integration Tests
~~~~~~~~~~~~~~~~~~

These exercise the full C++ -> libblkio -> io_uring -> kernel path.
The test fixture auto-provisions a test device using this priority:

1. **``LMCACHE_BLKIO_TEST_DEVICE`` env var** -- use a pre-existing block device
2. **Auto-created loopback device** -- requires root (``losetup``)
3. **Sparse temp file** -- always available, but no ``O_DIRECT`` support

.. code-block:: bash

    # Automatic device provisioning (temp file fallback, no root needed)
    pytest -xvs tests/v1/storage_backend/test_blkio_connector.py

    # With a real block device (full O_DIRECT, best for validation)
    sudo LMCACHE_BLKIO_TEST_DEVICE=/dev/loop0 \
        pytest -xvs tests/v1/storage_backend/test_blkio_connector.py

Setting Up a Loopback Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you don't have a spare NVMe partition, create a loopback device for testing:

.. code-block:: bash

    # Create a 64 MB backing file and attach as a loop device
    sudo dd if=/dev/zero of=/tmp/blkio_test.img bs=1M count=64
    LOOP_DEV=$(sudo losetup -f --show /tmp/blkio_test.img)
    echo "Using: $LOOP_DEV"

    # Run integration tests with O_DIRECT
    sudo LMCACHE_BLKIO_TEST_DEVICE=$LOOP_DEV \
        pytest -xvs tests/v1/storage_backend/test_blkio_connector.py

    # Cleanup
    sudo losetup -d $LOOP_DEV
    rm /tmp/blkio_test.img


Docker Considerations
~~~~~~~~~~~~~~~~~~~~~

Docker's default seccomp profile blocks ``io_uring`` syscalls. Add them to run
the blkio connector inside a container:

.. code-block:: bash

    # Download default seccomp profile
    wget -O seccomp.json \
        https://raw.githubusercontent.com/moby/moby/master/profiles/seccomp/default.json

    # Add to the "syscalls"."names" array in seccomp.json:
    #   "io_uring_setup", "io_uring_enter", "io_uring_register"

    # Run container with the updated profile and device access
    docker run --security-opt seccomp=seccomp.json \
        --device /dev/loop0:/dev/loop0 \
        -it <image>

What the Integration Tests Verify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Test
     - Description
   * - ``test_construct_and_close``
     - Connector can be created and cleanly shut down
   * - ``test_event_fd_is_valid``
     - ``event_fd()`` returns a valid file descriptor
   * - ``test_write_read_verify``
     - Write 4 KB of ``0xAB``, read back, verify contents match
   * - ``test_write_read_distinct_patterns``
     - Write ``0x55``, overwrite buffer with ``0xAA``, read back confirms ``0x55`` on device
   * - ``test_batch_write_read``
     - Batch write/read 4 blocks with different fill patterns
   * - ``test_multiple_workers``
     - Verified with 1, 2, and 4 worker threads
   * - ``test_sync_set_get_roundtrip``
     - Python ``BlkioClient`` sync set/get roundtrip


Current Limitations
-------------------

1. **io_uring only** -- libblkio supports ``virtio-blk-vhost-user`` and ``virtio-blk-vhost-vdpa`` drivers, but only ``io_uring`` is currently wired up.
2. **No native existence tracking** -- ``do_single_exists`` always returns ``false``. The Python layer must track which offsets have been written.
3. **Single device** -- each connector instance opens one block device. For multi-device setups, create multiple connectors.
4. **Offset-based addressing** -- keys encode a hex byte offset in the last ``@``-delimited field. The Python layer is responsible for slot allocation and metadata tracking.


Additional Resources
--------------------

- C++ source: ``csrc/storage_backends/blkio/``
- Detailed README: ``csrc/storage_backends/blkio/README.md``
- Python client: ``lmcache/v1/storage_backend/native_clients/blkio_client.py``
- Native connector architecture: :doc:`Adding Native Connectors <../../developer_guide/extending_lmcache/native_connectors>`
