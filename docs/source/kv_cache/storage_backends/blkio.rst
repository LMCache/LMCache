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

The ``io_engine="libblkio"`` option for ``RawBlockDevice`` in the
``lmcache_rust_raw_block_io`` crate provides the libblkio integration, enabled
via the ``blkio`` cargo feature flag. It plugs into the ``RustRawBlockBackend``
storage plugin as an alternative I/O engine alongside the default posix
pread/pwrite and io_uring paths.

Key characteristics:

- **io_uring via libblkio**: Each I/O operation goes through libblkio's io_uring driver -- no manual ``io_uring`` setup required.
- **Map/IO/Unmap per operation**: Each read or write maps the DRAM buffer, submits I/O, waits for completion, then unmaps. This matches the NIXL ``registerBlkioBuf`` pattern.
- **O_DIRECT support**: Bypasses the page cache for predictable latency and no double-buffering overhead (enabled by default).
- **Bounce-buffer alignment**: Hybrid aligned-prefix + bounce-tail paths handle unaligned buffers transparently for O_DIRECT.

.. code-block:: text

   CacheEngine -> RustRawBlockBackend (storage plugin)
                      |
                  RawBlockDevice(io_engine="libblkio") (Rust, PyO3)
                      |
                  libblkio FFI -> io_uring -> block device


Prerequisites
-------------

- ``libblkio`` development headers and library installed (see `Installing libblkio`_ below)
- The ``lmcache_rust_raw_block_io`` Rust extension built with the ``blkio`` feature
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


Building
--------

Build the Rust extension with the ``blkio`` feature:

.. code-block:: bash

    cd rust/raw_block
    pip install maturin
    maturin develop --release --features blkio

Without ``--features blkio``, only ``posix`` and ``io_uring`` engines are
available; ``io_engine="libblkio"`` will raise ``ValueError``.


I/O Engine Comparison
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Engine
     - Best for
   * - ``posix`` (default)
     - Simple synchronous I/O; widest compatibility.
   * - ``io_uring``
     - High-throughput NVMe I/O with batch submission and
       fixed-buffer zero-copy via the Rust io_uring worker thread.
   * - ``libblkio``
     - Environments that already depend on libblkio (e.g. NIXL);
       single-queue synchronous I/O via libblkio's io_uring driver.


Configuration
-------------

Set ``rust_raw_block.io_engine`` to ``"libblkio"`` in ``extra_config``:

.. code-block:: yaml

    extra_config:
      rust_raw_block.device_path: "/dev/nvme0n1"
      rust_raw_block.io_engine: "libblkio"
      rust_raw_block.use_odirect: true
      rust_raw_block.block_align: 4096
      # rust_raw_block.blkio_driver: "io_uring"   # optional, defaults to "io_uring"

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``rust_raw_block.device_path``
     - str
     - (required)
     - Path to the block device (e.g. ``/dev/nvme0n1``, ``/dev/loop0``)
   * - ``rust_raw_block.io_engine``
     - str
     - ``"posix"``
     - I/O engine: ``"posix"``, ``"io_uring"``, or ``"libblkio"``
   * - ``rust_raw_block.blkio_driver``
     - str
     - ``"io_uring"``
     - libblkio driver to use (only relevant when ``io_engine="libblkio"``)
   * - ``rust_raw_block.use_odirect``
     - bool
     - ``false``
     - Enable ``O_DIRECT`` to bypass the page cache
   * - ``rust_raw_block.block_align``
     - int
     - 4096
     - Alignment for O_DIRECT I/O (must match device sector size)


Direct Usage
------------

.. code-block:: python

    from lmcache_rust_raw_block_io import RawBlockDevice

    dev = RawBlockDevice(
        "/dev/nvme0n1",
        writable=True,
        use_odirect=True,
        alignment=4096,
        io_engine="libblkio",
    )
    print(dev.size_bytes())

    data = bytearray(4096)
    dev.pwrite_from_buffer(offset=0, data=data, payload_len=100, total_len=4096)

    out = bytearray(4096)
    dev.pread_into(offset=0, out=out, payload_len=100, total_len=4096)
    dev.close()


Testing
-------

.. code-block:: bash

    # Smoke + integration tests (temp file, no device needed)
    pytest -xvs tests/v1/storage_backend/test_blkio_block_device.py

    # With O_DIRECT on a real block device or loopback
    LMCACHE_BLKIO_TEST_DEVICE=/dev/loop0 \
        pytest -xvs tests/v1/storage_backend/test_blkio_block_device.py

.. list-table::
   :header-rows: 1
   :widths: 40 10 50

   * - Test class
     - Count
     - Coverage
   * - ``TestBlkioBlockDeviceSmoke``
     - 9
     - Open/close, read/write roundtrip, padding, error handling
   * - ``TestBlkioRawBlockBackendIntegration``
     - 4
     - Put/get, batched get, eviction, checkpoint recovery
   * - ``TestBlkioBlockDeviceODirect``
     - 4
     - O_DIRECT roundtrip, large buffer, padding, multi-offset
   * - ``TestBlkioRawBlockBackendODirect``
     - 1
     - Full backend put/get with O_DIRECT

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
        pytest -xvs tests/v1/storage_backend/test_blkio_block_device.py

    # Cleanup
    sudo losetup -d $LOOP_DEV
    rm /tmp/blkio_test.img


Docker Considerations
~~~~~~~~~~~~~~~~~~~~~

Docker's default seccomp profile blocks ``io_uring`` syscalls. Add them to run
the blkio backend inside a container:

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


Current Limitations
-------------------

1. **io_uring only** -- libblkio supports multiple drivers, but only ``io_uring`` is currently the default ``blkio_driver``. Other drivers can be selected via the ``blkio_driver`` parameter.
2. **Single queue** -- the libblkio engine uses a single libblkio queue. For higher parallelism, use ``io_engine="io_uring"`` which provides batch submission via the Rust worker thread.
3. **Single device** -- each backend instance opens one block device. For multi-device setups, create multiple backends.
4. **No async batch methods** -- when using ``io_engine="libblkio"``, the async io_uring batch methods (``batched_write``, ``batched_read``, ``wait_iouring``, ``register_fixed_buffers``) are not available.


Additional Resources
--------------------

- Rust crate source: ``rust/raw_block/``
- Crate README: ``rust/raw_block/README.md``
- Python backend: ``lmcache/v1/storage_backend/plugins/rust_raw_block_backend.py``
