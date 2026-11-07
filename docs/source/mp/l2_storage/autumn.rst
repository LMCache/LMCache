autumn-rs
=========

An L2 adapter backed by an `autumn-rs <https://github.com/journeymidnight/autumn-rs>`_
cluster: a distributed storage engine whose replicated, ordered KV layer holds
the chunks. The cluster is shared, durable and large, so it serves as a warm
tier that survives restarts and is shared across LMCache instances. The client
reads and writes straight into the L1 buffers with the GIL released; there is
no C++ extension to build.

**Install the client:**

.. code-block:: bash

    pip install autumn-rs

The published wheel serves CPython >= 3.11 on Linux (x86_64, aarch64) and
uses TCP. RDMA (UCX) needs a client built from source with ``--features ucx``.

**Prepare the cluster.** Keys are stored under a namespace, ``kvc`` by
default, which must exist before the adapter starts. Create it cut into 16
partitions so chunks spread over all of them from the start:

.. code-block:: bash

    autumn-op namespace-create --name kvc --parts 16

**Required fields:**

- ``manager``: comma-separated ``ip:port`` list of autumn managers. The
  client dials addresses, so resolve host names first.

**Optional fields:**

- ``namespace`` (string, default ``"kvc"``): autumn namespace for the keys.
- ``credential_file`` (string, default ``""``): an autumn principal
  credential file, as ``autumn-op principal-create`` writes it. Required when
  authorization protects the namespace.
- ``num_workers`` (int, default ``4``): batches queued or in flight (> 0).
  The client runs gets and puts one batch at a time.
- ``transport`` (string, default ``"tcp"``): ``"tcp"`` or ``"ucx"``.
- ``direct`` (bool, default ``false``): read large values straight from the
  storage nodes instead of through the partition servers. Needs the client to
  reach the storage nodes.
- ``ttl_secs`` (int, default ``0``): expiry of each stored chunk in seconds;
  ``0`` keeps chunks until they are evicted.
- ``max_capacity_gb`` (float, default ``0``): max L2 capacity in GB for usage
  tracking / aggregate eviction. ``0`` disables tracking.

Keys are ``{namespace}/{h}/lmcache/{storage key}``, where ``h`` is the chunk
hash's first hex digit, the digit the ``kvc`` namespace is cut on when it is
created with ``--parts 16``.

**Configuration examples:**

.. code-block:: bash

    # Basic
    --l2-adapter '{"type": "autumn", "manager": "10.0.0.1:9001"}'

    # Authorization on, chunks expire after a day, a capacity cap
    --l2-adapter '{"type": "autumn", "manager": "10.0.0.1:9001,10.0.0.2:9001", "credential_file": "/etc/autumn/kvc.cred", "ttl_secs": 86400, "max_capacity_gb": 500}'

**Partition-server memory tier.** Partition servers started with
``--mem-tier-bytes <budget> --mem-tier-prefix kvc/`` hold the chunks in DRAM
and serve reads from memory; a partition-server restart loses what the tier
held, which a cache can afford.
