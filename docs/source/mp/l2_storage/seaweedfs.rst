SeaweedFS
=========

SeaweedFS provides a shared, persistent L2 backend for LMCache multiprocess
mode. The integration is distributed as the external ``seaweedkv-native``
wheel and loaded through LMCache's ``native_plugin`` adapter. The connector
runs in the LMCache process; it is not a payload sidecar.

Architecture
------------

The connector follows SeaweedFS's normal control-plane and data-plane split:

* the Filer stores each LMCache key as a normal Entry in its configured
  persistent metadata store;
* the Master supplies Volume placement and location information;
* Volume servers persist normal Needle data and indexes;
* the connector resolves metadata through the Filer and transfers payloads
  directly to or from Volume servers over HTTP, RDMA RC, or RDMA DC.

The SeaweedFS S3 API can expose the same namespace for compatibility, but S3
is not in the optimized connector payload path.

Prerequisites
-------------

* LMCache multiprocess mode with ``native_plugin`` support;
* a SeaweedFS Master, Filer, and one or more Volume servers;
* a ``seaweedkv-native`` wheel matching the Linux CPython ABI and CPU
  architecture of the LMCache runtime;
* for RC or DC, supported RDMA hardware, ``rdma-core``, sufficient memlock,
  and matching connector and Volume endpoint configuration.

The wheel is not built or published by LMCache. Obtain a qualified artifact
from SeaweedFS or build it against the exact LMCache source revision used by
the runtime.

Installation
------------

Install the wheel in the same environment as the LMCache process:

.. code-block:: bash

   pip install ./seaweedkv_native-0.1.0-cp312-cp312-linux_x86_64.whl
   python -c 'from seaweedkv_native import SeaweedKVNativeConnector; print("ok")'

HTTP Configuration
------------------

The Filer gRPC endpoint is the only SeaweedFS address configured in LMCache.
The connector resolves Volume locations from Filer metadata.

.. code-block:: bash

   lmcache server \
     --host 0.0.0.0 --port 5555 \
     --http-host 0.0.0.0 --http-port 8080 \
     --l1-size-gb 60 --max-workers 8 \
     --eviction-policy LRU \
     --l2-adapter '{
       "type": "native_plugin",
       "module_path": "seaweedkv_native",
       "class_name": "SeaweedKVNativeConnector",
       "adapter_params": {
         "filer_grpc": "filer.seaweedfs.svc:18888",
         "backing": "loader-http",
         "num_workers": 8
       },
       "max_capacity_gb": 1024
     }'

``max_capacity_gb`` is the aggregate capacity advertised to LMCache for usage
accounting. It does not allocate connector-local memory. LMCache L1 remains
the memory tier.

RDMA Configuration
------------------

Use ``"backing": "loader-rdma"`` only after Volume servers and LMCache hosts
have matching RC or DC configuration. Device, port, GID, NUMA placement, MTU,
pipe count, and maximum value size are deployment settings supplied by the
SeaweedFS release. Required-RDMA configurations fail closed rather than
silently reporting HTTP fallback as RDMA traffic.

The current native RC/DC listener is intended for a trusted storage network.
Use HTTP when the deployment requires Filer read-JWT enforcement at the
payload endpoint.

Lifecycle and Recovery
----------------------

The connector implements asynchronous batch store, lookup, load, and delete.
Committed keys are normal Filer Entries whose chunks reference Volume FIDs,
so Filer and Volume restarts use normal SeaweedFS persistence rather than a
second connector metadata database.

Delete makes a key logically unavailable and writes Needle tombstones. Native
Volume vacuum later reclaims those bytes. Filer Entry TTL can provide
time-based logical expiry. SeaweedFS also offers an opt-in, collection-scoped
pressure policy for whole-Volume cache eviction; the initial policy accepts
only non-replicated cache Volumes and does not migrate hot survivors.

Validation
----------

Before admitting production traffic, run the LMCache L2 benchmark with byte
verification and the intended value-size distribution:

.. code-block:: bash

   lmcache bench l2 \
     --num-keys 32 --in-flight 2 --data-size-kb 4096 \
     --l1-align-bytes 4096 --rounds 3 --warmup-rounds 1 \
     --lookup-max-hit-rate 1.0 --no-skip-verify

A storage qualification should also restart Master, Filer, Volume, and the
LMCache process; verify delete and TTL behavior; and confirm that required
RDMA counters account for all payload bytes with no fallback.

See the `SeaweedFS LMCache release and installation guide
<https://github.com/seaweedfs/seaweed-mono/blob/main/docs/LMCACHE-SEAWEEDFS-RELEASE-INSTALLATION.md>`_
for server configuration, packaged artifacts, and the current support
boundary.
