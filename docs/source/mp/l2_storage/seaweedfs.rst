SeaweedFS
=========

SeaweedFS can provide a persistent, distributed L2 for LMCache multiprocess
mode. The integration is distributed separately as the ``seaweedkv-native``
wheel and is loaded through LMCache's ``native_plugin`` adapter. It does not
run a sidecar in the payload path.

Architecture
------------

The connector uses SeaweedFS's existing control-plane and data-plane split:

* the Filer stores key-to-FID metadata in its configured persistent store;
* the Master provides Volume placement and location information;
* Volume servers store normal Needle data and indexes;
* the connector performs metadata lookup through the Filer and transfers
  payload directly to or from Volume servers over HTTP or RDMA.

S3 may expose the same SeaweedFS namespace for compatibility, but it is not in
the optimized LMCache payload path.

Prerequisites
-------------

* LMCache multiprocess mode with ``native_plugin`` support;
* a SeaweedFS Master, Filer, and one or more Volume servers;
* the ``seaweedkv-native`` wheel matching the host's CPython ABI and CPU
  architecture;
* for RC or DC, a supported RDMA device, ``rdma-core``, sufficient memlock,
  and matching connector/Volume endpoint configuration.

The connector wheel is not built or published by LMCache. Obtain a qualified
release from SeaweedFS or build it against the exact LMCache source revision
used by the runtime.

Installation
------------

Install the wheel in the same environment as the LMCache MP process:

.. code-block:: bash

   python3.12 -m venv /opt/lmcache-venv
   . /opt/lmcache-venv/bin/activate
   pip install lmcache
   pip install ./seaweedkv_native-0.1.0-cp312-cp312-linux_x86_64.whl
   python -c 'from seaweedkv_native import SeaweedKVNativeConnector; print("ok")'

HTTP Configuration
------------------

Pass the adapter JSON to ``lmcache server`` with ``--l2-adapter``. The Filer
gRPC endpoint is the only SeaweedFS address required by the connector; Volume
locations are resolved from metadata.

.. code-block:: bash

   lmcache server \
     --l1-size-gb 60 \
     --eviction-policy LRU \
     --l2-adapter '{
       "type": "native_plugin",
       "module_path": "seaweedkv_native",
       "class_name": "SeaweedKVNativeConnector",
       "adapter_params": {
         "filer_grpc": "filer.seaweedfs.svc:18888",
         "backing": "loader-http",
         "num_workers": 8,
         "capacity_bytes": 8589934592,
         "ttl_millis": 3600000,
         "hot_engine": "slab"
       },
       "max_capacity_gb": 8
     }'

``capacity_bytes`` and ``ttl_millis`` configure the connector's optional
process-local hot cache. They do not limit the shared SeaweedFS namespace.

RDMA Configuration
------------------

Use ``"backing": "loader-rdma"`` only after the Volume and connector have
matching RC or DC listeners. RDMA endpoint, device, port, GID, pipe count, and
maximum value size are deployment settings supplied by the SeaweedFS release.
They are deliberately not inferred by LMCache.

Required-RDMA deployments fail closed when the selected endpoint or registered
memory is unavailable. They do not silently report HTTP fallback results as
RDMA performance. HTTP remains a separate deployment profile and diagnostic
path.

Operations and Lifecycle
------------------------

The native connector implements asynchronous batch ``store``, ``load``,
``exists``, and ``delete`` operations. LMCache keys are stored as normal Filer
Entries whose chunks reference Volume FIDs. Filer and Volume restart recovery
therefore use normal SeaweedFS persistence rather than a second connector
metadata database.

``delete`` makes a key logically unavailable through the Filer. Needle bytes
are reclaimed later by native Volume vacuum. Operators can also configure
Filer Entry TTL and an opt-in SeaweedFS cache-Volume rotation policy for
physical capacity control.

LMCache currently treats native connectors as not supporting its generic L2
eviction controller. Capacity policy for this backend must therefore be
configured on the SeaweedFS side.

Validation
----------

Run LMCache's L2 benchmark with byte verification before admitting traffic:

.. code-block:: bash

   export L2_ADAPTER_JSON='<the native_plugin JSON above>'
   lmcache bench l2 \
     --num-keys 32 --in-flight 2 --data-size-kb 4096 \
     --l1-align-bytes 4096 --rounds 3 --warmup-rounds 1 \
     --lookup-max-hit-rate 1.0 --no-skip-verify

A production qualification should additionally restart Master, Filer, Volume,
and the connector process; verify delete and TTL behavior; and confirm that
required-RDMA counters account for all payload bytes with no fallback.

For server installation, release artifacts, and the current support boundary,
see the `SeaweedFS LMCache connector documentation
<https://github.com/seaweedfs/seaweed-mono/tree/main/enterprise/python/seaweedkv-native>`_.
