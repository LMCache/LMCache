Encoder cache
=============

The **Encoder Cache (EC)** stores the output of a multimodal model's
vision encoder, keyed by vLLM's per-input ``mm_hash``. When two
requests share an input (the same image, the same video), the second
request loads the encoder output from the cache and the vision tower
does not run.

vLLM exposes the encoder-cache extension point via the
``ECConnectorBase`` interface (vLLM v1 only). LMCache provides an
``LMCacheECConnector`` shim on the vLLM side and an ``ECCacheEngine``
on the LMCache side; together they back the encoder cache with any of
LMCache's storage backends (local CPU, local disk, remote, NIXL).

How to enable it
----------------

You need both:

1. A vLLM build with the LMCache EC connector entrypoint registered.
   This lives in ``vllm/distributed/ec_transfer/ec_connector/lmcache_connector.py``
   and is registered in ``factory.py`` under the name
   ``LMCacheECConnector``. (See vLLM PR #38668 for the upstream
   wiring.)
2. An ``LMCacheEngineConfig`` with at least one storage backend
   configured for EC. The standard LMCache config knobs apply; EC also
   accepts overrides prefixed with ``LMCACHE_EC_`` (env) or ``ec_``
   (YAML) so you can size EC storage separately from KV.

Server launch
~~~~~~~~~~~~~

Pass ``--ec-transfer-config`` to ``vllm serve``:

.. code-block:: bash

   vllm serve <model> \
       --ec-transfer-config '{
         "ec_connector": "LMCacheECConnector",
         "ec_role": "ec_both",
         "ec_connector_module_path": "vllm.distributed.ec_transfer.ec_connector.lmcache_connector"
       }'

``ec_role`` choices:

* ``ec_producer`` — saves encoder outputs to LMCache, never reads.
  Useful for a dedicated encoder/prefill instance.
* ``ec_consumer`` — only reads from LMCache. Useful for a decode
  instance fed by an upstream producer.
* ``ec_both`` — produces and consumes. Single-instance / development
  default.

LMCache config
~~~~~~~~~~~~~~

A minimal ``lmcache_ec.yaml``:

.. code-block:: yaml

   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 2          # GiB
   local_disk: "file:///var/lmcache/ec"
   max_local_disk_size: 16        # GiB

Set ``LMCACHE_CONFIG_FILE`` to point at the YAML before launching vLLM.

EC-specific overrides
~~~~~~~~~~~~~~~~~~~~~

To size the EC cache independently from the (separate) KV cache, use
the ``ec_`` / ``LMCACHE_EC_`` prefix:

* YAML: ``ec_max_local_disk_size: 64``
* env: ``LMCACHE_EC_MAX_LOCAL_DISK_SIZE=64``

Any standard LMCache config key works under the prefix; EC values
override the base config when an EC engine is built.

Verifying it's working
----------------------

Three independent signals confirm EC is hitting:

1. **vLLM metric.** ``loggers.py`` reports ``MM cache hit rate: X%``.
   For a request whose ``mm_hash`` was cached, ``has_cache_item`` hits.
2. **LMCache log line.** Cold (first-time) requests emit
   ``LMCache INFO: EC put: stored N bytes for mm_hash=H``. Warm
   requests emit no ``EC put``.
3. **On-disk file.** Under ``local_disk``, an entry of the form
   ``<model>@1@0@<chunk_hash>@<dtype>.pt`` appears after the first
   request. The ``@1@0@`` prefix reflects EC's deliberate sentinel
   ``world_size=1, worker_id=0`` so all tensor-parallel ranks share
   one entry; concurrent puts from multiple ranks are idempotent.

Design notes
------------

The full design rationale (cache key derivation, dtype decoupling
from KV quant, why EC and KV use separate ``StorageManager`` instances,
the dual-role connector class) lives in
:doc:`/non_kv_cache/encoder_cache/design`.

Benchmarks
----------

End-to-end measurement on a single H100 with Qwen2.5-VL-7B and a
720p / 60 MB video lives at :doc:`/non_kv_cache/encoder_cache/benchmark`.

.. toctree::
   :maxdepth: 1
   :hidden:

   benchmark
   design
