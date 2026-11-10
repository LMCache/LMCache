.. SPDX-License-Identifier: Apache-2.0

Rebellions NPU
==============

The ``rbln`` backend runs LMCache on Rebellions NPUs, with
`vLLM-RBLN <https://github.com/RBLN-SW/vllm-rbln>`_ as the serving engine. It
implements the community ``DeviceSpec`` and ``DeviceOps`` interfaces for
multiprocess (MP) **engine-driven** transfers: on store, the vLLM-RBLN worker
gathers KV blocks from its paged cache on the NPU into the LMCache server's
host memory, and on retrieve it scatters them back. KV moves with
``torch.rbln`` tensor operations.

Prerequisites
-------------

Install the RBLN driver and the matching runtime stack (``torch-rbln`` and
``rebel-compiler``) by following the
`RBLN SDK documentation <https://docs.rbln.ai>`_, then install ``vllm-rbln``.
Do not let the LMCache installation replace the vendor's PyTorch.

Each LMCache release publishes an RBLN wheel, built against the ``torch`` that
``torch-rbln`` pins, to a ``v<version>-rbln`` GitHub Release. Install it into
that environment (see the *Rebellions RBLN* tab of :ref:`installation_guide`):

.. code-block:: bash

   VERSION=X.Y.Z  # target LMCache release
   pip install lmcache==${VERSION}+rbln \
       --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v${VERSION}-rbln

To build from source instead, build LMCache's common C++ extensions against
that PyTorch installation. RBLN adds no native extension; the build selects
the RBLN profile when ``torch_rbln`` is installed, and ``BUILD_WITH_RBLN=1``
forces it:

.. code-block:: bash

   BUILD_WITH_RBLN=1 pip install -e . --no-build-isolation

Use an environment with the repository's build dependencies installed.

Running vLLM-RBLN with LMCache
------------------------------

As on GPUs, start an LMCache server and point vLLM-RBLN at it with
``LMCacheMPConnector``. With ``N`` as the vLLM block size:

.. code-block:: bash

   lmcache server --port 6555 --l1-size-gb 32 --eviction-policy LRU \
       --chunk-size N --supported-transfer-mode engine_driven

   vllm serve <model> \
       --model-impl vllm \
       --block-size N \
       --kv-transfer-config '{
           "kv_connector": "LMCacheMPConnector",
           "kv_role": "kv_both",
           "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
           "kv_connector_extra_config": {
               "lmcache.mp.host": "tcp://127.0.0.1",
               "lmcache.mp.port": 6555
           }
       }'

Three settings are specific to RBLN:

* ``--supported-transfer-mode engine_driven`` on the server. The server
  defaults to ``lmcache_driven``, which does not register the engine-driven
  handlers the RBLN worker calls. ``auto`` also works, as it enables both
  paths.
* ``--model-impl vllm``. KV connectors work only with vLLM's model
  implementation. With the default ``auto``, vLLM-RBLN can choose a model
  path that never hands its KV caches to a connector.
* The LMCache chunk size must be a positive multiple of the vLLM block size,
  because the backend moves whole blocks, so set ``--chunk-size``
  explicitly. Setting the block size equal to the chunk size is recommended.

LMCache detects ``torch.rbln`` and selects ``rbln`` automatically when an NPU
can be opened. To pin the backend, set:

.. code-block:: bash

   export LMCACHE_DEVICE_BACKEND=rbln

Pinning is recommended. Without it, a process that cannot open an NPU (for
example, because another process holds the device) falls back to the CPU stub
without an error. With it, startup fails instead.

Supported KV cache layouts
--------------------------

The backend accepts the per-layer KV cache layouts that vLLM-RBLN allocates:

* Blocks-first HND: ``[num_blocks, 2, num_kv_heads, 1, block_size, head_size]``.
* K/V-first HND: ``[2, num_blocks, num_kv_heads, 1, block_size, head_size]``.
* MLA: ``[num_blocks, block_size, head_size]``.

Any other layout raises ``ValueError``.

Capability limits
-----------------

LMCache-driven MP transfer is not supported. On the worker, setting
``LMCACHE_MP_TRANSFER_MODE=lmcache_driven`` fails at validation, and ``auto``
(the default) resolves to ``engine_driven``. The server does not check the
device: started with ``--supported-transfer-mode lmcache_driven`` (its
default), it runs but cannot serve an RBLN worker.

Validation
----------

Run the backend tests on a host with an RBLN NPU:

.. code-block:: bash

   python -m pytest -q -rs tests/v1/platform/devices/rbln

Tests that need an NPU skip when none can be opened; skipped cases are not
hardware validation. The ``rbln`` Buildkite pipeline
(``.buildkite/k3_tests/rbln``) runs these tests and an end-to-end vLLM-RBLN
check on RBLN hardware.
