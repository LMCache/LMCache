musa-aiter with LMCache on MUSA
================================

This guide shows how to install ``musa-aiter`` and use its optional LMCache KV-transfer
adapter with vLLM on MUSA.

Prerequisites
-------------

Use a machine or container with:

- MUSA runtime and TorchMUSA installed.
- A Python environment that can import ``torch`` and ``torch.musa``.
- LMCache source available on ``PYTHONPATH`` or installed in the environment.
- vLLM installed with MUSA support.

Verify the MUSA runtime first:

.. code-block:: bash

   python - <<'PY'
   import torch
   print("torch", torch.__version__)
   print("has torch.musa", hasattr(torch, "musa"))
   print("musa available", torch.musa.is_available())
   print("device count", torch.musa.device_count())
   print("torch.version.musa", getattr(torch.version, "musa", None))
   PY

Step 1: Activate the target environment
---------------------------------------

Activate the same Python environment that will run vLLM and LMCache:

.. code-block:: bash

   source /path/to/your/venv/bin/activate
   python -V
   which python

Step 2: Install LMCache
-----------------------

For source-tree validation, add LMCache to ``PYTHONPATH``:

.. code-block:: bash

   export LMCACHE_SRC=/path/to/LMCache
   export PYTHONPATH=$LMCACHE_SRC:$PYTHONPATH

For a normal editable install, install LMCache from the repository root:

.. code-block:: bash

   cd /path/to/LMCache
   NO_NATIVE_EXT=1 pip install -e .

Use ``NO_NATIVE_EXT=1`` when you only need the Python path and do not want to rebuild
LMCache native extensions in the test environment.

Step 3: Install musa-aiter
--------------------------

Install from a local ``musa-aiter`` source checkout:

.. code-block:: bash

   cd /tmp/musa-aiter-latest
   pip uninstall -y musa-aiter musa_aiter
   pip install -v .

Verify the installed package and LMCache ABI entry point from outside the source
checkout so Python imports the installed package, not the current working tree:

.. code-block:: bash

   cd /tmp
   python - <<'PY'
   import importlib.metadata as md
   import musa_aiter
   print("musa_aiter version", md.version("musa-aiter"))
   print("musa_aiter file", musa_aiter.__file__)
   print("lmcache kv transfer ABI", musa_aiter.native_lmcache_kv_transfer_abi_version())
   PY

Expected result:

.. code-block:: text

   lmcache kv transfer ABI 1

Step 4: Enable LMCache's optional MUSA native transfer
------------------------------------------------------

LMCache does not require ``musa-aiter`` for correctness. The default torch fallback path
still works when native transfer is disabled. Enable the optional native path explicitly:

.. code-block:: bash

   export LMCACHE_MUSA_NATIVE_KV_TRANSFER=1

Disable it for a correctness baseline:

.. code-block:: bash

   export LMCACHE_MUSA_NATIVE_KV_TRANSFER=0

Stage 4 MUSA handle support is merged. Prefer the LMCache-driven handle path when
TorchMUSA provides the required memory and event IPC APIs:

.. code-block:: bash

   export LMCACHE_MUSA_HANDLE_TRANSFER=1
   export LMCACHE_MP_TRANSFER_MODE=lmcache_driven

This path lets the LMCache server access registered MUSA KV-cache tensors through
device handles and avoids worker-side gather/scatter copies. The ``auto`` mode still
selects ``engine_driven`` for MUSA, so request ``lmcache_driven`` explicitly to use
the handle path. If the required TorchMUSA IPC APIs are unavailable, use the
engine-driven compatibility path instead:

.. code-block:: bash

   unset LMCACHE_MUSA_HANDLE_TRANSFER
   export LMCACHE_MP_TRANSFER_MODE=engine_driven

Step 5: Create an LMCache config
--------------------------------

Example local CPU offload config:

.. code-block:: bash

   cat > /tmp/lmcache_musa_e2e.yaml <<'YAML'
   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 2
   YAML

   export LMCACHE_CONFIG_FILE=/tmp/lmcache_musa_e2e.yaml

Step 6: Run vLLM with LMCache
-----------------------------

Example vLLM server command:

.. code-block:: bash

   export PYTHONPATH=$LMCACHE_SRC:$PYTHONPATH
   export LMCACHE_CONFIG_FILE=/tmp/lmcache_musa_e2e.yaml
   export LMCACHE_MUSA_NATIVE_KV_TRANSFER=1
   export LMCACHE_MUSA_HANDLE_TRANSFER=1
   export LMCACHE_MP_TRANSFER_MODE=lmcache_driven
   export VLLM_USE_V1=1
   export MUSA_VISIBLE_DEVICES=7

   python -m vllm.entrypoints.cli.main serve /path/to/model \
     --served-model-name qwen3-8b \
     --host 127.0.0.1 \
     --port 18082 \
     --trust-remote-code \
     --max-model-len 2048 \
     --gpu-memory-utilization 0.35 \
     --enforce-eager \
     --no-enable-prefix-caching \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Step 7: Validate cache store and hit
------------------------------------

Send two repeated long-prompt requests. The first request should store KV chunks; the
second request should hit and retrieve them from LMCache.

.. code-block:: bash

   python - <<'PY'
   import json
   import urllib.request

   url = "http://127.0.0.1:18082/v1/completions"
   prompt = "Moore Threads MUSA LMCache vLLM cache reuse validation. " * 120
   payload = {"model": "qwen3-8b", "prompt": prompt, "max_tokens": 8, "temperature": 0}

   for i in range(2):
       req = urllib.request.Request(
           url,
           data=json.dumps(payload).encode(),
           headers={"Content-Type": "application/json"},
       )
       with urllib.request.urlopen(req, timeout=240) as resp:
           body = json.loads(resp.read().decode())
       print("request", i + 1, "ok", body["choices"][0]["text"][:80])
   PY

Look for LMCache logs like:

.. code-block:: text

   Stored ... tokens
   LMCache hit tokens: ...
   Retrieved ... required tokens

Troubleshooting
---------------

- If ``musa_aiter`` import fails, reinstall it in the exact environment that launches
  vLLM.
- If ``native_lmcache_kv_transfer_abi_version`` is missing, the installed package does
  not provide the LMCache native adapter ABI expected by LMCache.
- If native transfer fails but ``LMCACHE_MUSA_NATIVE_KV_TRANSFER=0`` passes, the issue is
  isolated to ``musa-aiter`` or the native input layout/contiguity requirements.
- If vLLM startup fails with low free memory, lower ``--gpu-memory-utilization`` or choose
  a less busy device with ``MUSA_VISIBLE_DEVICES``.
