.. SPDX-License-Identifier: Apache-2.0

Adding a New Device Backend
===========================

This guide explains how to add a **new non-CUDA accelerator** to LMCache in
**Multiprocess (MP) engine-driven mode**.

The integration is intentionally small and self-contained:

- Add one ``DeviceSpec`` under ``lmcache/v1/platform/<device>/__init__.py``
- Add one ops backend module ``lmcache/v1/platform/<device>/ops.py``
- Add a native fast path adapter ``native_kv_transfer.py``

You do **not** need to modify global dispatch code.

Scope
-----

This guide explains how to add a **new non-CUDA accelerator** to LMCache in
**Multiprocess (MP) engine-driven mode**.

The integration is intentionally small and self-contained:

- Add one ``DeviceSpec`` under ``lmcache/v1/platform/<device>/__init__.py``
- Add one ops backend module ``lmcache/v1/platform/<device>/ops.py``
- Add a native fast path adapter ``native_kv_transfer.py``

You do **not** need to modify global dispatch code.

Architecture Rules
------------------

Current backend selection is registry-driven:

- Device detection: ``lmcache/v1/platform/__init__.py::_detect_device()``
- Backend loading: ``lmcache/v1/platform/__init__.py::get_backend(device_type)``
- Device descriptor base class: ``lmcache/v1/platform/base_device_spec.py::DeviceSpec``

``DeviceSpec`` subclasses are discovered automatically under
``lmcache.v1.platform``.

What not to do
~~~~~~~~~~~~~~

- Do not edit ``lmcache/__init__.py`` for device detection/backend selection.
- Do not add any ``register_availability(...)`` call (no such API exists).
- Do not treat ``lmcache/v1/platform/_registry.py`` as device-detection
  registry; it is for KV IPC wrapper factories (``register_kv_wrapper`` /
  ``get_kv_wrapper_factory``).

Prerequisites
-------------

Your PyTorch backend should support:

- ``torch.<device>.is_available()``
- ``torch.<device>.device_count()``
- ``torch.<device>.set_device()`` / ``current_device()`` / ``synchronize()``
- Tensor movement between host and device (``.cpu()``, ``.to("cpu")``,
  ``.to(<device>)``)

If these are available, engine-driven transfer can run with Python fallback
before native kernels exist.

Step 1 — Add ``DeviceSpec``
---------------------------

Create ``lmcache/v1/platform/<device>/__init__.py``:

.. code-block:: python

    # SPDX-License-Identifier: Apache-2.0
    """<device>-specific platform primitives."""

    from lmcache.v1.platform.base_device_spec import DeviceSpec


    class <Device>DeviceSpec(DeviceSpec):
        """<device> device specification for LMCache registry discovery."""

        @property
        def device_type(self) -> str:
            return "<device>"

        @property
        def torch_module_name(self) -> str:
            return "<device>"

        @property
        def ops_module(self) -> str | None:
            return "lmcache.v1.platform.<device>.ops"

        def is_available(self) -> bool:
            """Check backend availability without importing lmcache.__init__."""
            try:
                import torch

                return hasattr(torch, "<device>") and torch.<device>.is_available()
            except Exception:
                return False

Notes:

- Defining this class is enough for registration.
- No global list or manual registration call is required.
- ``DEVICE_TYPE=<device>`` can force selection when this backend is available.

Step 2 — Add ops backend module
-------------------------------

Create ``lmcache/v1/platform/<device>/ops.py``.

This module is merged over ``lmcache.python_ops_fallback`` by
``get_backend(device_type)``. Any function not implemented here uses fallback.

.. code-block:: python

    # SPDX-License-Identifier: Apache-2.0
    """<device> ops backend assembled into ``lmcache.c_ops`` at import time."""

    from __future__ import annotations

    import torch

    import lmcache.python_ops_fallback as py_ops


    def _tensor_list(value: object) -> list[torch.Tensor] | None:
        if not isinstance(value, list):
            return None
        if not all(isinstance(item, torch.Tensor) for item in value):
            return None
        return value


    def multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor: torch.Tensor | list,
        lmcache_objects_ptrs: list[int] | list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: py_ops.TransferDirection,
        shape_desc: py_ops.PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: py_ops.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Block-based multi-layer KV transfer for <device>."""
        from lmcache.v1.platform.<device>.native_kv_transfer import (
            try_native_multi_layer_block_kv_transfer,
        )

        object_tensors = _tensor_list(lmcache_objects_ptrs)
        if object_tensors is not None and try_native_multi_layer_block_kv_transfer(
            paged_layers=paged_buffer_ptrs_tensor,
            object_tensors=object_tensors,
            block_ids=block_ids,
            direction=direction,
            shape_desc=shape_desc,
            lmcache_chunk_size=lmcache_chunk_size,
            engine_kv_format=engine_kv_format,
            skip_prefix_n_blocks=skip_prefix_n_blocks,
        ):
            return

        py_ops.multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )

Initial bring-up can use Python fallback for validation, but native path
is required for production performance.

Step 3 — Native fast path adapter
---------------------------------

Add ``lmcache/v1/platform/<device>/native_kv_transfer.py`` as a
fail-closed adapter. Native performance is the standard expectation for new
hardware.

Contract
~~~~~~~~

- Native path must be optional (missing wheel should not break inference).
- Return ``False`` when unavailable/unsupported so fallback path runs.
- Keep ABI-compatible symbol set.

Required native symbols
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Symbol
     - Purpose
   * - ``native_lmcache_kv_transfer_abi_version``
     - ABI version check
   * - ``lmcache_kv_paged_to_buffer``
     - D2H gather (non-MLA)
   * - ``lmcache_kv_buffer_to_paged``
     - H2D scatter (non-MLA)
   * - ``lmcache_mla_paged_to_buffer``
     - D2H gather (MLA)
   * - ``lmcache_mla_buffer_to_paged``
     - H2D scatter (MLA)

Current ABI version is ``1``.

Reference implementation:
``lmcache/v1/platform/musa/native_kv_transfer.py``
(``LMCACHE_MUSA_NATIVE_KV_TRANSFER``, ABI version check, required symbol checks,
fail-closed behavior).

Verification
------------

Start LMCache server:

.. code-block:: bash

    lmcache server --l1-size-gb 10 --eviction-policy LRU --port 5555

Run vLLM with MP connector and engine-driven mode:

.. code-block:: bash

    vllm serve <your-model> \
        --kv-transfer-config '{
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": "5555",
                "lmcache.mp.mp_transfer_mode": "engine_driven"
            }
        }' \
        --no-enable-prefix-caching \
        --port 8000

Checklist:

- [ ] Backend is discoverable via ``DeviceSpec`` and selected correctly.
- [ ] ``lmcache.c_ops.multi_layer_block_kv_transfer`` resolves on your device.
- [ ] Engine-driven pickle transfer works end-to-end.
- [ ] Engine-driven SHM transfer works end-to-end.
- [ ] Store/retrieve correctness is verified.
- [ ] TP>1 / multi-worker behavior is verified.
- [ ] Native adapter path is hit and bit-exact.

References
----------

.. list-table::
   :header-rows: 1

   * - Topic
     - Path
   * - Device detection and backend loading
     - ``lmcache/v1/platform/__init__.py``
   * - Device spec base
     - ``lmcache/v1/platform/base_device_spec.py``
   * - Full ``DeviceSpec`` example
     - ``lmcache/v1/platform/musa/__init__.py``
   * - Ops backend example
     - ``lmcache/v1/platform/musa/ops.py``
   * - Native adapter example
     - ``lmcache/v1/platform/musa/native_kv_transfer.py``
   * - Engine-driven gather/scatter call site
     - ``lmcache/v1/multiprocess/transfer_context/base.py``
   * - Python fallback implementation
     - ``lmcache/python_ops_fallback.py``
   * - KV wrapper registry (advanced/optional)
     - ``lmcache/v1/platform/_registry.py``

Related docs
------------

- Multi-hardware architecture: ``docs/design/ARCHITECTURE_MULTI_HARDWARE.md``
- Engine-driven transfer design:
  ``docs/design/v1/multiprocess/engine_driven_transfer_design.md``
- Event notifier design: ``docs/design/v1/platform/event_notifier.md``
- MP protocol docs: ``docs/lmcache/v1/multiprocess/protocols/README.md``
