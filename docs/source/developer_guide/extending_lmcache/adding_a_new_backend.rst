.. SPDX-License-Identifier: Apache-2.0

Adding a New Device Backend
===========================

This guide explains how to add a **new accelerator** to LMCache in
**Multiprocess (MP) mode**.

**For basic users:** read :ref:`Part 1 <part-1-basic>` only — adding a
``DeviceSpec`` is all you need to get your device working with the
built-in Python fallback ops.

**For advanced users:** continue to :ref:`Part 2 <part-2-performance>`
for native ops and advanced transfer modes.

.. _part-1-basic:

Part 1 — Basic Function Enabling
--------------------------------

For the majority of devices, a single ``DeviceSpec`` class is
sufficient.  LMCache ships with a complete Python fallback ops
(``lmcache/python_ops_fallback.py``) that works on any device
supporting standard PyTorch tensor operations — **no custom kernels
required**.

Prerequisites
~~~~~~~~~~~~~

Your PyTorch backend should support:

- ``torch.<device>.is_available()``
- ``torch.<device>.device_count()``
- ``torch.<device>.set_device()`` / ``current_device()`` /
  ``synchronize()``
- Tensor movement between host and device (``.cpu()``,
  ``.to(<device>)``)

Add ``DeviceSpec``
~~~~~~~~~~~~~~~~~~

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
            return None

        def is_available(self) -> bool:
            """Check backend availability without importing lmcache.__init__."""
            try:
                import torch

                return hasattr(torch, "<device>") and torch.<device>.is_available()
            except Exception:
                return False

Key properties:

.. list-table::
   :header-rows: 1

   * - Property
     - Required
     - Purpose
   * - ``device_type``
     - yes
     - Device type string (e.g. ``"cuda"``, ``"musa"``)
   * - ``torch_module_name``
     - yes
     - Attribute on the ``torch`` package (e.g. ``"cuda"`` →
       ``torch.cuda``)
   * - ``ops_module``
     - no
     - Ops module path; leave the base-class default (``None``) for
       pure fallback, or override in :ref:`Part 2 <part-2-performance>`
   * - ``is_available()``
     - yes
     - Returns ``True`` when the device is usable

.. note::

   ``ops_module`` defaults to ``None`` in the base ``DeviceSpec``, so
   you can simply omit that property for a minimal Part 1 setup.  The
   ``hasattr(torch, "<device>")`` guard shown above is only needed for
   out-of-tree PyTorch extensions (e.g. ``torch.musa``, ``torch.xpu``
   when installed as a plug-in).  For accelerators shipped inside
   PyTorch itself (like ``torch.cuda``) a plain
   ``torch.<device>.is_available()`` is enough.

That's it.  Defining this class is enough for auto-discovery — no
global list or manual registration call is required.  All ops
automatically route through ``lmcache.python_ops_fallback``, which is
not performant but is functionally applicable to any device that
supports the standard PyTorch tensor surface.  Later, each device can
provide its own ops (Python, C, CUDA, Rust, or any language exposing
Python bindings) to override the fallback function-by-function.

Verification
~~~~~~~~~~~~

Start LMCache server:

.. code-block:: bash

    lmcache server --l1-size-gb 10 --eviction-policy LRU --port 5555

Run vLLM with MP connector.  If multiple accelerators are visible on
the host, set ``DEVICE_TYPE`` to force LMCache to pick the new backend
instead of auto-detecting:

.. code-block:: bash

    export DEVICE_TYPE=<device>            # optional; only when auto-detection picks the wrong device

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

Check the LMCache logs — with ``ops_module = None`` you should see::

    torch_dev=..., torch_device_type=<device>
    Custom ops not supported for device: <device>, using fallback ops.

This confirms your ``DeviceSpec`` was discovered and the Python
fallback is active.  (Once you set ``ops_module`` in
:ref:`Part 2 <part-2-performance>`, the second line becomes
``Using backend: <your.ops.module>`` instead.)

Debugging checklist:

- [ ] ``torch.<device>.is_available()`` returns ``True``.
- [ ] Set ``DEVICE_TYPE=<device>`` to force selection if not picked up
  automatically.
- [ ] Log shows either ``Custom ops not supported for device: <device>,
  using fallback ops.`` (pure fallback) or
  ``Using backend: <your.ops.module>`` (custom ops loaded).
- [ ] Engine-driven transfer works end-to-end (check the LMCache logs
  to confirm whether the SHM or Pickle sub-path is chosen — both
  should succeed).
- [ ] Store/retrieve correctness is verified.
- [ ] TP>1 / multi-worker behavior is verified.

.. _part-2-performance:

Part 2 — Performance Optimization
---------------------------------

Once basic functionality is verified, add device-specific
optimizations.

Device-specific ops
~~~~~~~~~~~~~~~~~~~

The generic ops interface is defined in
``lmcache/python_ops_fallback.py``.  Each vendor may replace any
subset of these functions with a device-specific implementation; the
choice of language (C, CUDA, SYCL, Python, Rust, …) and packaging is
entirely up to the vendor.  LMCache only cares that the resulting
symbols are importable from Python.

**How it works.** At import time, ``get_backend(device_type)`` imports
the module named by your ``DeviceSpec.ops_module`` and merges its
symbols over ``python_ops_fallback``:

.. code-block:: text

    callers  →  lmcache.c_ops  →  <your ops module>       (functions you defined)
                               →  lmcache.python_ops_fallback  (everything else)

Integration contract
^^^^^^^^^^^^^^^^^^^^

Regardless of how you build your ops module, the following contract
must hold:

- **Same symbol names.** Every function you override must be exposed
  under the exact name used in ``python_ops_fallback`` (e.g.
  ``multi_layer_block_kv_transfer``).
- **Same call signature.** Positional/keyword arguments, argument
  order and semantics must match the fallback; callers invoke the
  merged module without knowing which backend answered.
- **Importable Python module.** ``ops_module`` is a fully-qualified
  Python module path (``importlib.import_module`` must succeed).  How
  the module gets there — a pure-Python file, a pybind11 extension,
  a ctypes wrapper, a Rust ``PyO3`` module, etc. — is your choice.
- **Partial override is allowed.** You do not have to reimplement
  every function.  Anything you leave out keeps using the Python
  fallback, so incremental optimization is supported.
- **Point ``DeviceSpec.ops_module`` at the module** once it is
  reachable on ``sys.path``:

  .. code-block:: python

      class <Device>DeviceSpec(DeviceSpec):
          @property
          def ops_module(self) -> str | None:
              return "my_device.ops"   # any importable module path

Implementation notes
^^^^^^^^^^^^^^^^^^^^

- For engine-driven transfer the hot entry point is
  ``multi_layer_block_kv_transfer``; other functions in
  ``python_ops_fallback`` can be overridden as needed.
- If you fall back to the generic path from inside a device-specific
  wrapper (e.g. when inputs are unsupported), call the corresponding
  ``lmcache.python_ops_fallback`` function directly to preserve
  semantics.
- Keep your ops module free of side effects at import time — it is
  imported eagerly during ``get_backend()``.

For concrete reference implementations, see
``lmcache/v1/platform/cuda/`` and ``lmcache/v1/platform/musa/``.
Both are examples of what a vendor *may* do, not templates every new
backend has to follow.

Advanced transfer mode
~~~~~~~~~~~~~~~~~~~~~~

By default, devices use **engine-driven** transfer mode.  Some devices
can support the **LMCache-driven** mode for better multi-worker
throughput via IPC-based zero-copy handle transfer.

When the caller (or ``LMCACHE_MP_TRANSFER_MODE``) explicitly requests
``lmcache_driven``, ``_build_lmcache_driven_context`` performs two
hard checks against the device — both must succeed, otherwise the
factory raises ``ValueError`` (there is no silent fallback):

1. A KV IPC wrapper factory must be registered for the device via
   ``lmcache/v1/platform/_registry.py`` (see ``register_kv_wrapper`` /
   ``get_kv_wrapper_factory``).
2. The device's ``DeviceSpec.is_handle_transfer_available()`` must
   return ``True``.

Host-side pinning via ``pin_memory_backend`` is *optional* and only
affects staging-buffer performance; it is not required to enable
LMCache-driven mode.

Override these methods in your ``DeviceSpec`` accordingly (note that
``is_handle_transfer_available()`` defaults to ``True`` in the base
class, so you only need to override it when your device does *not*
support IPC handle transfer):

.. code-block:: python

    class <Device>DeviceSpec(DeviceSpec):
        def is_handle_transfer_available(self) -> bool:
            """Return True if your device supports IPC handle transfer."""
            return True  # base-class default; override to False if unsupported

        @property
        def pin_memory_backend(self):
            """Return a PinMemoryBackend subclass, or None.

            Optional; only affects host staging performance.
            """
            return None  # default

Once the checks above pass, opt into LMCache-driven mode by setting
``lmcache.mp.mp_transfer_mode`` to ``lmcache_driven`` in the vLLM
``kv_connector_extra_config`` shown in :ref:`Part 1 <part-1-basic>`,
or by exporting ``LMCACHE_MP_TRANSFER_MODE=lmcache_driven``.  If
either capability check above fails, the factory raises
``ValueError`` and refuses to construct the context — the caller must
switch back to ``engine_driven`` or ``auto``.

.. note::

   In ``auto`` mode the router still dispatches strictly by device
   type: only ``device_type == "cuda"`` is routed to
   ``LMCacheDrivenTransferContext``; every other device is routed to
   ``EngineDrivenTransferContext``.  A non-CUDA device that supports
   handle transfer must therefore opt in explicitly via
   ``lmcache.mp.mp_transfer_mode = "lmcache_driven"`` /
   ``LMCACHE_MP_TRANSFER_MODE=lmcache_driven``.

References
----------

.. list-table::
   :header-rows: 1

   * - Topic
     - Path
   * - Device spec base
     - ``lmcache/v1/platform/base_device_spec.py``
   * - Backend loading
     - ``lmcache/v1/platform/__init__.py``
   * - Python fallback
     - ``lmcache/python_ops_fallback.py``
   * - Reference ``DeviceSpec`` (engine-driven baseline)
     - ``lmcache/v1/platform/cuda/__init__.py``
   * - Reference ``DeviceSpec`` (LMCache-driven capable)
     - ``lmcache/v1/platform/musa/__init__.py``
   * - Engine-driven call site
     - ``lmcache/v1/multiprocess/transfer_context/worker_transfer.py``
       (``EngineDrivenTransferContext``, ``create_transfer_context``)
