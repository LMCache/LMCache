.. SPDX-License-Identifier: Apache-2.0

Adding a New Device Backend
===========================

This guide explains how to add a **new accelerator** to LMCache in
**Multiprocess (MP) mode**. Device vendors can choose either supported
ownership model:

- **In-tree integration:** contribute and maintain the backend under
  ``lmcache/v1/platform/<device>/`` in the LMCache repository. This fits
  backends that should ship, test, and release with LMCache.
- **External wheel integration:** maintain the backend in a vendor repository
  and publish a wheel through the ``lmcache.device_plugins`` entry-point
  group. This fits backends that need an independent release cadence or own
  native-package distribution.

Both models implement the same ``DeviceSpec`` and ``DeviceOps`` interfaces
and use the same runtime detection and dispatch paths.

**For basic users:** read :ref:`Part 1 <part-1-basic>` only — a ``DeviceSpec``
and a ``DeviceOps`` subclass are all you need to use the built-in torch
baseline ops.

**For advanced users:** continue to :ref:`Part 2 <part-2-performance>`
for native ops and advanced transfer modes.

.. _part-1-basic:

Part 1 — Basic Function Enabling
---------------------------------

For the majority of devices, a ``DeviceSpec`` class plus a minimal
``DeviceOps`` subclass are sufficient.  LMCache ships with a complete
torch baseline ops layer (``lmcache/v1/platform/torch_ops.py``) that
works on any device supporting standard PyTorch tensor operations —
**no custom kernels required**.

Prerequisites
~~~~~~~~~~~~~

Your PyTorch backend should support:

**Device Discovery & Status:**

- ``torch.<device>.is_available()`` → ``bool``
- ``torch.<device>.device_count()`` → ``int``

**Device Context & Synchronization:**

- ``torch.<device>.set_device(device)`` → ``None``
- ``torch.<device>.current_device()`` → ``int``
- ``torch.<device>.synchronize()`` → ``None``

**Data Movement:**

- ``tensor.to(device)`` / ``tensor.cpu()`` (host↔device transfers)

Step 1: Choose the ownership model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Option A — integrate in the LMCache repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a platform package directly in LMCache::

    lmcache/v1/platform/foo/
    ├── __init__.py
    └── device_ops.py

Define ``FooDeviceSpec`` in ``__init__.py``. LMCache scans its built-in
platform packages, so this option needs no entry point or registration list.
Submit the code, tests, and device documentation in an LMCache PR; after
merge, the backend follows the LMCache release lifecycle.

Option B — maintain an external wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a ``src`` layout so the wheel owns only its vendor namespace::

    lmcache-foo-device/
    ├── pyproject.toml
    └── src/
        └── lmcache_foo/
            ├── __init__.py
            ├── device.py
            └── device_ops.py

The project must declare one entry point in the
``lmcache.device_plugins`` group. Its name is the lowercase
``DeviceSpec.backend_name`` and its value points to the ``DeviceSpec`` class:

.. code-block:: toml

    # pyproject.toml
    [build-system]
    requires = ["setuptools>=77", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "lmcache-foo-device"
    version = "0.1.0"
    dependencies = ["lmcache"]

    [project.entry-points."lmcache.device_plugins"]
    foo = "lmcache_foo.device:FooDeviceSpec"

    [tool.setuptools.packages.find]
    where = ["src"]

Pin ``lmcache`` to the version range tested by your plugin before publishing
it. LMCache treats ``DeviceSpec`` and ``DeviceOps`` as the plugin interface;
incompatible interface changes should be caught by that dependency range.

Step 2: Implement ``FooDeviceSpec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the spec in the location selected in Step 1:

- In-tree: ``lmcache/v1/platform/foo/__init__.py``.
- External wheel: ``src/lmcache_foo/device.py``.

The implementation is the same in both layouts:

.. code-block:: python

    # SPDX-License-Identifier: Apache-2.0
    """LMCache platform registration for Foo devices."""

    from __future__ import annotations

    from typing import TYPE_CHECKING

    from lmcache.v1.platform.base.device_spec import DeviceSpec

    if TYPE_CHECKING:
        from lmcache.v1.platform.base.device_ops import DeviceOps


    class FooDeviceSpec(DeviceSpec):
        """Foo device specification for LMCache registry discovery."""

        @property
        def device_type(self) -> str:
            return "foo"

        @property
        def backend_name(self) -> str:
            return "foo"

        @property
        def torch_module_name(self) -> str:
            return "foo"

        @property
        def ops_cls(self) -> type[DeviceOps]:
            from .device_ops import FooDeviceOps

            return FooDeviceOps

        def is_available(self) -> bool:
            """Check backend availability without importing lmcache.__init__."""
            try:
                import torch

                return hasattr(torch, "foo") and torch.foo.is_available()
            except Exception:
                return False

For an external wheel, the entry-point target must be the class itself, not a
class instance or factory. For either model, the class must have a no-argument
constructor; LMCache instantiates it once per process and caches it.

Step 3: Implement ``FooDeviceOps``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``device_ops.py`` next to the spec package:

- In-tree: ``lmcache/v1/platform/foo/device_ops.py``.
- External wheel: ``src/lmcache_foo/device_ops.py``.

.. code-block:: python

    # SPDX-License-Identifier: Apache-2.0
    """Foo ops backend."""

    from __future__ import annotations

    from typing import ClassVar

    from lmcache.v1.platform.base.device_ops import DeviceOps


    class FooDeviceOps(DeviceOps):
        device_type: ClassVar[str] = "foo"

        def ensure_native(self) -> None:
            """Keep the torch baseline until native ops are available."""
            return None

.. note::

   If you have no native extension yet, leave ``ensure_native`` as a no-op.
   The torch baseline handles everything. See :ref:`Part 2
   <part-2-performance>` when the backend also ships native code.

Key properties:

.. list-table::
   :header-rows: 1

   * - Property / Method
     - Required
     - Purpose
   * - ``DeviceSpec.device_type``
     - yes
     - Device type string (e.g. ``"cuda"``, ``"musa"``, ``"xpu"``)
   * - ``DeviceSpec.backend_name``
     - yes
     - Unique LMCache selector for one concrete backend implementation
   * - ``DeviceSpec.torch_module_name``
     - yes
     - Attribute on the ``torch`` package (e.g. ``"cuda"`` →
       ``torch.cuda``)
   * - ``DeviceSpec.ops_cls``
     - no
     - Returns the ``DeviceOps`` subclass for this device.
       Base returns ``DeviceOps`` itself (pure torch baseline).
   * - ``DeviceSpec.is_available()``
     - yes
     - Returns ``True`` when the device is usable
   * - ``DeviceOps.ensure_native()``
     - no
     - Called once on first use; override to bind native ops.
       Base is a no-op (e.g. ``HpuDeviceOps`` inherits it unchanged).

.. note::

   The ``hasattr(torch, "foo")`` guard shown above is only needed for
   out-of-tree PyTorch extensions (e.g. ``torch.musa``, ``torch.xpu``
   when installed as a plug-in).  For accelerators shipped inside
   PyTorch itself (like ``torch.cuda``) a plain
   ``torch.foo.is_available()`` is enough.

Step 4: Install the selected integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an in-tree backend, build or install LMCache normally from the branch
containing the platform package. Once merged and released, users receive the
backend with LMCache itself.

For an external backend, build the vendor project with any PEP 517 frontend,
then install its wheel in the same Python environment as LMCache and the
serving engine:

.. code-block:: bash

    python -m pip install build
    python -m build --wheel
    python -m pip install dist/lmcache_foo_device-0.1.0-py3-none-any.whl

Restart every LMCache and serving-engine process after installing either
integration. The platform registry is built once and cached for the lifetime
of each process.

Neither model requires editing a global device-name list. All ops route
through ``torch_ops.py`` until the backend overrides them.

External wheel loading rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``DeviceSpec.device_type`` is the torch-facing device category (for example,
  ``"cuda"``). Multiple backends may intentionally share one ``device_type``.
- ``DeviceSpec.backend_name`` is the LMCache-specific selector for one concrete
  backend implementation. It must be a unique, non-empty lowercase string.
- For external wheels, the entry-point name must match
  ``DeviceSpec.backend_name``.
- Duplicate ``backend_name`` values are ignored after the first deterministic
  match.
- A plugin that cannot be imported, resolves to the wrong object type, or
  raises during construction is logged and skipped. Other devices remain
  usable.
- Entry-point modules are imported while the platform registry initializes.
  Import base interfaces such as
  ``lmcache.v1.platform.base.device_spec.DeviceSpec`` directly and keep ops,
  native libraries, IPC wrappers, and cache contexts behind lazy properties.
- Installed entry-point packages are executable Python code. Only install
  wheels from sources you trust.

Verification
~~~~~~~~~~~~

Start LMCache server.  The worker below uses the engine-driven
transfer path, which the server only loads when
``--supported-transfer-mode`` is ``engine_driven`` or ``auto`` (the
default is ``lmcache_driven``):

.. code-block:: bash

    lmcache server --l1-size-gb 10 --eviction-policy LRU --port 5555 \
        --supported-transfer-mode engine_driven

Run vLLM with MP connector. If you want a specific torch device category, set
``DEVICE_TYPE``. Backends that share one device type are selected automatically
when exactly one reports available. If multiple backends report available, set
``LMCACHE_DEVICE_BACKEND`` to select the exact implementation:

.. code-block:: bash

    export DEVICE_TYPE=foo             # optional; selects the torch-facing device type
    export LMCACHE_DEVICE_BACKEND=foo  # optional; disambiguates multiple available backends

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

.. note::

   The default transfer mode is ``auto`` (CUDA → LMCache-driven, other
   devices → engine-driven).  The example above explicitly sets
   ``engine_driven`` so that a new non-CUDA device works without
   additional capability checks.  For the ``lmcache_driven`` mode
   (IPC zero-copy), see :ref:`Advanced transfer mode <part-2-performance>`.

Check the LMCache logs::

    torch_dev=..., torch_device_type=foo

This confirms your ``DeviceSpec`` was discovered and the torch
baseline is active. When ``LMCACHE_DEVICE_BACKEND`` is set, LMCache also binds
the backend whose ``backend_name`` matches that value.

Debugging checklist:

- [ ] ``torch.foo.is_available()`` returns ``True``.
- [ ] For an external wheel,
  ``importlib.metadata.entry_points(group="lmcache.device_plugins")`` includes
  ``foo`` and points to ``FooDeviceSpec``.
- [ ] If another available backend shares ``device_type="foo"``, set
  ``LMCACHE_DEVICE_BACKEND=foo`` to disambiguate them.
- [ ] Set ``DEVICE_TYPE=foo`` to force the torch-facing device category if not
  picked up automatically.
- [ ] Engine-driven transfer works end-to-end (check the LMCache logs
  to confirm whether the SHM or Pickle sub-path is chosen — both
  should succeed).
- [ ] Store/retrieve correctness is verified.
- [ ] TP>1 / multi-worker behavior is verified.

.. _part-2-performance:

Part 2 — Performance Optimization
----------------------------------

Once basic functionality is verified, add device-specific
optimizations.

Device-specific ops via ``bind_native``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DeviceOps`` base class delegates every op to the torch baseline
in ``lmcache/v1/platform/torch_ops.py``.  Each vendor may replace any
subset of these functions with a device-specific implementation.

**How it works.** When ``ensure_native()`` calls
``self.bind_native(native_module)``, the method walks the module's
public symbols and rebinds them as instance attributes — overriding the
base-class methods that delegate to ``torch_ops``:

.. code-block:: text

    callers  →  lmcache.device_ops (DeviceOps instance)
                                           │
                                     bind_native() overlay:
                                       ├── native.multi_layer_kv_transfer  ← vendor CUDA/SYCL kernel
                                       ├── native.calculate_cdf            ← vendor kernel
                                       └── (everything else)               ← torch_ops baseline

Integration contract
^^^^^^^^^^^^^^^^^^^^

Regardless of how you build your ops module, the following contract
must hold:

- **Same symbol names.** Every function you override must be exposed
  under the exact name used in ``torch_ops.py`` (e.g.
  ``multi_layer_block_kv_transfer``).
- **Same call signature.** Positional/keyword arguments, argument
  order and semantics must match the baseline; callers invoke the
  ``DeviceOps`` instance without knowing which backend answered.
- **Importable Python module.** Your native module must be importable
  via ``import`` (how it gets there — a pure-Python file, a pybind11
  extension, a ctypes wrapper, a Rust ``PyO3`` module, etc. — is your
  choice).
- **Partial override is allowed.** You do not have to reimplement
  every function.  Anything you leave out keeps using the torch
  baseline, so incremental optimization is supported.
- **Types from the native module are also bound.** ``bind_native``
  also binds types (classes) — this is how native plan types like
  ``StagingCopy``, ``KernelGroupSpec``, etc. overlay the stubs in
  ``ops_types.py``.

Implementation notes
^^^^^^^^^^^^^^^^^^^^

- ``multi_layer_block_kv_transfer`` and ``lmcache_memcpy_async`` are
  the hot entry points for both engine-driven and LMCache-driven
  transfer; other functions in ``torch_ops.py`` can be overridden as
  needed.
- If you fall back to the generic path from inside a device-specific
  wrapper (e.g. when inputs are unsupported), call the corresponding
  ``lmcache.v1.platform.torch_ops`` function directly to preserve
  semantics.
- ``ensure_native()`` is called once when ``DeviceSpec.get_ops()``
  first creates the singleton. It is safe to fail soft (log a warning
  and return without binding).

For concrete reference implementations, see:

- ``lmcache/v1/platform/cuda/device_ops.py`` — binds the compiled
  ``lmcache.cuda_ops`` pybind11 extension.
- ``lmcache/v1/platform/xpu/device_ops.py`` — binds the SYCL
  ``lmcache.xpu_ops`` extension.
- ``lmcache/v1/platform/musa/device_ops.py`` — method overrides
  without a separate native module.

Advanced transfer mode
~~~~~~~~~~~~~~~~~~~~~~

By default, the transfer mode is **AUTO**: the router dispatches
strictly by ``device_type`` — ``device_type == "cuda"`` goes to
``LMCacheDrivenTransferContext`` (IPC zero-copy), everything else to
``EngineDrivenTransferContext``.  A non-CUDA device that supports IPC
handle transfer can still opt into LMCache-driven explicitly (below).

.. note::

   Under PyTorch a ROCm GPU reports ``device_type == "cuda"``. LMCache
   registers a distinct ``RocmDeviceSpec`` with ``backend_name == "rocm"``
   while reusing the CUDA platform's ops, cache context, and IPC wrapper.
   CUDA and ROCm availability checks are mutually exclusive, so AUTO mode
   selects the correct backend without extra configuration.

When the caller (or ``LMCACHE_MP_TRANSFER_MODE``) explicitly requests
``lmcache_driven``, ``_build_lmcache_driven_context`` performs two hard
checks — both must succeed, otherwise the factory raises
``ValueError`` (no silent degradation):

1. Your ``DeviceSpec`` subclass must bind a ``DeviceIPCWrapper``
   subclass (exposing a ``wrap`` classmethod) via
   :attr:`~lmcache.v1.platform.base.device_spec.DeviceSpec.ipc_wrapper_cls`.
   :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory` reads that
   binding off the registered spec — no separate registry / auto-scan.
2. ``DeviceSpec.is_handle_transfer_available()`` must return ``True``
   (the base-class default; override to ``False`` only if your device
   lacks IPC handle transfer).

Separately, the LMCache-driven server module also requires a
``BaseCacheContext`` subclass next to the backend (for example,
``lmcache/v1/platform/foo/cache_context.py`` in-tree or
``lmcache_foo/cache_context.py`` externally) **and** a matching
``DeviceSpec.create_cache_context`` override that lazy-imports and
instantiates it.  The platform-agnostic factory
``lmcache.v1.platform.cache_context.create_cache_context`` dispatches
by ``device_type`` through the ``DeviceSpec`` registry and invokes
that hook; the default ``DeviceSpec.create_cache_context`` raises
``NotImplementedError`` so a missing override surfaces loudly instead
of silently falling back.  The cache context itself manages the KV
cache layout and pointers used for IPC transfer.

Host-side pinning via ``pin_memory_backend`` is *optional* and only
affects staging-buffer performance; it is not required to enable
LMCache-driven mode.

Event IPC capability
^^^^^^^^^^^^^^^^^^^^

The LMCache-driven multiprocess handle path also requires a platform
event-IPC backend.  The capability is declared by
``DeviceSpec.event_ipc_backend`` and is intentionally separate from
``DeviceOps`` and ``DeviceIPCWrapper``:

* The base ``DeviceSpec`` returns ``None``.  A concrete device must
  explicitly opt in.
* CUDA-style event APIs can use
  ``DefaultEventIPCBackend(event_module=..., device_type=...)``.
* Devices with a different event ABI should implement an
  ``EventIPCBackend`` next to their in-tree or external backend.
* Concrete device specs should cache the backend because request futures
  may query this property repeatedly.

The backend contract covers event creation, handle export/import, event
recording, stream wait, query, and synchronization.
``check_event_support(device)`` must raise ``RuntimeError`` when those
operations are unavailable for the requested device.  The default backend
checks for a CUDA-style Event type that accepts ``interprocess=True`` and
provides ``from_ipc_handle``.  A custom backend should instead validate the
equivalent prerequisites for its own event ABI.

For a CUDA-style device, bind and cache the default backend as follows:

.. code-block:: python

    from typing import TYPE_CHECKING

    from lmcache.v1.platform.base.device_spec import DeviceSpec

    if TYPE_CHECKING:
        from lmcache.v1.platform.base.event_ipc import EventIPCBackend

    class FooDeviceSpec(DeviceSpec):
        _event_backend_cache: "EventIPCBackend | None" = None

        @property
        def event_ipc_backend(self) -> "EventIPCBackend":
            backend = self._event_backend_cache
            if backend is None:
                import torch

                from lmcache.v1.platform.base.event_ipc import (
                    DefaultEventIPCBackend,
                )

                backend = DefaultEventIPCBackend(
                    event_module=torch.foo,
                    device_type=self.device_type,
                )
                self._event_backend_cache = backend
            return backend

Event IPC operations must preserve producer/consumer stream ordering without
adding a device-wide synchronization.  ``query_event`` must remain
non-blocking so request futures can poll completion safely.

.. note::

   If the device does not support Event IPC, leave the base
   ``event_ipc_backend`` implementation unchanged so it returns ``None``.
   Engine-driven mode does not require this capability.  LMCache-driven
   mode raises an explicit error rather than falling back to CUDA or to the
   accelerator active in the process.

The capability is checked during worker/server registration and before
constructing a device-aware completion future.  This keeps unsupported
platforms from entering an asynchronous transfer path that cannot order
KV-cache memory safely.  The STORE and RETRIEVE message wire format is
unchanged: the existing event handle bytes are still carried in the
request and response payloads.

Override these methods in your ``DeviceSpec``:

.. code-block:: python

    class FooDeviceSpec(DeviceSpec):
        @property
        def ipc_wrapper_cls(self):
            """Bind the DeviceIPCWrapper subclass for this device.

            Lazy import so the accelerator-specific module is only
            pulled in when the LMCache-driven path is actually used.
            """
            from .ipc_wrapper import FooIPCWrapper

            return FooIPCWrapper

        def is_handle_transfer_available(self) -> bool:
            """Return True if your device supports IPC handle transfer."""
            return True  # base-class default; override to False if unsupported

        @property
        def pin_memory_backend(self):
            """Return a PinMemoryBackend subclass, or None.

            Optional; only affects host staging performance.
            """
            return None  # default

        def create_cache_context(self, *args, **kwargs):
            """Lazy-import and instantiate the BaseCacheContext for this device.

            Required for LMCache-driven mode; the base-class default
            raises ``NotImplementedError``.
            """
            from .cache_context import FooCacheContext

            return FooCacheContext(*args, **kwargs)

Opt into LMCache-driven mode by setting ``lmcache.mp.mp_transfer_mode``
to ``lmcache_driven`` in the vLLM ``kv_connector_extra_config`` shown
in :ref:`Part 1 <part-1-basic>`, or by exporting
``LMCACHE_MP_TRANSFER_MODE=lmcache_driven``.  If either hard check
fails, the factory raises ``ValueError`` and refuses to construct the
context — switch back to ``engine_driven`` or ``auto``.

References
----------

.. list-table::
   :header-rows: 1

   * - Topic
     - Path
   * - Device spec base
     - ``lmcache/v1/platform/base/device_spec.py``
   * - Device ops base
     - ``lmcache/v1/platform/base/device_ops.py``
   * - Torch ops baseline
     - ``lmcache/v1/platform/torch_ops.py``
   * - Ops types and enums
     - ``lmcache/v1/platform/ops_types.py``
   * - Event IPC base
     - ``lmcache/v1/platform/base/event_ipc.py``
   * - Backend loading (``resolve_device_ops`` / ``_detect_device``)
     - ``lmcache/v1/platform/_device_detect.py`` and
       ``lmcache/v1/platform/__init__.py``
   * - Package-level ``device_ops`` resolution
     - ``lmcache/__init__.py``
   * - Cache context base
     - ``lmcache/v1/platform/base/cache_context.py``
   * - Cache context factory
     - ``lmcache/v1/platform/cache_context.py``
   * - Reference ``DeviceSpec`` (CUDA)
     - ``lmcache/v1/platform/cuda/__init__.py``
   * - Reference ``DeviceOps`` (CUDA, bind_native)
     - ``lmcache/v1/platform/cuda/device_ops.py``
   * - Reference ``DeviceOps`` (XPU, bind_native)
     - ``lmcache/v1/platform/xpu/device_ops.py``
   * - Reference ``DeviceOps`` (MUSA, method overrides)
     - ``lmcache/v1/platform/musa/device_ops.py``
   * - Reference ``DeviceSpec`` (Neuron / Trainium, engine-driven only)
     - ``lmcache/v1/platform/neuron/__init__.py``
   * - Reference ``DeviceOps`` (Neuron, torch baseline)
     - ``lmcache/v1/platform/neuron/device_ops.py``
   * - Engine-driven call site
     - ``lmcache/v1/multiprocess/transfer_context/worker_transfer.py``
       (``EngineDrivenTransferContext``, ``create_transfer_context``)
