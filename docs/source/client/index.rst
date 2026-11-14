.. _lmcache_client_distribution:

Client distribution: choosing between ``lmcache`` and ``lmcache-client``
========================================================================

This page covers which LMCache distribution to install on an inference worker,
how to install and replace it, how to configure the SDK and the supported
engine integrations, and how to troubleshoot the problems that come up when a
worker and a server are installed from different packages.

.. note::

   ``lmcache-client`` is not published yet. The distribution is defined in
   `#6305 <https://github.com/LMCache/LMCache/issues/6305>`_ and its build and
   release work is tracked in
   `#6315 <https://github.com/LMCache/LMCache/issues/6315>`_ and
   `#6316 <https://github.com/LMCache/LMCache/issues/6316>`_. Sections below
   that describe the client wheel say so explicitly; today the installable
   distribution is ``lmcache``.

.. contents::
   :local:
   :depth: 2

Two distributions, one import namespace
---------------------------------------

LMCache is currently released as a single wheel containing the engine
adapters, client transport, server execution, storage backends and
control-plane services. That means an inference worker installs and upgrades
server code it never executes. The plan in
`#6305 <https://github.com/LMCache/LMCache/issues/6305>`_ adds a second,
lighter distribution for that case.

.. list-table:: What each distribution contains
   :header-rows: 1
   :widths: 26 37 37

   * - Distribution
     - Contents
     - Use it when
   * - ``lmcache``
     - The complete product surface: client, common, transport, platform,
       engine integrations, SDK and all four server-side services (cache
       server, fleet frontend, MP coordinator, memory orchestrator).
     - You run a server, or you want one package for a worker and a server on
       the same host.
   * - ``lmcache-client`` (planned)
     - A strict subset selected from the same sources: client, common,
       transport, the client-required platform closure, the engine
       integrations and the SDK. Server service implementations and their
       server-only dependencies are omitted.
     - You only run inference workers that talk to an LMCache server started
       elsewhere.

Both distributions expose the **same ``lmcache`` import namespace**. They are
alternatives for an environment, not packages to install side by side:
installing both means two distributions own the same module files, which makes
uninstall and upgrade behavior ambiguous. The full ``lmcache`` wheel remains the
superset and keeps every existing public import, CLI entry point and feature.

Decision guide
~~~~~~~~~~~~~~

.. list-table:: Choosing a distribution
   :header-rows: 1
   :widths: 50 50

   * - If you ...
     - Install
   * - start ``lmcache server`` on this host
     - ``lmcache``
   * - run only ``vllm serve`` / an engine process that connects to a remote
       LMCache server, and never start a server here
     - ``lmcache-client`` once it is released; ``lmcache`` until then
   * - run the fleet frontend, MP coordinator or memory orchestrator
     - ``lmcache``
   * - need a backend or CLI that ships with the cache server
     - ``lmcache``
   * - are unsure
     - ``lmcache`` — it is a superset, so nothing is missing

Installation
------------

The commands below run in a clean environment. They do not assume a source
checkout, a developer virtualenv or a machine-local path.

Prerequisites
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Notes
   * - Operating system
     - Linux for GPU deployments. Check the release notes for other platforms.
   * - Python
     - Check the ``Requires-Python`` of the wheel you are installing and use that
       minor version. The declared range and the versions a release actually
       resolves on are not the same: ``lmcache`` 0.5.6 declares ``>=3.10``, but a
       clean Python 3.10 environment fails to resolve because a dependency needs
       3.11 or newer. Python 3.12 installs cleanly.
   * - Accelerator runtime
     - The CUDA, ROCm or other runtime must match the wheel flavor. See
       :doc:`../getting_started/compatibility`.
   * - vLLM (optional)
     - Needed only for the vLLM integration. Version pairing is described in
       :doc:`../getting_started/compatibility`.

Installing the full distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install lmcache

``uv venv`` does not install ``pip`` into the environment, so use ``uv pip``
inside it (as above) rather than ``python -m pip``.

``pip install lmcache`` installs the same wheel. See
:doc:`../getting_started/installation` for the per-CUDA-version tabs and the
GitHub release channel used for some CUDA builds.

Optional extras
~~~~~~~~~~~~~~~

Extras depend on the build profile and differ between an editable install from
a checkout and the wheel published to PyPI; ``lmcache`` 0.5.6 on PyPI reports
``kafka`` and ``nixl``. Do not assume a fixed set, ask the installed wheel:

.. code-block:: bash

   python -c "import importlib.metadata as m; print(sorted(m.metadata('lmcache').get_all('Provides-Extra') or []))"

Installing the client distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   ``lmcache-client`` is not on PyPI yet. The commands, extras and version
   matrix for it are added once the wheel exists
   (`#6315 <https://github.com/LMCache/LMCache/issues/6315>`_). Until then,
   install ``lmcache`` on the worker as shown above; it contains the same
   client code.

Replacing, upgrading and uninstalling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because ``lmcache`` and ``lmcache-client`` share the ``lmcache`` namespace,
switching between them is a replacement, not a co-installation. The safe
sequence is uninstall first, then install the other distribution:

.. code-block:: bash

   uv pip uninstall lmcache lmcache-client
   uv pip install lmcache

.. warning::

   Do not install both distributions into one environment. If both are already
   present, uninstall both and reinstall the one you need — see
   :ref:`client-troubleshooting-coinstall`.

To upgrade within a distribution, reinstall it and let the resolver move the
version:

.. code-block:: bash

   uv pip install --upgrade lmcache

SDK and engine integration configuration
----------------------------------------

The SDK is the programmatic client. It connects to a running LMCache server and
is described in detail in :doc:`../mp/sdk`.

.. code-block:: python

   # First Party
   from lmcache.sdk import LMCacheSDKCacheKind, connect

   ctx = connect(
       kind=LMCacheSDKCacheKind.KV,
       url="tcp://localhost:5555",     # multiprocess request endpoint
       http_url="http://localhost:8080",
       model_name="Qwen/Qwen2.5-0.5B-Instruct",
   )

The ``url`` scheme selects the transport (``tcp://`` for ZMQ, ``grpc://`` for
gRPC). ``model_name`` must match the name the server was started for; a
mismatch shows up as a lookup that never resolves rather than an error.

Engine integrations stay at their existing import paths (``lmcache.integration``
for vLLM, SGLang and TensorRT-LLM adapters) in both distributions. The engine
library itself remains an optional dependency: installing LMCache does not pull
a specific inference engine, so install the engine you use and configure the
connector as described in :doc:`../mp/index`.

A minimal client/server request
-------------------------------

The smallest end-to-end check is one server, one worker, and a prompt sent
twice so the second request reuses the cached prefix.

Start the server in one terminal:

.. code-block:: bash

   lmcache server --host localhost --port 5555

Start the engine in another, pointing the MP connector at that server:

.. code-block:: bash

   vllm serve <model> --port 8000 \
     --kv-transfer-config '{
       "kv_connector": "LMCacheMPConnector",
       "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
       "kv_role": "kv_both",
       "kv_connector_extra_config": {
         "lmcache.mp.host": "localhost",
         "lmcache.mp.port": 5555
       }
     }'

Send the same prompt twice, then read the two cache-reuse signals:

.. code-block:: text

   # server log, first request
   Stored 432 tokens in 0.221 seconds

   # server log, second request with the same prefix
   Retrieved 432 tokens in 0.037 seconds

``Stored`` on the first request and ``Retrieved`` on the second is the signal
that the client and the server agree on the cache. A second ``Stored`` instead of
``Retrieved`` means the request missed — see the troubleshooting section.

The same prefix must produce the same object keys on both sides. Cache identity
covers the model name, the KV layout, and (for the MP connector) the model
revision or tag you configured; :doc:`../developer_guide/cache_identity`
describes the contract.

.. _client-troubleshooting:

Troubleshooting
---------------

.. _client-troubleshooting-coinstall:

Both distributions installed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptoms vary by tool: files owned by the other distribution, an import that
resolves to a path you did not expect, or a version that keeps reverting after
an upgrade. Confirm what owns the module, then replace instead of co-install:

.. code-block:: bash

   python -c "import lmcache, importlib.metadata as m; print(lmcache.__file__); print(m.version('lmcache'))"

If more than one distribution claims the namespace, uninstall both and install
one:

.. code-block:: bash

   uv pip uninstall lmcache lmcache-client
   uv pip install lmcache

Dependency conflicts with the engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An engine that pins a different version of a shared dependency will either fail
resolution or downgrade it. Prefer installing into a fresh environment, and let
the resolver report the conflict rather than forcing a version:

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install lmcache
   uv pip install vllm

When resolution fails, the message names the two requirements; choose the engine
and LMCache pair documented in :doc:`../getting_started/compatibility` rather
than forcing ``--no-deps``.

Optional engine not installed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integrations are optional on purpose, so importing one without its engine
raises ``ModuleNotFoundError`` for the engine module, not for ``lmcache``.
Install the engine you actually run:

.. code-block:: bash

   uv pip install vllm

Inside a worker process, ``LMCacheMPConnector`` fails fast when vLLM is absent;
plugins and other engine adapters are skipped at registration instead.

``cuda_ops`` extension warning on a stock wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Importing a wheel that does not ship the compiled CUDA ops logs this once:

.. code-block:: text

   WARNING: lmcache.cuda_ops compiled extension not found; CudaDeviceOps stays on the torch baseline

That is expected for a plain wheel: the device ops fall back to the torch
implementation. It matters only if you expected the native kernels, for example
because you built with the CUDA extension profile; then check that the installed
wheel is the one you built, as described below.

Native library or runtime mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A wheel built for a different accelerator runtime fails while loading its
native extension, typically with an ``ImportError`` mentioning a ``.so`` file or
a driver API. Two checks isolate it:

.. code-block:: bash

   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   python -c "import lmcache; print(lmcache.__file__)"

The first prints the runtime the wheel must match; the second shows whether the
import reaches the wheel you intended. Use the wheel flavor that matches the
runtime, or build from source against the already-installed PyTorch as described
in :doc:`../getting_started/compatibility`.

Protocol or version mismatch between client and server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MP transport requires the client and server to agree on the wire format and
on the cache identity. A mismatch shows up as one of:

- a handshake failure at connector or SDK startup;
- ``tcp://`` requests that time out instead of returning a lookup result;
- a server that stores objects but never retrieves them, because the two sides
  compute different object keys (see the cache-identity contract).

Check both sides first. ``lmcache`` has no top-level ``--version`` flag, so
read the installed distribution version on each host:

.. code-block:: bash

   python -c "import importlib.metadata as m; print(m.version('lmcache'))"

Install the same LMCache release on the worker and on the server host. The MP
transport has been changed in place before (request keys, transfer modes), so a
mixed pair is not supported for those fields.

Contributing
------------

The distribution split is tracked in
`#6305 <https://github.com/LMCache/LMCache/issues/6305>`_; the package layout and
migration rules are in
`docs/design/v1/package_boundaries.md <https://github.com/LMCache/LMCache/blob/92a1ffc8b7307c0ca1ec144227d42cdb99900224/docs/design/v1/package_boundaries.md>`_.
Contribution mechanics are in :doc:`../developer_guide/contributing`.
