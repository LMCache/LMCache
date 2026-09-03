.. _vllm_lmcache_compatibility:

vLLM and LMCache compatibility
==============================

There is no single ``LMCache x vLLM`` version pair that guarantees every
model, connector, device, and storage backend. Compatibility has three
independent layers:

1. the Python, PyTorch, and accelerator runtime must be ABI-compatible;
2. vLLM must be able to load the LMCache connector implementation that you
   intend to use; and
3. the specific model or feature must be validated for that combination.

Use this page to choose an installation path and to report an exact
compatibility result. The :doc:`installation` page contains the commands for
each runtime and release channel.

What the version numbers mean
-----------------------------

``vLLM`` and ``LMCache`` are released independently. A newer LMCache release
does not automatically mean that every older vLLM release can load every
LMCache connector, and a newer vLLM release can change the KV-cache APIs that
an integration uses.

The following table separates the checks that are often incorrectly treated as
one compatibility matrix:

.. list-table:: Compatibility layers
   :header-rows: 1
   :widths: 24 40 36

   * - Layer
     - What must match
     - Recommended action
   * - Runtime and ABI
     - Python minor version, PyTorch build, CUDA/ROCm/oneAPI runtime, native
       extension ABI, and C++ ABI where applicable.
     - Use the LMCache wheel or container for the matching runtime. If the
       environment uses a different PyTorch build, build LMCache from source
       against that already-installed PyTorch.
   * - Connector loading
     - The vLLM release must support the connector loading mechanism used by
       the deployment.
     - For vLLM versions that support external connector modules, set
       ``kv_connector_module_path`` to the LMCache connector explicitly. Older
       releases use the connector implementation bundled by vLLM.
   * - Feature and model
     - KV layout, block size, hybrid/recurrent behavior, transfer mode, device,
       and storage backend.
     - Follow the relevant model or feature recipe and treat an unlisted
       combination as unverified until it has been smoke-tested.

Installation and ABI compatibility
-----------------------------------

For a published CUDA installation, use the stable or nightly instructions in
:doc:`installation`. LMCache's published native artifacts are built for a
particular PyTorch and accelerator runtime; installing a wheel from a different
runtime channel can produce native-extension or undefined-symbol errors even
when the Python package versions look compatible.

The practical choices are:

* **Published stable image or wheel:** use the LMCache release and runtime
  channel documented together on the installation page. This is the simplest
  choice for a production deployment.
* **Nightly vLLM or a vLLM source checkout:** use the matching LMCache nightly
  when the feature is only available on ``dev``. Nightlies are appropriate for
  testing upcoming integration changes, not as a stable-version guarantee.
* **Custom PyTorch or accelerator build:** install LMCache from source with
  ``--no-build-isolation`` so its native extensions are compiled against the
  PyTorch already present in the environment.

The release image follows the same rule: the vLLM and LMCache packages inside
the image are built as one runtime stack. Do not copy a native LMCache wheel
from another CUDA, ROCm, XPU, Python, or PyTorch channel into that image unless
the ABI tuple is known to match.

Connector loading in vLLM
-------------------------

For the vLLM multiprocess connector, the version boundary that matters is
whether vLLM can select an external connector module:

.. list-table:: MP connector loading
   :header-rows: 1
   :widths: 22 38 40

   * - vLLM version
     - What ``LMCacheMPConnector`` resolves to
     - Configuration guidance
   * - ``< 0.20.0``
     - vLLM's bundled ``LMCacheMPConnector``. These releases cannot redirect
       the name to the connector shipped by the LMCache package.
     - Use the connector and server protocol supported by that vLLM release,
       or upgrade vLLM before relying on a newer LMCache connector feature.
   * - ``>= 0.20.0``
     - The name still defaults to vLLM's bundled connector, but vLLM can load
       an external implementation.
     - Set ``kv_connector_module_path`` explicitly when using the connector
       shipped by LMCache.

The explicit external-connector configuration is:

.. code-block:: json

   {
     "kv_connector": "LMCacheMPConnector",
     "kv_connector_module_path":
       "lmcache.integration.vllm.lmcache_mp_connector",
     "kv_role": "kv_both"
   }

The external module path is important in a mixed environment: without it,
vLLM may silently select a bundled connector whose protocol or feature set is
older than the LMCache server. See :doc:`quickstart` for a complete MP command
and the :doc:`../api_reference/dynamic_connector` page for the vLLM external
connector mechanism.

Known version-specific validation points
-----------------------------------------

The following entries are exact facts currently recorded in the LMCache
documentation. They are validation points, not claims that all versions in the
same row or column are interchangeable.

.. list-table:: Feature-level compatibility points
   :header-rows: 1
   :widths: 30 28 42

   * - Feature or model
     - Version point
     - Meaning
   * - vLLM external MP connector
     - vLLM ``>= 0.20.0``
     - The LMCache-shipped connector can be selected with
       ``kv_connector_module_path``. The vLLM bundled connector remains the
       default unless it is overridden.
   * - vLLM KV events
     - vLLM ``0.13.0+``
     - This is the minimum version documented for the KV-events integration;
       event publishing still requires the corresponding vLLM configuration;
       see :doc:`../production/kv_cache_events`.
   * - :doc:`../recipes/glm5_2` (GLM-5.2 DSA)
     - vLLM ``0.23.0`` + LMCache ``0.4.7``
     - This exact model/engine combination is recorded as validated in the
       GLM-5.2 recipe. It is not a general support promise for every GLM or
       every later release.
   * - :doc:`../recipes/kimi_k3` (Kimi K3 hybrid)
     - LMCache nightly from 2026-07-27 or newer; stable LMCache from ``0.5.3``
     - The recipe currently requires the pre-release vLLM K3 image because the
       model support had not reached a stable vLLM release when it was
       validated. Check the recipe before upgrading the engine image.
   * - :doc:`../recipes/kimi_linear` (Kimi-Linear DCP)
     - vLLM newer than ``0.27.1``
     - DCP support for the documented Kimi-Linear example depends on the vLLM
       change identified in the MP configuration guide. The LMCache server
       chunk size must also satisfy the resolved block-span constraints.

For hybrid models, the model recipe and :doc:`../mp/hybrid_models` are more
authoritative than a package-version comparison. Different attention and
recurrent groups can have different physical page geometry, and a deployment
can be incompatible even when both packages import successfully.

How to evaluate a new combination
----------------------------------

When trying a vLLM release that is not listed in a recipe, record the complete
runtime tuple:

This includes a newly released version such as vLLM ``0.28.0``: being the
latest vLLM release does not by itself make the combination validated. Select
the appropriate LMCache stable/nightly channel, load the intended external
connector explicitly, and run the checks below before using it in production.

* LMCache version or commit and vLLM version or commit;
* Python minor version and PyTorch version/build;
* CUDA, ROCm, or XPU runtime and GPU model;
* connector name, external module path, transfer mode, and server version;
* model, tensor/pipeline/context parallel sizes, KV cache dtype, and block
  size; and
* the storage backend and whether the test is cold, warm, or cross-process.

At minimum, run these checks:

1. start the server and confirm that the intended connector module is loaded;
2. run one cold request that stores a prefix;
3. run a second request with the same prefix and confirm a non-zero cache hit;
4. clear or restart the engine as appropriate and verify a cross-process
   retrieve; and
5. compare output correctness and inspect the LMCache server logs for layout,
   transfer, or worker-liveness errors.

For a model-specific feature, also run the commands and correctness checks in
its recipe. A successful import or a single cache hit is not enough to mark a
new model, hybrid layout, or accelerator path as validated.

Support labels used on this page
--------------------------------

* **Validated** means the exact environment and scenario are documented with a
  successful test result.
* **Required** means a version boundary is needed to access a particular API
  or integration path; it does not validate every feature above that boundary.
* **Recommended** means the installation or development workflow should prefer
  that channel, but it is not a compatibility guarantee.
* **Unverified** means that no exact LMCache result is recorded in the current
  documentation. It may work, but it should be tested before production use.

The historical :download:`Installation compatibility CSV
<Installation_compatibility_matrix.csv>` covers older LMCache 0.3--0.4
releases. It should not be used as the current source of truth; this page and
the linked feature recipes describe the current compatibility policy.
