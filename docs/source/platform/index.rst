Platform
========

LMCache is device-neutral and supports mainstream accelerator hardware. The
platform layer detects the active device backend and provides the
device-specific operations used by LMCache without tying the rest of LMCache
to a single hardware vendor.

Device backends are provided in two ways:

- **Built-in backends** are maintained in the LMCache repository and released
  with LMCache.
- **Device plugins** are vendor-maintained Python packages installed alongside
  LMCache. They register a backend through the ``lmcache.device_plugins``
  entry-point group and can follow an independent release cycle.

Built-in devices
----------------

LMCache currently includes the following built-in device backends:

- :doc:`CPU <devices/cpu/index>`
- :doc:`CUDA <devices/cuda/index>`
- :doc:`HPU <devices/hpu/index>`
- :doc:`MLU <devices/mlu/index>`
- :doc:`MUSA <devices/musa/index>`
- :doc:`Neuron <devices/neuron/index>`
- :doc:`NPU <devices/npu/index>`
- :doc:`RBLN <devices/rbln/index>`
- :doc:`ROCm <devices/rocm/index>`
- :doc:`SUPA <devices/supa/index>`
- :doc:`XPU <devices/xpu/index>`

Using a device plugin
---------------------

Install the vendor's device-plugin wheel in the same Python environment as
LMCache and the serving engine:

.. code-block:: bash

   python -m pip install lmcache-foo-device

Replace ``lmcache-foo-device`` with the package name published by the device
vendor.

Restart the LMCache and serving-engine processes after installation. LMCache
discovers installed ``lmcache.device_plugins`` entry points automatically and
selects the available backend. No LMCache source change or manual registration
is required.

To verify which device plugins are installed:

.. code-block:: bash

   python - <<'PY'
   from importlib.metadata import entry_points

   for plugin in entry_points(group="lmcache.device_plugins"):
       print(plugin.name, "=", plugin.value)
   PY

If more than one available backend uses the same device type, select the
intended plugin by its backend name before starting LMCache and the serving
engine:

.. code-block:: bash

   export LMCACHE_DEVICE_BACKEND=foo

Replace ``foo`` with the plugin's registered backend name.

Only install plugins from trusted sources because entry-point packages execute
Python code when the platform registry initializes. Device vendors and backend
developers can follow :doc:`Adding a New Device Backend
<../developer_guide/extending_lmcache/adding_a_new_device_backend>` to implement
and package a plugin.

.. toctree::
   :hidden:
   :maxdepth: 2

   devices/cpu/index
   devices/cuda/index
   devices/hpu/index
   devices/mlu/index
   devices/musa/index
   devices/neuron/index
   devices/npu/index
   devices/rbln/index
   devices/rocm/index
   devices/supa/index
   devices/xpu/index
