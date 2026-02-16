Native Remote Backend (Experimental)
=====================================

.. note::

   This feature is **experimental**. The core logic mirrors the Python
   ``RemoteBackend`` — it supports ``put``, ``get``, ``contains``, and
   ``remove`` — but **batched APIs** (``batched_get_blocking``,
   ``batched_submit_put_task`` with batched connector calls, etc.) and
   advanced features like MLA mode are **not yet implemented**.
   Performance optimizations are planned for future releases.

Overview
--------

The *Native Remote Backend* (``RustRemoteBackend``) is a storage plugin
that delegates all I/O to a **native shared library** (written in C, C++,
or Rust) instead of going through Python-level connectors and serializers.

This is useful when:

- The third-party storage SDK is **natively implemented in C++ or Rust**
  (e.g., a vendor-specific object store, an RDMA-based transport, or a
  custom distributed file system). With this backend you can call the
  native SDK directly — no Python wrapper or binding required.
- You want to **eliminate Python GIL overhead** on the I/O hot path.
  The Rust backend releases the GIL before calling into the connector,
  so the Python event loop and other threads remain unblocked.
- You need a **plugin-style deployment**: the connector ships as a
  standalone ``.so`` / ``.dylib`` that can be built, versioned, and
  installed independently of LMCache.

Architecture
------------

.. mermaid::

   flowchart TB
      subgraph "Python Process"
         direction TB
         plugin["RustRemoteBackend&lt;br/&gt;(StoragePluginInterface)"]
         pymodule["lmcache_rust_remote_backend_io&lt;br/&gt;(PyO3 Extension)"]
      end

      subgraph "Rust Layer (GIL-released)"
         direction TB
         backend["RustRemoteBackend&lt;br/&gt;(PyO3 Class)"]
         state["BackendState&lt;br/&gt;• meta_index: HashMap&lt;br/&gt;• put_tasks: HashSet"]
         loader["ConnectorHandle&lt;br/&gt;(libloading / dlopen)"]
      end

      subgraph "Native Connector (cdylib)"
         direction TB
         cabi["C ABI&lt;br/&gt;connector_api.h"]
         impl1["Built-in: lmcache_connector_fs&lt;br/&gt;(Rust cdylib)"]
         impl2["Third-party: your_connector&lt;br/&gt;(C/C++/Rust cdylib)"]
      end

      plugin --> pymodule
      pymodule --> backend
      backend --> state
      backend --> loader
      loader -->|"dlopen + symbol resolve"| cabi
      cabi --- impl1
      cabi --- impl2

The data flow for a ``put`` operation:

1. Python calls ``batched_submit_put_task(keys, objs)``
2. The plugin schedules an async coroutine per key
3. Inside the coroutine, ``put_blocking(key, buf)`` is called on the
   Rust ``RustRemoteBackend`` PyO3 class
4. Rust acquires the raw pointer from the Python buffer protocol
   (**zero-copy**), then releases the GIL
5. Rust calls ``connector_put`` via the loaded C ABI function pointer
6. The native connector performs the actual I/O (file write, network
   send, etc.)

The ``get`` path is symmetric: Rust calls ``connector_get`` which writes
directly into a pre-allocated PyTorch tensor buffer — again zero-copy.

Connector C ABI
---------------

Every native connector must export the following C functions. The full
header is available at ``rust/remote_backend/include/connector_api.h``.

.. code-block:: c

   /* Create a connector instance from a JSON config string. */
   ConnectorHandle connector_create(
       const char *config_json,
       size_t config_json_len);

   /* Destroy the connector instance. */
   void connector_destroy(ConnectorHandle handle);

   /* Check if key exists. Returns 1=yes, 0=no. */
   int32_t connector_exists(
       ConnectorHandle handle, const char *key);

   /* Write data for key. Returns 0=ok, non-zero=error. */
   int32_t connector_put(
       ConnectorHandle handle, const char *key,
       const uint8_t *data, size_t data_len);

   /* Read data for key into buffer.
    * Returns 0=ok, 1=not-found, -1=error.
    * Writes bytes read into *out_len. */
   int32_t connector_get(
       ConnectorHandle handle, const char *key,
       uint8_t *out_buf, size_t out_cap,
       size_t *out_len);

   /* Remove key. Returns 1=removed, 0=not-found, -1=error. */
   int32_t connector_remove(
       ConnectorHandle handle, const char *key);

   /* Get data size. Returns 0=ok, 1=not-found, -1=error. */
   int32_t connector_file_size(
       ConnectorHandle handle, const char *key,
       uint64_t *out_size);

   /* List all keys as newline-separated UTF-8 in out_buf. */
   int32_t connector_list_keys(
       ConnectorHandle handle,
       char *out_buf, size_t out_cap,
       size_t *out_len);

Lifecycle:

1. ``connector_create(json_cfg, len)`` → opaque handle
2. Any number of ``connector_exists / put / get / remove / list_keys``
   calls (thread-safe is recommended but not required — the Rust
   backend currently serializes calls per key)
3. ``connector_destroy(handle)``

Built-in FS Connector
---------------------

LMCache ships with a built-in filesystem connector
(``rust/connector_fs/``) as a reference implementation. It stores each
KV chunk as a separate file under a configurable directory, using atomic
``rename`` for crash-safety and optional ``O_DIRECT`` for bypassing the
page cache on Linux.

To build it:

.. code-block:: bash

   cd rust/connector_fs
   cargo build --release

The resulting shared library is at
``rust/connector_fs/target/release/liblmcache_connector_fs.so``
(or ``.dylib`` on macOS).

Writing a Custom Connector
--------------------------

A custom connector can be implemented in **any language that can produce
a C-compatible shared library** — C, C++, Rust, or even Zig.

Step 1: Create a Standalone Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your connector lives in its own repository / build system, completely
independent of LMCache.

**Rust example** (``Cargo.toml``):

.. code-block:: toml

   [package]
   name = "my_custom_connector"
   version = "0.1.0"
   edition = "2021"

   [lib]
   crate-type = ["cdylib"]

   [dependencies]
   libc = "0.2"

**C++ example** (``CMakeLists.txt``):

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.16)
   project(my_custom_connector)

   add_library(my_custom_connector SHARED connector.cpp)
   target_include_directories(my_custom_connector PRIVATE
       ${LMCACHE_ROOT}/rust/remote_backend/include)

Step 2: Implement the C ABI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy ``connector_api.h`` from
``rust/remote_backend/include/connector_api.h`` into your project and
implement every function.

**Rust example** (``src/lib.rs``):

.. code-block:: rust

   use std::ffi::CStr;

   struct MyConnector {
       // your state ...
   }

   #[no_mangle]
   pub unsafe extern "C" fn connector_create(
       config_json: *const libc::c_char,
       config_json_len: libc::size_t,
   ) -> *mut libc::c_void {
       let slice = std::slice::from_raw_parts(
           config_json as *const u8, config_json_len,
       );
       let _json = std::str::from_utf8(slice).unwrap();
       // parse config, create your connector ...
       let conn = Box::new(MyConnector { /* ... */ });
       Box::into_raw(conn) as *mut libc::c_void
   }

   #[no_mangle]
   pub unsafe extern "C" fn connector_destroy(
       handle: *mut libc::c_void,
   ) {
       if !handle.is_null() {
           drop(Box::from_raw(handle as *mut MyConnector));
       }
   }

   // ... implement connector_exists, connector_put,
   //     connector_get, connector_remove,
   //     connector_file_size, connector_list_keys ...

**C++ example** (``connector.cpp``):

.. code-block:: cpp

   #include "connector_api.h"
   #include <string>
   #include <unordered_map>

   struct MyConnector {
       std::unordered_map<std::string, std::string> store;
   };

   extern "C" ConnectorHandle connector_create(
       const char *config_json,
       size_t config_json_len
   ) {
       // parse config_json (a UTF-8 JSON string) ...
       return new MyConnector();
   }

   extern "C" void connector_destroy(ConnectorHandle h) {
       delete static_cast<MyConnector*>(h);
   }

   extern "C" int32_t connector_put(
       ConnectorHandle h, const char *key,
       const uint8_t *data, size_t data_len
   ) {
       auto *c = static_cast<MyConnector*>(h);
       c->store[key] = std::string(
           reinterpret_cast<const char*>(data), data_len
       );
       return 0;
   }

   // ... implement remaining functions ...

Step 3: Build and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rust:**

.. code-block:: bash

   cargo build --release
   # Output: target/release/libmy_custom_connector.so

**C++:**

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make
   # Output: libmy_custom_connector.so

Install the ``.so`` / ``.dylib`` to a known path on your system (e.g.,
``/usr/local/lib/`` or alongside your LMCache deployment).

Step 4: Configure LMCache
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tell LMCache to use your connector via the YAML configuration:

.. code-block:: yaml

   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 0.5
   storage_plugins: "rust_remote"
   extra_config:
     # Plugin loader settings
     storage_plugin.rust_remote.module_path: >-
       lmcache.v1.storage_backend.plugins.rust_remote_backend
     storage_plugin.rust_remote.class_name: RustRemoteBackend

     # Path to your connector shared library
     rust_remote.connector_lib: /path/to/libmy_custom_connector.so

     # Connector-specific config (forwarded as JSON)
     # Keys prefixed with "rust_remote.connector." are
     # collected and passed to connector_create().
     rust_remote.connector.base_path: /mnt/cache
     rust_remote.connector.use_odirect: true
     rust_remote.connector.alignment: 4096

All keys starting with ``rust_remote.connector.`` are stripped of that
prefix and packed into a JSON object, which is then passed to
``connector_create()``. For example, the above config results in:

.. code-block:: json

   {
     "base_path": "/mnt/cache",
     "use_odirect": true,
     "alignment": 4096
   }

Configuration Reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Key
     - Required
     - Description
   * - ``rust_remote.connector_lib``
     - Yes
     - Absolute path to the connector ``.so`` / ``.dylib``
   * - ``rust_remote.connector.*``
     - No
     - Connector-specific config forwarded as JSON

Prerequisites
-------------

The Rust remote backend PyO3 extension must be installed:

.. code-block:: bash

   cd rust/remote_backend
   pip install -e .

And the connector shared library must be built and accessible at the
path specified in ``rust_remote.connector_lib``.

Current Limitations
-------------------

- **No batched APIs**: ``batched_get_blocking`` and batched connector
  calls are not implemented. Each key is processed individually.
- **No serde layer**: Data is written and read as raw bytes. The Python
  ``RemoteBackend``'s serializer/deserializer pipeline is not used.
- **No MLA mode**: The ``remote_enable_mla_worker_id_as0`` option is
  not supported.
- **No reconnection logic**: Unlike the Python ``RemoteBackend``, there
  is no automatic reconnection or health monitoring.
- **Metadata is in-memory only**: The shape/dtype metadata cache used
  for ``get_blocking`` reconstruction is not persisted. If the process
  restarts, previously stored data can still be read but the allocation
  shape will fall back to the model's default chunk shape.
