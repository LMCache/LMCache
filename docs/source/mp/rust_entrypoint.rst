Experimental Rust Entry Point
=============================

``lmcache-server`` embeds CPython and starts the existing ``lmcache server``
command in the same process. Request handling, storage management and shutdown
continue to use Python. This is an initial bootstrap for native server work.

Build from source on Linux or macOS with Cargo and CPython 3.12 or 3.13.
Install the full LMCache server in the selected Python environment first,
following :doc:`../getting_started/installation`. The interpreter must provide
development libraries and a linkable libpython. From the repository root:

.. code-block:: bash

   export PYO3_PYTHON="$(command -v python)"
   cargo build --release --locked --manifest-path rust/server/Cargo.toml
   rust/server/target/release/lmcache-server --help
   rust/server/target/release/lmcache-server --host 127.0.0.1 --port 5555

Use the same flags as ``lmcache server``. Existing Python entry points remain
available. This binary requires the selected interpreter and environment at
their build-time locations; rebuild it to select a different environment.
It is not a standalone or relocatable distribution.
