Remote Storage Plugins
========================

LMCache supports built-in remote storage connectors for Redis, InfiniStore, MooncakeStore, S3, and more.
The remote storage plugin system provides the ability to add custom storage connectors through dynamic loading. This enables extending remote storage capabilities without modifying core code.

Connector Definition Requirements
---------------------------------
A custom remote storage connector requires two classes:

1. **ConnectorAdapter**: Handles URL scheme matching and connector instantiation

   - Inherit from ``ConnectorAdapter``
   - Set the URL scheme in the constructor (e.g., ``mystore://``)
   - Implement the ``create_connector`` method

2. **RemoteConnector**: Implements the actual storage operations

   - Inherit from ``RemoteConnector``
   - Implement all abstract methods: ``exists``, ``exists_sync``, ``get``, ``put``, ``list``, ``close``

.. note::

   The ``ConnectorAdapter`` constructor receives no arguments from LMCache. The scheme should be set by calling the parent constructor with the scheme string.

   The ``create_connector`` method receives a ``ConnectorContext`` object containing the URL, event loop, local CPU backend, config, and metadata.

How to Integrate Remote Storage with LMCache
----------------------------------------------
1. Install your connector package in the LMCache environment
2. Add ``remote_storage_plugins`` and its related ``module_path`` and ``class_name`` to the ``extra_config`` section of LMCache configuration as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    remote_url: "mystore://localhost:8080"
    remote_storage_plugins: ["mystore"]
    extra_config:
      remote_storage_plugin.mystore.module_path: <module_path>
      remote_storage_plugin.mystore.class_name: <adapter_class_name>

An example configuration for a custom remote storage connector is as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    remote_url: "mystore://localhost:8080/data"
    remote_storage_plugins: ["mystore"]
    extra_config:
      remote_storage_plugin.mystore.module_path: my_package.my_connector
      remote_storage_plugin.mystore.class_name: MyStoreConnectorAdapter

.. note::

   - The ``remote_url`` scheme must match the scheme registered by your ``ConnectorAdapter``
   - ``remote_storage_plugin.<connector_name>`` distinguishes different dynamically loaded connectors
   - Multiple remote storage plugins can be loaded simultaneously

ConnectorAdapter Implementation
-------------------------------
The ``ConnectorAdapter`` class is responsible for:

- Defining the URL scheme it handles
- Creating the appropriate ``RemoteConnector`` instance

.. code-block:: python

    from lmcache.v1.storage_backend.connector import (
        ConnectorAdapter,
        ConnectorContext,
    )
    from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

    class MyStoreConnectorAdapter(ConnectorAdapter):
        """Adapter for MyStore remote storage."""

        def __init__(self) -> None:
            # Register the URL scheme this adapter handles
            super().__init__("mystore://")

        def create_connector(self, context: ConnectorContext) -> RemoteConnector:
            """Create and return a MyStoreConnector instance."""
            # Access context properties as needed:
            # - context.url: the full remote URL
            # - context.loop: asyncio event loop
            # - context.config: LMCacheEngineConfig
            # - context.metadata: LMCacheEngineMetadata
            return MyStoreConnector(context.config, context.metadata)

RemoteConnector Implementation
------------------------------
The ``RemoteConnector`` class defines the interface for remote storage operations. Your implementation must provide the following abstract methods:

.. code-block:: python

    from typing import List, Optional
    from lmcache.config import LMCacheEngineMetadata
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

    class MyStoreConnector(RemoteConnector):
        """Custom connector for MyStore remote storage."""

        def __init__(
            self,
            config: LMCacheEngineConfig,
            metadata: Optional[LMCacheEngineMetadata]
        ):
            super().__init__(config, metadata)
            # Initialize your connection here

        async def exists(self, key: CacheEngineKey) -> bool:
            """Check if a key exists in the remote store."""
            raise NotImplementedError

        def exists_sync(self, key: CacheEngineKey) -> bool:
            """Synchronous version of exists."""
            raise NotImplementedError

        async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
            """Retrieve a memory object by key. Return None if not found."""
            raise NotImplementedError

        async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
            """Store a memory object with the given key."""
            raise NotImplementedError

        async def list(self) -> List[str]:
            """List all keys in the remote store."""
            raise NotImplementedError

        async def close(self):
            """Close the connection to the remote store."""
            raise NotImplementedError

Optional Methods
----------------
The ``RemoteConnector`` base class also provides optional methods that can be overridden for enhanced functionality:

- ``support_ping()`` / ``ping()``: Health check support
- ``support_batched_get()`` / ``batched_get()``: Batch retrieval operations
- ``support_batched_put()`` / ``batched_put()``: Batch storage operations
- ``support_batched_contains()`` / ``batched_contains()``: Batch existence checks
- ``remove_sync()``: Synchronous key removal

Implementation Example
----------------------
A complete remote storage connector implementation would include both the adapter and connector classes in a single module or package. Here's a minimal working example structure:

.. code-block:: text

    my_connector_package/
    ├── __init__.py
    ├── adapter.py      # Contains MyStoreConnectorAdapter
    └── connector.py    # Contains MyStoreConnector

The adapter module (``adapter.py``):

.. code-block:: python

    from lmcache.v1.storage_backend.connector import (
        ConnectorAdapter,
        ConnectorContext,
    )
    from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
    from .connector import MyStoreConnector

    class MyStoreConnectorAdapter(ConnectorAdapter):
        def __init__(self) -> None:
            super().__init__("mystore://")

        def create_connector(self, context: ConnectorContext) -> RemoteConnector:
            return MyStoreConnector(
                context.url,
                context.config,
                context.metadata
            )

Configuration would then reference the adapter:

.. code-block:: yaml

    remote_storage_plugins: ["mystore"]
    extra_config:
      remote_storage_plugin.mystore.module_path: my_connector_package.adapter
      remote_storage_plugin.mystore.class_name: MyStoreConnectorAdapter
