# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    extract_plugin_type,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

# Local
from .gcs_connector import GCSConnector, resolve_gcs_connector_config

logger = init_logger(__name__)

PLUGIN_TYPE = "gcs"


class GCSConnectorAdapter(ConnectorAdapter):
    """Adapter for Google Cloud Storage remote connectors."""

    def __init__(self) -> None:
        super().__init__("plugin://")

    def can_parse(self, url: str) -> bool:
        """Match plugin URLs for the built-in ``gcs`` connector type."""
        if not url.startswith(self.schema):
            return False
        plugin_name = url[len(self.schema) :]
        return extract_plugin_type(plugin_name) == PLUGIN_TYPE

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create a configured ``GCSConnector`` for the given context."""
        if context.config is None:
            raise ValueError("config is required for GCSConnector")
        if context.metadata is None:
            raise ValueError("metadata is required for GCSConnector")

        plugin_name = context.plugin_name or PLUGIN_TYPE
        connector_config = resolve_gcs_connector_config(
            context.config,
            plugin_name=plugin_name,
        )
        logger.info(
            "Creating GCS connector for plugin %s and bucket %s",
            plugin_name,
            connector_config.bucket_location.bucket_name,
        )
        return GCSConnector(
            local_cpu_backend=context.local_cpu_backend,
            config=context.config,
            metadata=context.metadata,
            connector_config=connector_config,
        )
