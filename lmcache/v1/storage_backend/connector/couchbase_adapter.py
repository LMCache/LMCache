# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class CouchbaseConnectorAdapter(ConnectorAdapter):
    """Adapter for Couchbase connectors."""

    def __init__(self) -> None:
        super().__init__("couchbase://")

    def can_parse(self, url: str) -> bool:
        """Check if this adapter can parse the given URL."""
        return url.startswith(self.schema) or url.startswith("couchbases://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create a Couchbase connector using the given context."""
        # Local
        from .couchbase_connector import CouchbaseConnector

        logger.info(f"Creating Couchbase connector for URL: {context.url}")

        # Parse the URL
        parsed_url = parse_remote_url(context.url)

        # Extract connection parameters
        host = parsed_url.host or "localhost"
        port = parsed_url.port or 8091
        username = parsed_url.username
        password = parsed_url.password

        # Extract bucket, scope, and collection from query parameters or use defaults
        query_params = parsed_url.query_params
        bucket_name = query_params.get("bucket", ["default"])[0]
        scope_name = query_params.get("scope", ["_default"])[0]
        collection_name = query_params.get("collection", ["_default"])[0]

        # Support environment variables as fallback
        bucket_name = os.environ.get("COUCHBASE_BUCKET", bucket_name)
        scope_name = os.environ.get("COUCHBASE_SCOPE", scope_name)
        collection_name = os.environ.get("COUCHBASE_COLLECTION", collection_name)

        logger.info(
            f"Connecting to Couchbase: {host}:{port}, "
            f"bucket={bucket_name}, scope={scope_name}, "
            f"collection={collection_name}"
        )

        return CouchbaseConnector(
            host=host,
            port=port,
            username=username,
            password=password,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )
