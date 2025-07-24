# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class MooncakestoreConnectorAdapter(ConnectorAdapter):
    """Adapter for Mooncakestore connectors."""

    def __init__(self) -> None:
        super().__init__("mooncakestore://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .mooncakestore_connector import MooncakestoreConnector

        logger.info(f"Creating Mooncakestore connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for mooncakestore, but got {hosts}"
            )

        parse_url = parse_remote_url(context.url)
        device_name = parse_url.query_params.get("device", [""])[0]
        return MooncakestoreConnector(
            host=parse_url.host,
            port=parse_url.port,
            dev_name=device_name,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            lmcache_config=context.config,
        )
