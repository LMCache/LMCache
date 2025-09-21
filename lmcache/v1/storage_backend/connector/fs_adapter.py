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


class FsConnectorAdapter(ConnectorAdapter):
    """Adapter for Filesystem connectors."""

    def __init__(self) -> None:
        super().__init__("fs://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .fs_connector import FSConnector

        logger.info(f"Creating FS connector for URL: {context.url}")
        parse_url = parse_remote_url(context.url)
        relative_tmp_dir = (
            None
            if context.config is None
            else context.config.get_extra_config_value(
                "fs_connector_relative_tmp_dir", None
            )
        )
        return FSConnector(
            parse_url.path, context.loop, context.local_cpu_backend, relative_tmp_dir
        )
