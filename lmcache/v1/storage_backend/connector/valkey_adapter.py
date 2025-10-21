# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class ValkeyConnectorAdapter(ConnectorAdapter):
    """Adapter for Valkey Server connectors."""

    def __init__(self) -> None:
        super().__init__("valkey://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .valkey_connector import ValkeyClusterConnector, ValkeyConnector

        config = context.config

        if config is not None and config.extra_config is not None:
            self.valkey_username = config.extra_config.get("valkey_username", "")
            self.valkey_password = config.extra_config.get("valkey_password", "")
            self.valkey_database = config.extra_config.get("valkey_database", None)
            self.valkey_mode = config.extra_config.get("valkey_mode", "standalone")
        else:
            self.valkey_username = ""
            self.valkey_password = ""
            self.valkey_database = None
            self.valkey_mode = "standalone"

        logger.info(f"Creating Valkey connector for URL: {context.url}")

        if self.valkey_mode == "cluster":
            hosts_and_ports: List[Tuple[str, int]] = []
            assert self.schema is not None
            for sub_url in context.url.split(","):
                if not sub_url.startswith(self.schema):
                    sub_url = self.schema + sub_url

                parsed_url = parse_remote_url(sub_url)
                hosts_and_ports.append((parsed_url.host, parsed_url.port))

            return ValkeyClusterConnector(
                hosts_and_ports=hosts_and_ports,
                loop=context.loop,
                local_cpu_backend=context.local_cpu_backend,
                username=self.valkey_username,
                password=self.valkey_password,
            )
        else:
            url = context.url[len(self.schema) :]
            return ValkeyConnector(
                url=url,
                loop=context.loop,
                local_cpu_backend=context.local_cpu_backend,
                username=self.valkey_username,
                password=self.valkey_password,
                database_id=self.valkey_database,
            )
