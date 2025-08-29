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


class RedisConnectorAdapter(ConnectorAdapter):
    """Adapter for Redis connectors."""

    def __init__(self) -> None:
        super().__init__("redis://")

    def can_parse(self, url: str) -> bool:
        return url.startswith((self.schema, "rediss://", "unix://"))

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .redis_connector import RedisConnector

        logger.info(f"Creating Redis connector for URL: {context.url}")
        return RedisConnector(
            url=context.url,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )


class RedisSentinelConnectorAdapter(ConnectorAdapter):
    """Adapter for Redis Sentinel connectors."""

    def __init__(self) -> None:
        super().__init__("redis-sentinel://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .redis_connector import RedisSentinelConnector

        logger.info(f"Creating Redis Sentinel connector for URL: {context.url}")
        url = context.url[len(self.schema) :]

        # Parse username and password
        username: str = ""
        password: str = ""
        if "@" in url:
            auth, url = url.split("@", 1)
            if ":" in auth:
                username, password = auth.split(":", 1)
            else:
                username = auth

        # Parse host and port
        hosts_and_ports: List[Tuple[str, int]] = []
        assert self.schema is not None
        for sub_url in url.split(","):
            if not sub_url.startswith(self.schema):
                sub_url = self.schema + sub_url

            parsed_url = parse_remote_url(sub_url)
            hosts_and_ports.append((parsed_url.host, parsed_url.port))

        return RedisSentinelConnector(
            hosts_and_ports=hosts_and_ports,
            username=username,
            password=password,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )
