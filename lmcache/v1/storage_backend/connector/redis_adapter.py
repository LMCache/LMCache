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


class RESPConnectorAdapter(ConnectorAdapter):
    """Adapter for RESP connectors."""

    def __init__(self) -> None:
        super().__init__("resp://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .redis_connector import RESPConnector

        config = context.config
        assert config is not None

        # Get config from extra_config with defaults
        extra_config = config.extra_config if config.extra_config is not None else {}

        # Validate that save_chunk_meta and save_unfull_chunk are False for RESP
        self.save_chunk_meta = bool(extra_config.get("save_chunk_meta", False))
        assert not self.save_chunk_meta, "save_chunk_meta must be False for RESP"

        assert not config.save_unfull_chunk, "save_unfull_chunk must be False for RESP"

        # Get number of threads for RESP connection pool (default is 8)
        self.resp_num_threads = int(extra_config.get("resp_num_threads", 8))

        # Get authentication credentials from extra_config
        username = str(extra_config.get("username", ""))
        password = str(extra_config.get("password", ""))

        logger.info(f"Creating RESP connector for URL: {context.url}")
        parsed_url = parse_remote_url(context.url)
        return RESPConnector(
            host=parsed_url.host,
            port=parsed_url.port,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            num_threads=self.resp_num_threads,
            username=username,
            password=password,
        )


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


class RedisClusterConnectorAdapter(ConnectorAdapter):
    """Adapter for Redis Cluster connectors."""

    def __init__(self) -> None:
        super().__init__("redis-cluster://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .redis_connector import RedisClusterConnector

        logger.info(f"Creating Redis Cluster connector for URL: {context.url}")
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

        return RedisClusterConnector(
            hosts_and_ports=hosts_and_ports,
            username=username,
            password=password,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )
