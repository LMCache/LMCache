# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
import asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector
from lmcache.v1.storage_backend.connector.redis_connector import (
    RedisConnector,
    RedisSentinelConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# Local
from .audit_connector import AuditConnector
from .blackhole_connector import BlackholeConnector
from .fs_connector import FSConnector
from .infinistore_connector import InfinistoreConnector
from .instrumented_connector import InstrumentedRemoteConnector
from .mooncakestore_connector import MooncakestoreConnector

logger = init_logger(__name__)


@dataclass
class ParsedRemoteURL:
    """
    The parsed URL of the format:
        <host>:<port>[/path][?query]
    """

    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    path: Optional[str] = None
    query_params: Dict[str, List[str]] = field(default_factory=dict)


def parse_remote_url(url: str) -> ParsedRemoteURL:
    """
    Parses the remote URL into its constituent parts with support for:
    - Multiple hosts (comma-separated)
    - Path and query parameters in each host definition
    - Forward compatibility with legacy format

    Args:
        url: The URL to parse

    Returns:
        ParsedRemoteURL: The parsed URL components

    Raises:
        ValueError: If the URL is invalid
    """

    logger.debug(f"Parsing remote URL: {url}")
    parsed = urlparse(url)

    username = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port
    path = parsed.path if parsed.path else ""
    query = parse_qs(parsed.query) if parsed.query else {}

    return ParsedRemoteURL(
        host=host,
        port=port,
        path=path,
        username=username,
        password=password,
        query_params=query,
    )


class ConnectorContext:
    """
    Context for creating a connector.

    Attributes:
        url: The remote URL
        loop: The asyncio event loop
        local_cpu_backend: The local CPU backend
        config: Optional LMCache engine configuration
        parsed_url: Parsed representation of the URL
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig],
    ):
        self.url = url
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.config = config


class ConnectorAdapter(ABC):
    """Base class for connector adapters."""

    def __init__(self, schema: str):
        self.schema = schema

    @abstractmethod
    def can_parse(self, url: str) -> bool:
        """
        Check if this adapter can parse the given URL.
        """
        pass

    @abstractmethod
    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create a connector using the given context.
        """
        pass


class RedisConnectorAdapter(ConnectorAdapter):
    """Adapter for Redis connectors."""

    def __init__(self) -> None:
        super().__init__("redis://")

    def can_parse(self, url: str) -> bool:
        # The Redis adaptor is also applicable to the URL format of rediss.
        return url.startswith((self.schema, "rediss://", "unix://"))

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create a Redis connector.

        URL formats:
        - redis://[[username]:[password]@]host[:port][/database][?option=value]
        - rediss://[[username]:[password]@]host[:port][/database][?option=value] (SSL)
        - unix://[username@]/path/to/socket.sock?db=0[&password=password]

        Examples:
        - redis://localhost:6379
        - redis://user:password@redis.example.com:6380/0
        - redis://:password@localhost:6379/1
        - rediss://user:password@redis.example.com:6379?ssl_cert_reqs=CERT_REQUIRED
        """
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
        """
        Create a Redis Sentinel connector.

        URL format:
        - redis-sentinel://[username:password@]host1:port1[,host2:port2,...]/service_name
        """
        logger.info(f"Creating Redis connector for URL: {context.url}")
        url = context.url[len(self.schema) :]

        # parse username and password from url.
        username: str = ""
        password: str = ""
        if "@" in url:
            auth, url = url.split("@", 1)
            if ":" in auth:
                username, password = auth.split(":", 1)
            else:
                username = auth

        # parse host and port from url.
        hosts_and_ports: List[Tuple[str, int]] = []
        for sub_url in url.split(","):
            # add a schema to parse the url correctly.
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


class LMServerConnectorAdapter(ConnectorAdapter):
    """Adapter for LM Server connectors."""

    def __init__(self) -> None:
        super().__init__("lm://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create an LM Server connector.
        URL format:
        - lm://host:port
        """
        logger.info(f"Creating LM Server connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for lm server, but got {hosts}"
            )

        parse_url = parse_remote_url(context.url)
        return LMCServerConnector(
            host=parse_url.host,
            port=parse_url.port,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )


class InfinistoreConnectorAdapter(ConnectorAdapter):
    """Adapter for Infinistore connectors."""

    def __init__(self) -> None:
        super().__init__("infinistore://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create an Infinistore connector.

        URL format:
        - infinistore://host:port[?device=device_name]

        Examples:
        - infinistore://127.0.0.1:12345
        - infinistore://infinistore-server.example.com:12345?device=mlx5_0
        - infinistore://10.0.0.1:12345?device=custom_device
        """
        logger.info(f"Creating Infinistore connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for infinistore, but got {hosts}"
            )

        parse_url = parse_remote_url(context.url)
        device_name = parse_url.query_params.get("device", ["mlx5_0"])[0]
        return InfinistoreConnector(
            host=parse_url.host,
            port=parse_url.port,
            dev_name=device_name,
            loop=context.loop,
            memory_allocator=context.local_cpu_backend,
        )


class MooncakestoreConnectorAdapter(ConnectorAdapter):
    """Adapter for Mooncakestore connectors."""

    def __init__(self) -> None:
        super().__init__("mooncakestore://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create a Mooncakestore connector.

        URL format:
        - mooncakestore://host:port[?device=device_name]

        Examples:
        - mooncakestore://127.0.0.1:50051
        - mooncakestore://mooncake-server.example.com:50051?device=custom_device
        """
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
            config=context.config,
        )


class BlackholeConnectorAdapter(ConnectorAdapter):
    """Adapter for Blackhole connectors (for testing)."""

    def __init__(self) -> None:
        super().__init__("blackhole://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create a Blackhole connector. This connector is used for testing
        and discards all data.

        URL format:
        - blackhole://[any_text]

        Examples:
        - blackhole://127.0.0.1:8080
        """
        logger.info(f"Creating Blackhole connector for URL: {context.url}")
        return BlackholeConnector()


class AuditConnectorAdapter(ConnectorAdapter):
    """Adapter for Audit connectors (for debugging and verification)."""

    def __init__(self) -> None:
        super().__init__("audit://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create an Audit connector. This connector wraps another connector
        and audits all operations.

        URL format:
        - audit://host:port[?verify=true|false]

        Examples:
        - audit://localhost:8080
        - audit://audit-server.example.com:8080?verify=true
        - audit://127.0.0.1:8080?verify=false
        """
        logger.info(f"Creating Audit connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for audit connector, but got {hosts}"
            )

        if not context.config or not context.config.audit_actual_remote_url:
            raise ValueError("audit_actual_remote_url is not set in the config")

        parse_url = parse_remote_url(context.url)
        verify_param = parse_url.query_params.get("verify", ["false"])[0]
        verify_checksum = verify_param.lower() in ("true", "1", "yes")
        real_url = context.config.audit_actual_remote_url
        connector = CreateConnector(
            real_url, context.loop, context.local_cpu_backend, context.config
        )
        return AuditConnector(connector, verify_checksum)


class FsConnectorAdapter(ConnectorAdapter):
    """Adapter for Filesystem connectors."""

    def __init__(self) -> None:
        super().__init__("fs://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create a Filesystem connector. This connector stores data
        in the local filesystem.

        URL format:
        fs://[host:port]/path

        Examples:
        - fs:///tmp/lmcache
        - fs://localhost:0/var/lib/lmcache
        - fs://127.0.0.1:0/home/user/lmcache_data

        Note: The host:port part is optional and ignored. The path is
        the important part.
        """
        logger.info(f"Creating FS connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for fs connector, but got {hosts}"
            )

        parse_url = parse_remote_url(context.url)
        assert parse_url.path, "Path is required for fs connector"

        if not parse_url.path.startswith("/"):
            parse_url.path = "/" + parse_url.path

        return FSConnector(parse_url.path, context.loop, context.local_cpu_backend)


class ConnectorManager:
    """
    Manager for creating connectors based on URL.

    This class maintains a registry of connector adapters and creates
    the appropriate connector based on the URL.
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig] = None,
    ) -> None:
        self.context = ConnectorContext(
            url=url,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            config=config,
        )
        self.adapters: List[ConnectorAdapter] = []
        self._register_adapters()

    def _register_adapters(self) -> None:
        """Register all available connector adapters."""

        self.adapters.append(RedisConnectorAdapter())
        self.adapters.append(RedisSentinelConnectorAdapter())
        self.adapters.append(LMServerConnectorAdapter())
        self.adapters.append(InfinistoreConnectorAdapter())
        self.adapters.append(MooncakestoreConnectorAdapter())
        self.adapters.append(BlackholeConnectorAdapter())
        self.adapters.append(AuditConnectorAdapter())
        self.adapters.append(FsConnectorAdapter())

    def create_connector(self) -> RemoteConnector:
        for adapter in self.adapters:
            if adapter.can_parse(self.context.url):
                return adapter.create_connector(self.context)

        raise ValueError(f"No adapter found for URL: {self.context.url}")


def CreateConnector(
    url: str,
    loop: asyncio.AbstractEventLoop,
    local_cpu_backend: LocalCPUBackend,
    config: Optional[LMCacheEngineConfig] = None,
) -> Optional[RemoteConnector]:
    """
    Create a remote connector from the given URL.

    Supported URL formats:
    - redis://[[username]:[password]@]host[:port][/database][?option=value]
    - rediss://[[username]:[password]@]host[:port][/database][?option=value] (SSL)
    - redis-sentinel://[[username]:[password]@]host1:port1[,host2:port2,...]/service_name
    - lm://host:port
    - infinistore://host:port[?device=device_name]
    - mooncakestore://host:port[?device=device_name]
    - blackhole://[any_text]
    - audit://host:port[?verify=true|false]
    - fs://[host:port]/path

    Examples:
    - redis://localhost:6379
    - rediss://user:password@redis.example.com:6380/0
    - redis-sentinel://user:password@sentinel1:26379,sentinel2:26379/mymaster
    - lm://localhost:65432
    - infinistore://127.0.0.1:12345?device=mlx5_0
    - mooncakestore://127.0.0.1:50051
    - blackhole://
    - audit://localhost:8080?verify=true
    - fs:///tmp/lmcache

    Args:
        url: The remote URL
        loop: The asyncio event loop
        local_cpu_backend: The local CPU backend
        config: Optional LMCache engine configuration

    Returns:
        RemoteConnector: The created connector

    Raises:
        ValueError: If the connector cannot be created
    """

    # Basic URL validation - check for scheme
    if "://" not in url:
        raise ValueError(f"Invalid remote url {url}: missing scheme")

    manager = ConnectorManager(url, loop, local_cpu_backend, config)
    connector = manager.create_connector()

    return InstrumentedRemoteConnector(connector)
