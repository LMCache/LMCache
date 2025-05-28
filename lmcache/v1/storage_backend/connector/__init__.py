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
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
import re

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
    <connector_type>://<host>:<port>[/path][?query],<host2>:<port2>[/path2][?query2],...
    """

    connector_type: str
    hosts: List[str]
    ports: List[int]
    paths: List[str]
    query_params: List[Dict[str, str]]


def parse_remote_url(url: str) -> ParsedRemoteURL:
    """
    Parses the remote URL into its constituent parts with support for:
    - Multiple hosts (comma-separated)
    - Path and query parameters in each host definition
    - Forward compatibility with legacy format

    Raises:
        ValueError: If the URL is invalid.
    """
    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    if m is None:
        logger.error(f"Cannot parse remote_url {url} in the config")
        raise ValueError(f"Invalid remote url {url}")

    connector_type, hosts_section = m.groups()

    hosts = []
    ports = []
    paths = []
    query_params = []

    for host_def in hosts_section.split(","):
        host_pattern = r"""
                ^
                ([^:]+)        # hostname
                :              # :
                (\d+)          # port
                (/?[^?]*)      # path（optional, start with /）
                (?:\?(.*))?    # query（optional，? content after ?）
                $
            """
        match = re.match(host_pattern, host_def, re.VERBOSE)

        if not match:
            raise ValueError(f"Invalid host definition: {host_def} in URL: {url}")

        host = match.group(1)
        port = int(match.group(2))
        path = match.group(3).lstrip("/")
        path = path.lstrip("/")
        query_str = match.group(4) or ""

        params_dict = {}
        if query_str:
            for param in query_str.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params_dict[key] = value
                elif param:
                    params_dict[param] = ""

        hosts.append(host)
        ports.append(port)
        paths.append(path)
        query_params.append(params_dict)

    return ParsedRemoteURL(
        connector_type=connector_type,
        hosts=hosts,
        ports=ports,
        paths=paths,
        query_params=query_params,
    )


def CreateConnector(
    url: str,
    loop: asyncio.AbstractEventLoop,
    local_cpu_backend: LocalCPUBackend,
    config: Optional[LMCacheEngineConfig] = None,
) -> Optional[RemoteConnector]:
    """
    Creates the corresponding remote connector from the given URL.
    """
    m = re.match(r"(.*)://(.*):(\d+)", url)
    if m is None:
        raise ValueError(f"Invalid remote url {url}")

    parsed_url = parse_remote_url(url)
    num_hosts = len(parsed_url.hosts)

    connector: Optional[RemoteConnector] = None
    connector_type = parsed_url.connector_type
    match connector_type:
        case "redis":
            if num_hosts == 1:
                host, port = parsed_url.hosts[0], parsed_url.ports[0]
                connector = RedisConnector(host, port, loop, local_cpu_backend)
            else:
                raise ValueError(
                    f"Redis connector only supports a single host, but got url: {url}"
                )

        case "redis-sentinel":
            connector = RedisSentinelConnector(
                list(
                    zip(
                        parsed_url.hosts,
                        map(int, parsed_url.ports),
                        strict=False,
                    )
                ),
                loop,
                local_cpu_backend,
            )

        case "lm":
            if num_hosts == 1:
                host, port = parsed_url.hosts[0], parsed_url.ports[0]
                connector = LMCServerConnector(host, port, loop, local_cpu_backend)
            else:
                raise ValueError(
                    f"LM connector only supports a single host, but got url: {url}"
                )
        case "infinistore":
            host, port = parsed_url.hosts[0], parsed_url.ports[0]
            device_name = parsed_url.query_params[0].get("device", "mlx5_0")
            connector = InfinistoreConnector(host, port, device_name, loop)
        case "mooncakestore":
            host, port = parsed_url.hosts[0], parsed_url.ports[0]
            device_name = parsed_url.query_params[0].get("device", "")
            connector = MooncakestoreConnector(
                host, port, device_name, loop, local_cpu_backend
            )
        case "blackhole":
            connector = BlackholeConnector()
        case "fs":
            if num_hosts != 1:
                raise ValueError(
                    f"FS connector only supports a single path, but got url:{url}"
                )
            # For fs connector path is the base path of the url
            base_path = parsed_url.paths[0]
            # Ensure path starts with '/'
            if not base_path.startswith("/"):
                base_path = "/" + base_path
            connector = FSConnector(base_path, loop, local_cpu_backend)
        case "audit":
            if num_hosts != 1:
                raise ValueError(
                    f"Audit connector only supports a single host, but got url: {url}"
                )
            if not config or not config.audit_actual_remote_url:
                raise ValueError(
                    "Audit connector requires audit_actual_remote_url in config"
                )
            real_url = config.audit_actual_remote_url
            verify_checksum = parsed_url.query_params[0].get("verify") is not None
            logger.info(f"Creating audit connector for {real_url}")
            real_connector = CreateConnector(real_url, loop, local_cpu_backend)
            assert real_connector is not None
            return AuditConnector(
                real_connector=real_connector, verify_checksum=verify_checksum
            )
        case _:
            raise ValueError(f"Unknown connector type {connector_type} (url is: {url})")

    logger.info(f"Created connector {connector} for {connector_type}")
    return InstrumentedRemoteConnector(connector)
