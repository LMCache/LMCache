import re
from dataclasses import dataclass
from typing import Optional, List

from lmcache.storage_backend.connector.base_connector import RemoteConnector, RemoteConnectorDebugWrapper
from lmcache.storage_backend.connector.redis_connector import RedisConnector, RedisSentinelConnector
from lmcache.storage_backend.connector.lm_connector import LMCServerConnector
from lmcache.config import GlobalConfig

from lmcache.logging import init_logger
logger = init_logger(__name__)

@dataclass
class ParsedRemoteURL:
    """
    The parsed URL of the format:
    <connector_type>://<host>:<port>,<host2>:<port2>,...
    """
    connector_type: str
    hosts: List[str]
    ports: List[int]

def parse_remote_url(url: str) -> ParsedRemoteURL:
    """
    Parses the remote URL into its constituent parts.

    Raises:
        ValueError: If the URL is invalid.
    """
    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    if m is None:
        logger.error(f"Cannot parse remote_url {url} in the config")
        raise ValueError(f"Invalid remote url {url}")

    connector_type, hosts_and_ports = m.group(1), m.group(2)

    hosts = []
    ports = []
    for body in hosts_and_ports.split(","):
        m = re.match(r"(.+):(\d+)", body)
        if m is None:
            logger.error(f"Cannot parse url body {body} from remote_url {url} in the config")
            raise ValueError(f"Invalid remote url {url}")

        host, port = m.group(1), int(m.group(2))
        hosts.append(host)
        ports.append(port)

    return ParsedRemoteURL(connector_type, hosts, ports)


def CreateConnector(url: str) -> RemoteConnector:
    """
    Creates the corresponding remote connector from the given URL.
    """
    m = re.match(r"(.*)://(.*):(\d+)", url)
    if m is None:
        raise ValueError(f"Invalid remote url {url}")

    parsed_url = parse_remote_url(url)
    num_hosts = len(parsed_url.hosts)

    connector = None

    match parsed_url.connector_type:
        case "redis":
            if num_hosts == 1:
                host, port = parsed_url.hosts[0], parsed_url.ports[0]
                connector = RedisConnector(host, port)
            else:
                raise ValueError(f"Redis connector only supports a single host, but got url: {url}")

        case "redis-sentinel":
            connector = RedisSentinelConnector(list(zip(parsed_url.hosts, parsed_url.ports)))

        case "lm":
            if num_hosts == 1:
                host, port = parsed_url.hosts[0], parsed_url.ports[0]
                connector = LMCServerConnector(host, port)
            else:
                raise ValueError(f"LM connector only supports a single host, but got url: {url}")

        case _:
            raise ValueError(f"Unknown connector type {connector_type} (url is: {url})")

    return connector if not GlobalConfig.is_debug() else RemoteConnectorDebugWrapper(connector)

