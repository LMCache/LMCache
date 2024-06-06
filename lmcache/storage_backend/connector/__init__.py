import re

from lmcache.storage_backend.connector.base_connector import RemoteConnector, RemoteConnectorDebugWrapper
from lmcache.storage_backend.connector.redis_connector import RedisConnector
from lmcache.storage_backend.connector.lm_connector import LMCServerConnector
from lmcache.config import GlobalConfig

def CreateConnector(url: str) -> RemoteConnector:
    """
    Creates the corresponding remote connector from the given URL.
    """
    m = re.match(r"(.*)://(.*):(\d+)", url)
    if m is None:
        raise ValueError(f"Invalid remote url {config.remote_url}")
    connector_type, host, port = m.group(1), m.group(2), int(m.group(3))
    connector = None

    match connector_type:
        case "redis":
            connector = RedisConnector(host, port)
        case "lm":
            connector = LMCServerConnector(host, port)
        case _:
            raise ValueError(f"Invalid remote url {config.remote_url} -- Unknown connector type {connector_type}")

    return connector if not GlobalConfig.is_debug() else RemoteConnectorDebugWrapper(connector)
