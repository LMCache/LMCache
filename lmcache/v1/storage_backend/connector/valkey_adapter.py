# SPDX-License-Identifier: Apache-2.0
# Standard
import urllib.parse
from typing import List, Tuple, Optional

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
    """Unified adapter for all Valkey connection modes (standalone, cluster, sentinel)."""

    def __init__(self) -> None:
        super().__init__("valkey://")

    def can_parse(self, url: str) -> bool:
        """Check if this adapter can handle the given URL."""
        return url.startswith("valkey://")
    
    def _parse_database_from_url(self, url: str) -> Optional[int]:
        database_id: Optional[int] = None
        if "/" in url:
            url, db_part = url.split("/", 1)
            if db_part.isdigit():
                database_id = int(db_part)
        return database_id
        
    def _parse_auth_from_url(self, url: str) -> Tuple[str, str]:
        """Parse username, password from URL"""
        username: str = ""
        password: str = ""
        
        # Parse auth info
        if "@" in url:
            auth, url = url.split("@", 1)
            if ":" in auth:
                username, password = auth.split(":", 1)
            else:
                username = auth
                
        return username, password

    def _parse_cluster_url(self, url: str) -> Tuple[List[Tuple[str, int]], str, str]:
        """Parse cluster URL and return hosts_and_ports list with auth info."""
        
        # Parse auth info
        username, password = self._parse_auth_from_url(url)
        
        # Remove auth info
        if "@" in url:
            _, url = url.split("@", 1)
        
        # Parse host and port
        schema = "valkey://"
        hosts_and_ports: List[Tuple[str, int]] = []
        for sub_url in url.split(","):     
            parsed_url = parse_remote_url(schema + sub_url)
            hosts_and_ports.append((parsed_url.host, parsed_url.port))
        
        return hosts_and_ports, username, password

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create appropriate Valkey connector based on URL parameters."""
        logger.info(f"Creating Valkey connector for URL: {context.url}")
        
        # Parse URL to check for mode parameter
        parsed_url = urllib.parse.urlparse(context.url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        # Check mode parameter
        mode = query_params.get('mode', ['standalone'])[0].lower()
        url = context.url[len(self.schema):].split('?')[0]
        
        if mode == 'cluster':
            from .valkey_connector import ValkeyClusterConnector
            
            logger.info("Creating ValkeyClusterConnector for cluster mode")
            
            try:
                # Check if URL contains comma (multiple hosts) or single host
                netloc = parsed_url.netloc
                if '@' in netloc:
                    netloc = netloc.split('@', 1)[1]  # Take part after @
            
                if ',' in netloc:
                    # Multiple hosts: host1:port1,host2:port2?mode=cluster
                    hosts_and_ports, username, password = self._parse_cluster_url(url)
                else:
                    # Single cluster endpoint: cluster-endpoint:6379?mode=cluster
                    username, password = self._parse_auth_from_url(url)
                    
                    if "@" in url:
                        _, host_part = url.split("@", 1)
                        url = host_part
                    
                    parsed_clean = parse_remote_url(self.schema + url)
                    hosts_and_ports = [(parsed_clean.host, parsed_clean.port)]
                
                if len(hosts_and_ports) < 1:
                    raise ValueError("Cluster mode requires at least one host")
                    
                return ValkeyClusterConnector(
                    hosts_and_ports=hosts_and_ports,
                    username=username,
                    password=password,
                    loop=context.loop,
                    local_cpu_backend=context.local_cpu_backend,
                )
                
            except Exception as e:
                logger.error(f"Failed to parse Valkey cluster URL {context.url}: {e}")
                raise ValueError(f"Invalid Valkey cluster URL: {e}")
        else:
            # Standalone mode - use standalone connector
            from .valkey_connector import ValkeyConnector
            
            logger.info("Creating ValkeyConnector for standalone mode")
        
            # Parse auth info and database for standalone
            username, password = self._parse_auth_from_url(url)
            database_id = self._parse_database_from_url(url)
            
            # Remove auth info: user:pass@host:port/db -> host:port/db
            if "@" in url:
                _, host_part = url.split("@", 1)
                url = host_part
            
            # Remove database: host:port/db -> host:port
            if "/" in url:
                url = url.rsplit("/", 1)[0]
            
            return ValkeyConnector(
                url=url,
                loop=context.loop,
                local_cpu_backend=context.local_cpu_backend,
                username=username,
                password=password,
                database_id=database_id,
            )
