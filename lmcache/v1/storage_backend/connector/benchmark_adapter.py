# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

# Standard
from urllib.parse import urlparse, parse_qs

logger = init_logger(__name__)


class BenchmarkConnectorAdapter(ConnectorAdapter): 
    """Adapter for Benchmark Connector"""

    def __init__(self) -> None: 
        super().__init__("benchmark://")
    
    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)
    
    def create_connector(self, context: ConnectorContext) -> RemoteConnector: 
        # Local import to avoid circular dependencies
        from .benchmark_connector import BenchmarkConnector

        logger.info(f"Creating Benchmark connector for URL: {context.url}")

        parsed = urlparse(context.url)
        # capacity is provided as the netloc in URLs like: benchmark://100/?...
        if not parsed.netloc:
            raise ValueError(
                "benchmark connector requires capacity in GB as netloc, e.g. benchmark://100/?..."
            )
        try:
            capacity_gb = int(parsed.netloc)
        except ValueError:
            raise ValueError(
                f"Invalid capacity '{parsed.netloc}' for benchmark connector; must be an integer (GB)."
            )

        params = parse_qs(parsed.query) if parsed.query else {}
        # Defaults
        peeking_latency_ms = float(
            params.get("peeking_latency", ["1"])[0]
        )
        read_throughput_gbps = float(
            params.get("read_throughput", ["2"])[0]
        )
        write_throughput_gbps = float(
            params.get("write_throughput", ["2"])[0]
        )

        return BenchmarkConnector(
            url=context.url,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            capacity=capacity_gb,
            read_throughput=read_throughput_gbps,
            write_throughput=write_throughput_gbps,
            peeking_latency=peeking_latency_ms,
        ) 
        