# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class S3ConnectorAdapter(ConnectorAdapter):
    """Adapter for S3 Server connectors."""

    def __init__(self) -> None:
        super().__init__("s3://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .s3_connector import S3Connector

        config = context.config
        assert config is not None

        if config.extra_config is not None:
            # Different parts can be transferred in parallel.
            self.s3_part_size = config.extra_config.get("s3_part_size", None)
            self.s3_max_io_concurrency = config.extra_config.get(
                "s3_max_io_concurrency", 64
            )
            self.s3_max_inflight_reqs = config.extra_config.get(
                "s3_max_inflight_reqs", 64
            )
            self.s3_prefer_http2 = config.extra_config.get("s3_prefer_http2", True)
            self.s3_region = config.extra_config.get("s3_region", None)
            self.s3_enable_s3express = config.extra_config.get(
                "s3_enable_s3express", True
            )
            self.s3_file_prefix = config.extra_config.get("s3_file_prefix", None)

        logger.info(f"Creating S3 connector for URL: {context.url}")

        s3_endpoint = context.url

        return S3Connector(
            s3_endpoint=s3_endpoint,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            s3_part_size=self.s3_part_size,
            s3_file_prefix=self.s3_file_prefix,
            s3_max_io_concurrency=self.s3_max_io_concurrency,
            s3_max_inflight_reqs=self.s3_max_inflight_reqs,
            s3_prefer_http2=self.s3_prefer_http2,
            s3_region=self.s3_region,
            s3_enable_s3express=self.s3_enable_s3express,
        )
