# Hewlett Packard Enterprise Confidential
"""Connector adapter for the RDMA-enabled S3 backend."""

from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import ConnectorAdapter, ConnectorContext

logger = init_logger(__name__)


@dataclass
class S3RdmaConnectorSettings:
    """Configuration payload passed to the RDMA S3 connector."""

    bucket: str
    endpoint: str
    prefix: Optional[str]
    region: Optional[str]
    max_parallel_requests: int
    max_segment_size: Optional[int]
    boto_profile: Optional[str]


class S3RdmaConnectorAdapter(ConnectorAdapter):
    """Adapter enabling URLs of the form ``s3-rdma://``."""

    SCHEMA = "s3-rdma://"

    def __init__(self) -> None:
        super().__init__(self.SCHEMA)

    def create_connector(self, context: ConnectorContext):
        # Local import to avoid importing heavy dependencies during adapter discovery.
        from lmcache.v1.storage_backend.connector.s3_rdma_connector import S3RdmaConnector

        config = context.config
        if config is None:
            raise ValueError("S3 RDMA connector requires LMCache engine configuration")

        settings = self._parse_settings(context.url, config.extra_config)
        logger.info(
            "Creating S3 RDMA connector for bucket '%s' at endpoint '%s'",
            settings.bucket,
            settings.endpoint,
        )

        return S3RdmaConnector(
            settings=settings,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )

    def _parse_settings(
        self,
        url: str,
        extra_config: Optional[dict[str, Any]],
    ) -> S3RdmaConnectorSettings:
        parsed = urlparse(url)
        if parsed.scheme != "s3-rdma":
            raise ValueError(f"Unsupported RDMA S3 URL: {url}")

        bucket = parsed.netloc
        if not bucket:
            raise ValueError("RDMA S3 URL must include a bucket name (netloc)")

        prefix = parsed.path.lstrip("/") or None

        query = parse_qs(parsed.query)

        def pick(name: str, fallback_name: Optional[str] = None) -> Optional[str]:
            if name in query and query[name]:
                return query[name][-1]
            if extra_config is not None:
                if name in extra_config:
                    return str(extra_config[name])
                if fallback_name is not None and fallback_name in extra_config:
                    return str(extra_config[fallback_name])
            return None

        endpoint = pick("endpoint", "s3_rdma_endpoint")
        if endpoint is None:
            raise ValueError(
                "RDMA S3 connector requires an 'endpoint' parameter either in the URL query "
                "or LMCache extra_config under 's3_rdma_endpoint'."
            )

        region = pick("region", "s3_region")
        boto_profile = pick("profile", "aws_profile")

        max_parallel_requests = _coerce_positive_int(
            pick("max_parallel_requests", "s3_rdma_max_parallel_requests"), default=32
        )
        max_segment_size = _coerce_optional_positive_int(
            pick("max_segment_size", "s3_rdma_max_segment_size")
        )

        return S3RdmaConnectorSettings(
            bucket=bucket,
            endpoint=endpoint,
            prefix=prefix,
            region=region,
            max_parallel_requests=max_parallel_requests,
            max_segment_size=max_segment_size,
            boto_profile=boto_profile,
        )


def _coerce_positive_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Expected integer value, got '{value}'") from exc
    if parsed <= 0:
        raise ValueError(f"Expected positive integer, got {parsed}")
    return parsed


def _coerce_optional_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    parsed = _coerce_positive_int(value, default=1)
    return parsed
