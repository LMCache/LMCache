# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

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
    """Adapter for the ValkeyConnector (``valkey://`` scheme).

    Uses the GLIDE sync client with a ThreadPoolExecutor for
    high-throughput KV cache transfer.  Supports both standalone
    (default) and cluster modes via ``valkey_mode`` config.

    Requires ``valkey-glide`` 2.3+.
    """

    def __init__(self) -> None:
        super().__init__("valkey://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create a ValkeyConnector from the given context.

        Args:
            context: Connector creation context containing URL, config,
                event loop, and local CPU backend.

        Returns:
            A configured ValkeyConnector instance.
        """
        # Local
        from .valkey_connector import (
            DEFAULT_CONNECTION_TIMEOUT_SECS,
            DEFAULT_REQUEST_TIMEOUT_SECS,
            ValkeyConnector,
        )

        config = context.config
        extra_config = (
            config.extra_config
            if config is not None and config.extra_config is not None
            else {}
        )

        num_workers = int(
            extra_config.get(
                "valkey_num_workers",
                extra_config.get("valkey_sync_num_workers", 8),
            )
        )
        username = str(extra_config.get("valkey_username", ""))
        password = str(extra_config.get("valkey_password", ""))
        tls_enable = bool(extra_config.get("tls_enable", False))

        # Timeouts
        request_timeout = float(
            extra_config.get(
                "request_timeout",
                config.blocking_timeout_secs
                if config is not None and config.blocking_timeout_secs is not None
                else DEFAULT_REQUEST_TIMEOUT_SECS,
            )
        )
        connection_timeout = float(
            extra_config.get("connection_timeout", DEFAULT_CONNECTION_TIMEOUT_SECS)
        )

        # Mode: "standalone" (default) or "cluster"
        valkey_mode = str(extra_config.get("valkey_mode", "standalone"))
        cluster_mode = valkey_mode == "cluster"

        # Database ID (standalone only — cluster always uses DB 0)
        database_id: Optional[int] = None
        raw_db = extra_config.get("valkey_database", None)
        if raw_db is not None:
            database_id = int(raw_db)
            if cluster_mode:
                logger.warning(
                    "valkey_database=%s is ignored in cluster mode "
                    "(Valkey cluster always uses DB 0).",
                    database_id,
                )
                database_id = None

        parsed_url = parse_remote_url(context.url)
        logger.info(
            "Creating Valkey connector for %s:%d (mode=%s)",
            parsed_url.host,
            parsed_url.port,
            valkey_mode,
        )
        return ValkeyConnector(
            host=parsed_url.host,
            port=parsed_url.port,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            num_workers=num_workers,
            username=username,
            password=password,
            request_timeout=request_timeout,
            connection_timeout=connection_timeout,
            tls_enable=tls_enable,
            cluster_mode=cluster_mode,
            database_id=database_id,
        )
