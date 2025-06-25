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

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    CreateConnector,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class AuditConnectorAdapter(ConnectorAdapter):
    """Adapter for Audit connectors (for debugging and verification)."""

    def __init__(self) -> None:
        super().__init__("audit://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .audit_connector import AuditConnector

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
