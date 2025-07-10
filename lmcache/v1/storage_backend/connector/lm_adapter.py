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
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class LMServerConnectorAdapter(ConnectorAdapter):
    """Adapter for LM Server connectors."""

    def __init__(self) -> None:
        super().__init__("lm://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .lm_connector import LMCServerConnector

        logger.info(f"Creating LM Server connector for URL: {context.url}")
        parse_url = parse_remote_url(context.url)
        return LMCServerConnector(
            host=parse_url.host,
            port=parse_url.port,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )
