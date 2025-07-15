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


class InfinistoreConnectorAdapter(ConnectorAdapter):
    """Adapter for Infinistore connectors."""

    def __init__(self) -> None:
        super().__init__("infinistore://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Third Party
        import infinistore

        # Local
        from .infinistore_connector import InfinistoreConnector

        logger.info(f"Creating Infinistore connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for infinistore, but got {hosts}"
            )

        parse_url = parse_remote_url(context.url)
        device_name = parse_url.query_params.get("device", ["mlx5_0"])[0]

        link_type_str = "LINK_ETHERNET"
        if context.config and context.config.extra_config:
            link_type_str = context.config.extra_config.get(
                "infinistore_link_type", link_type_str
            )

        link_type_str = link_type_str.upper()
        try:
            link_type = getattr(infinistore, link_type_str)
        except AttributeError as e:
            raise ValueError(f"Invalid link_type: {link_type_str}") from e

        return InfinistoreConnector(
            host=parse_url.host,
            port=parse_url.port,
            dev_name=device_name,
            link_type=link_type,
            loop=context.loop,
            memory_allocator=context.local_cpu_backend,
        )
