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


class ExternalConnectorAdapter(ConnectorAdapter):
    """Adapter for External connectors."""

    def __init__(self) -> None:
        super().__init__("external://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create an External connector. This connector stores data
        in the key-value store.
        URL format:
        - external://host:port/module_path/?connector_name=ConnectorName
        Examples:
        - external://host:0/external_log_connector.lmc_external_log_connector/?connector_name=ExternalLogConnector
        """
        logger.info(f"Creating External connector for URL: {context.url}")
        hosts = context.url.split(",")
        if len(hosts) > 1:
            raise ValueError(
                f"Only one host is supported for external connector, but got {hosts}"
            )

        parse_url = parse_remote_url(context.url)

        # Get the module path and connector name
        module_path = parse_url.path.strip("/")
        connector_name = parse_url.query_params.get("connector_name", [""])[0]
        if not connector_name:
            raise ValueError(
                "External connector requires 'connector_name' in query parameters"
            )

        # Lazily import the module and get the connector class
        # Standard
        import importlib

        try:
            module = importlib.import_module(module_path)
            connector_class = getattr(module, connector_name)

            # Verify that it's a subclass of RemoteConnector
            if not issubclass(connector_class, RemoteConnector):
                raise TypeError(
                    f"{connector_name} must be a subclass of RemoteConnector"
                )

            # Create the connector instance
            connector = connector_class(
                loop=context.loop,
                local_cpu_backend=context.local_cpu_backend,
                config=context.config,
            )
            logger.info(f"Loaded external connector: {module_path}.{connector_name}")
            return connector
        except ImportError as e:
            raise ImportError(f"Could not import module '{module_path}'") from e
        except AttributeError as e:
            raise AttributeError(
                f"Module '{module_path}' has no class '{connector_name}'"
            ) from e
