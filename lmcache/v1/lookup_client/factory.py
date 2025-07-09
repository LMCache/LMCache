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

# Standard
from typing import TYPE_CHECKING, Optional

# First Party
from lmcache.integration.vllm.utils import lmcache_get_config
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    # First Party
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)


class LookupClientFactory:
    """Factory for creating lookup clients and servers based on configuration."""

    @staticmethod
    def create_lookup_client(
        role: "KVConnectorRole",
        is_tp: bool,
        vllm_config: "VllmConfig",
    ) -> LookupClientInterface:
        """
        Create a lookup client based on the configuration.

        Args:
            role: The KV connector role
            is_tp: Whether tensor parallelism is enabled
            vllm_config: The vLLM configuration

        Returns:
            A lookup client instance
        """
        config = lmcache_get_config()

        # Check if mooncake_lookup_client is configured
        if config.mooncake_lookup_client is not None:
            # First Party
            from lmcache.v1.lookup_client.mooncake_lookup_client import (
                MooncakeLookupClient,
            )

            return MooncakeLookupClient(
                role, is_tp, vllm_config, config.mooncake_lookup_client
            )
        else:
            # First Party
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupClient,
            )

            return LMCacheLookupClient(role, is_tp, vllm_config)

    @staticmethod
    def create_lookup_server(
        lmcache_engine: LMCacheEngine,
        role: "KVConnectorRole",
        is_tp: bool,
        vllm_config: "VllmConfig",
    ) -> Optional["LMCacheLookupServer"]:
        """
        Create a lookup server based on the configuration.

        Args:
            lmcache_engine: The LMCache engine instance
            role: The KV connector role
            is_tp: Whether tensor parallelism is enabled
            vllm_config: The vLLM configuration

        Returns:
            A lookup server instance, or None if no server should be created
        """
        config = lmcache_get_config()

        # Only create the KV lookup API server on worker rank 0
        # when there are multiple workers and when not using Mooncake lookup client
        if (
            vllm_config.parallel_config.rank == 0
            and config.mooncake_lookup_client is None
        ):
            # First Party
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupServer,
            )

            return LMCacheLookupServer(lmcache_engine, role, is_tp, vllm_config)

        return None
