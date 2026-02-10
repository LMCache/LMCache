# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)

logger = init_logger(__name__)


class ShmFileConnectorAdapter(ConnectorAdapter):
    """Adapter for shared-memory file connectors."""

    def __init__(self) -> None:
        super().__init__("shmfile://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .shmfile_connector import ShmFileConnector

        logger.info(
            "Creating ShmFile connector for URL: %s",
            context.url,
        )
        parse_url = parse_remote_url(context.url)
        shm_name = parse_url.query_params.get("shm_name", [None])[0]
        worker_bin = parse_url.query_params.get("worker_bin", [None])[0]

        # Inject shm_name into config.extra_config so that
        # MixedMemoryAllocator can pick it up during allocation.
        if shm_name and context.config is not None:
            if context.config.extra_config is None:
                context.config.extra_config = {}
            context.config.extra_config["shm_name"] = shm_name

        return ShmFileConnector(
            storage_dir=parse_url.path,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            config=context.config,
            shm_name=shm_name,
            worker_binary=worker_bin,
        )
