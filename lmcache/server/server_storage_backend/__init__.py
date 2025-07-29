# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.server.server_storage_backend.abstract_backend import (
    LMSBackendInterface,
)
from lmcache.server.server_storage_backend.local_backend import (
    LMSLocalBackend,
    LMSLocalDiskBackend,
)

logger = init_logger(__name__)


def CreateStorageBackend(device: str) -> LMSBackendInterface:
    match device:
        case "cpu":
            # cpu only
            logger.info("Initializing cpu-only cache server")
            return LMSLocalBackend()

        case _:
            # cpu only
            logger.info("Initializing disk-only cache server")
            return LMSLocalDiskBackend(path=device)
