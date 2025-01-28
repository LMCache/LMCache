from lmcache.logging import init_logger
from lmcache.experiemental.server.server_storage_backend.abstract_backend import \
    LMSBackendInterface
from lmcache.experimental.server.server_storage_backend.local_backend import (
    LMSLocalBackend, LMSLocalDiskBackend)

logger = init_logger(__name__)


def CreateStorageBackend(device: str) -> LMSBackendInterface:
    match device:
        case "cpu":
            # cpu only
            logger.info("Initializing cpu-only cache server")
            return LMSLocalBackend()

        # TODO(Jiayi): please implement heirarchical remote storage
        case _:
            logger.info("Initializing disk-only cache server")
            return LMSLocalDiskBackend(path=device)
