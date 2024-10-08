from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.hybrid_backend import \
    LMCHybridBackend  # , LMCPipelinedHybridBackend
from lmcache.storage_backend.local_backend import (LMCLocalBackend,
                                                   LMCLocalDiskBackend)
from lmcache.storage_backend.remote_backend import LMCRemoteBackend

logger = init_logger(__name__)


def CreateStorageBackend(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata) -> LMCBackendInterface:
    match config:
        case LMCacheEngineConfig(_, local_device=None,
                                 remote_url=str(p)) if p is not None:
            # remote only
            logger.info("Initializing remote-only backend")
            return LMCRemoteBackend(config, metadata)

        case LMCacheEngineConfig(_, local_device=str(p),
                                 remote_url=None) if p is not None:
            # local only
            match config.local_device:
                case "cpu" | "cuda":
                    logger.info(
                        f"Initializing local-only ({config.local_device})"
                        f" backend")
                    return LMCLocalBackend(config)
                case _:
                    logger.info(f"Initializing local-only (disk) backend at"
                                f" {config.local_device}")
                    return LMCLocalDiskBackend(config)

        case LMCacheEngineConfig(
                _, local_device=str(p),
                remote_url=str(q)) if p is not None and q is not None:
            logger.info("Initializing hybrid backend")
            return LMCHybridBackend(config, metadata)

        case _:
            raise ValueError(f"Invalid configuration: {config}")


# __all__ = [
#    "LMCBackendInterface",
#    "LMCLocalBackend",
#    "LMCRemoteBackend",
# ]
