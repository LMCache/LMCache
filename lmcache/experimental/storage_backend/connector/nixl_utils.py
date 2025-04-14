import enum
from dataclasses import dataclass
from typing import Union

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig


class NixlRole(enum.Enum):
    """
    Enum to represent the role of the Nixl connection.
    """
    SENDER = "sender"
    RECEIVER = "receiver"


@dataclass
class NixlConfig:
    role: Union[NixlRole, str]
    peer_host_name: str
    peer_port: int
    buffer_size: int
    buffer_device: str
    enable_gc: bool

    @staticmethod
    def from_cache_engine_config(
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata) -> "NixlConfig":
        """Convert the LMCacheEngineConfig to NixlConfig
        """
        worker_id = metadata.worker_id
        assert config.enable_nixl is True, \
            "NIXL is not enabled in the LMCacheEngineConfig"

        if isinstance(config.nixl_role, str):
            nixl_role = NixlRole(config.nixl_role)
        else:
            assert isinstance(config.nixl_role, NixlRole)
            nixl_role = config.nixl_role

        assert nixl_role in [NixlRole.SENDER, NixlRole.RECEIVER], \
                f"Invalid role: {config.nixl_role}, must be either "\
                f"{NixlRole.SENDER} or {NixlRole.RECEIVER}"

        assert config.nixl_peer_host is not None
        assert config.nixl_peer_port is not None
        assert config.nixl_buffer_size is not None
        assert config.nixl_buffer_device is not None
        assert config.nixl_enable_gc is not None

        return NixlConfig(role=nixl_role,
                          peer_host_name=config.nixl_peer_host,
                          peer_port=config.nixl_peer_port + worker_id,
                          buffer_size=config.nixl_buffer_size,
                          buffer_device=config.nixl_buffer_device,
                          enable_gc=config.nixl_enable_gc)
