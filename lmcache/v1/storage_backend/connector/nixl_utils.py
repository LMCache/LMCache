# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional, Union
import enum

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig


def get_correct_nixl_device(nixl_device: str, worker_id: int) -> str:
    """
    Get the correct Nixl device based on the given device string.

    Args:
        nixl_device (str): The device string, could be cpu or cuda

    Returns:
        str: The correct device string for Nixl -- with correct
          device id.
    """
    if nixl_device == "cpu":
        return "cpu"
    elif nixl_device.startswith("cuda"):
        return f"cuda:{worker_id}"
    else:
        raise ValueError(f"Invalid Nixl device: {nixl_device}")


class NixlRole(enum.Enum):
    """
    Enum to represent the role of the Nixl connection.
    """

    SENDER = "sender"
    RECEIVER = "receiver"


@dataclass
class NixlConfig:
    role: Union[NixlRole, str]
    receiver_host: str
    receiver_port: int
    buffer_size: int
    buffer_device: str
    enable_gc: bool = True
    backends: Optional[list[str]] = None

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> "NixlConfig":
        """Convert the LMCacheEngineConfig to NixlConfig"""
        worker_id = metadata.worker_id
        assert config.enable_nixl is True, (
            "NIXL is not enabled in the LMCacheEngineConfig"
        )

        if isinstance(config.nixl_role, str):
            nixl_role = NixlRole(config.nixl_role)
        else:
            assert isinstance(config.nixl_role, NixlRole)
            nixl_role = config.nixl_role

        assert nixl_role in [NixlRole.SENDER, NixlRole.RECEIVER], (
            f"Invalid role: {config.nixl_role}, must be either "
            f"{NixlRole.SENDER} or {NixlRole.RECEIVER}"
        )

        assert config.nixl_receiver_host is not None
        assert config.nixl_receiver_port is not None
        assert config.nixl_buffer_size is not None
        assert config.nixl_buffer_device is not None
        assert config.nixl_enable_gc is not None

        corrected_device = get_correct_nixl_device(
            config.nixl_buffer_device, metadata.worker_id
        )

        return NixlConfig(
            role=nixl_role,
            receiver_host=config.nixl_receiver_host,
            receiver_port=config.nixl_receiver_port + worker_id,
            buffer_size=config.nixl_buffer_size,
            buffer_device=corrected_device,
            enable_gc=config.nixl_enable_gc,
            backends=config.nixl_backends,
        )


@dataclass
class NixlConfigXpYd:
    role: Union[NixlRole, str]

    peer_host: str
    peer_init_port: list[int]
    peer_alloc_port: list[int]

    proxy_host: str
    proxy_port: int

    buffer_size: int
    buffer_device: str

    backends: Optional[list[str]]

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> "NixlConfigXpYd":
        """Convert the LMCacheEngineConfig to NixlConfigXpYd"""
        # TODO (Jiayi): add (heterogeneous) TP support
        # worker_id = metadata.worker_id
        # assert config.enable_nixl is True, (
        #     "NIXL is not enabled in the LMCacheEngineConfig"
        # )

        if isinstance(config.nixl_role, str):
            nixl_role = NixlRole(config.nixl_role)
        else:
            assert isinstance(config.nixl_role, NixlRole)
            nixl_role = config.nixl_role

        assert nixl_role in [NixlRole.SENDER, NixlRole.RECEIVER], (
            f"Invalid role: {config.nixl_role}, must be either "
            f"{NixlRole.SENDER} or {NixlRole.RECEIVER}"
        )

        assert config.nixl_buffer_size is not None
        assert config.nixl_buffer_device is not None

        if nixl_role == NixlRole.RECEIVER:
            assert config.nixl_peer_host is not None
            assert config.nixl_peer_init_port is not None
            assert config.nixl_peer_alloc_port is not None
        elif nixl_role == NixlRole.SENDER:
            assert config.nixl_proxy_host is not None
            assert config.nixl_proxy_port is not None

        corrected_device = get_correct_nixl_device(
            config.nixl_buffer_device, metadata.worker_id
        )

        return NixlConfigXpYd(
            role=nixl_role,
            peer_host=config.nixl_peer_host,
            peer_init_port=config.nixl_peer_init_port,
            peer_alloc_port=config.nixl_peer_alloc_port,
            proxy_host=config.nixl_proxy_host,
            proxy_port=config.nixl_proxy_port,
            buffer_size=config.nixl_buffer_size,
            backends=config.nixl_backends,
            buffer_device=corrected_device,
        )
