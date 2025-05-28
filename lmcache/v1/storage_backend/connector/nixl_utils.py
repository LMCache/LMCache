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
from dataclasses import dataclass
from typing import Union
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
    enable_gc: bool

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
        )
