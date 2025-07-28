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
from typing import Any, Dict, List, Optional
import json
import os
import re

# Third Party
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
import yaml

# First Party
from lmcache.logging import init_logger
import lmcache.config as orig_config

logger = init_logger(__name__)


class LMCacheEngineConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    chunk_size: int = 256
    local_cpu: bool = True
    max_local_cpu_size: float = 5.0  # in GB
    local_disk: Optional[str] = None
    max_local_disk_size: float = 0.0  # in GB

    remote_url: Optional[str] = None
    remote_serde: Optional[str] = "naive"

    use_layerwise: bool = False
    save_decode_cache: bool = False

    enable_blending: bool = False
    blend_recompute_ratio: float = 0.15
    blend_min_tokens: int = 256
    blend_special_str: str = " # # "

    enable_p2p: bool = False
    lookup_url: Optional[str] = None
    distributed_url: Optional[str] = None

    error_handling: bool = False

    enable_controller: bool = False
    lmcache_instance_id: str = "lmcache_default_instance"
    controller_url: Optional[str] = None
    lmcache_worker_port: Optional[int] = None
    # Algorithm used to hash tokens pre caching
    pre_caching_hash_algorithm: str = "builtin"

    enable_nixl: bool = False
    nixl_role: Optional[str] = None
    nixl_receiver_host: Optional[str] = None
    nixl_receiver_port: Optional[int] = None
    nixl_buffer_size: Optional[int] = None
    nixl_buffer_device: Optional[str] = None
    nixl_enable_gc: bool = False

    enable_xpyd: bool = False
    nixl_peer_host: Optional[str] = None
    nixl_peer_init_port: Optional[List[int]] = None
    nixl_peer_alloc_port: Optional[List[int]] = None
    nixl_proxy_host: Optional[str] = None
    nixl_proxy_port: Optional[int] = None

    audit_actual_remote_url: Optional[str] = None
    weka_path: Optional[str] = None
    gds_path: Optional[str] = None
    cufile_buffer_size: Optional[int] = None
    extra_config: Optional[Dict[str, Any]] = None
    save_unfull_chunk: bool = True
    blocking_timeout_secs: int = 10
    external_lookup_client: Optional[str] = None

    @field_validator("local_disk", mode="before")
    @classmethod
    def _parse_local_disk(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str) and (match := re.match(r"file://(.*)/", v)):
            return match.group(1)
        return v

    @field_validator("nixl_peer_init_port", "nixl_peer_alloc_port", mode="before")
    @classmethod
    def _parse_int_list(cls, v: Any) -> Optional[List[int]]:
        if v is None:
            return None
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            return [int(p) for p in parts]
        return v

    @field_validator("extra_config", mode="before")
    @classmethod
    def _parse_extra_config(cls, v: Any) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError("extra_config must be a valid JSON string") from e
        if isinstance(v, dict):
            return v
        raise TypeError("extra_config must be a JSON string or a dictionary")

    @model_validator(mode="after")
    def validate_model(self) -> "LMCacheEngineConfig":
        if self.enable_p2p:
            if self.lookup_url is None:
                raise ValueError("lookup_url must be set when enable_p2p is True")
            if self.distributed_url is None:
                raise ValueError("distributed_url must be set when enable_p2p is True")

        if self.enable_nixl:
            if self.nixl_role is None:
                raise ValueError("nixl_role must be set when enable_nixl is True")
            if self.nixl_buffer_size is None:
                raise ValueError(
                    "nixl_buffer_size must be set when enable_nixl is True"
                )
            if self.nixl_buffer_device is None:
                raise ValueError(
                    "nixl_buffer_device must be set when enable_nixl is True"
                )
            if self.local_cpu:
                raise ValueError("Nixl only supports local_cpu=False")
            if self.max_local_cpu_size != 0:
                raise ValueError("Nixl only supports max_local_cpu_size=0")
            if self.local_disk is not None:
                raise ValueError("Nixl only supports local_disk=None")
            if self.remote_url is not None:
                raise ValueError("Nixl only supports remote_url=None")
            if self.save_decode_cache:
                raise ValueError("Nixl only supports save_decode_cache=False")
            if self.enable_p2p:
                raise ValueError("Nixl only supports enable_p2p=False")

        if self.remote_url:
            if not re.match(r"(.*)://(.*):(\d+)", self.remote_url):
                raise ValueError(f"Invalid remote storage url: {self.remote_url}")

        return self

    @classmethod
    def from_file(cls, file_path: str) -> "LMCacheEngineConfig":
        """
        Load the config from a yaml file
        """
        with open(file_path, "r") as fin:
            config_dict = yaml.safe_load(fin)

        # Handle legacy nixl config keys for backward compatibility
        if "nixl_receiver_host" not in config_dict and "nixl_peer_host" in config_dict:
            logger.warning(
                "nixl_peer_host is deprecated, please use "
                "nixl_receiver_host in the config file instead"
            )
            config_dict["nixl_receiver_host"] = config_dict["nixl_peer_host"]

        if "nixl_receiver_port" not in config_dict and "nixl_peer_port" in config_dict:
            logger.warning(
                "nixl_peer_port is deprecated, please use "
                "nixl_receiver_port in the config file instead"
            )
            config_dict["nixl_receiver_port"] = config_dict["nixl_peer_port"]

        try:
            instance = cls(**config_dict)
            instance.log_config()
            return instance
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    @classmethod
    def from_env(cls) -> "LMCacheEngineConfig":
        """Load the config from the environment variables"""
        config_dict = {}
        for field_name in cls.model_fields:
            env_var_name = f"LMCACHE_{field_name.upper()}"
            if env_var_name in os.environ:
                config_dict[field_name] = os.environ[env_var_name]

        # Handle legacy nixl config keys for backward compatibility
        if (
            "nixl_receiver_host" not in config_dict
            and os.getenv("LMCACHE_NIXL_PEER_HOST") is not None
        ):
            logger.warning(
                "LMCACHE_NIXL_PEER_HOST is deprecated, please use "
                "LMCACHE_NIXL_RECEIVER_HOST environment variable instead"
            )
            config_dict["nixl_receiver_host"] = os.getenv("LMCACHE_NIXL_PEER_HOST")

        if (
            "nixl_receiver_port" not in config_dict
            and os.getenv("LMCACHE_NIXL_PEER_PORT") is not None
        ):
            logger.warning(
                "LMCACHE_NIXL_PEER_PORT is deprecated, please use "
                "LMCACHE_NIXL_RECEIVER_PORT environment variable instead"
            )
            config_dict["nixl_receiver_port"] = os.getenv("LMCACHE_NIXL_PEER_PORT")

        try:
            instance = cls(**config_dict)
            instance.log_config()
            return instance
        except ValidationError as e:
            logger.error(
                f"Configuration validation error from environment variables: {e}"
            )
            raise

    @classmethod
    def from_legacy(
        cls,
        chunk_size: int = 256,
        backend: str = "cpu",
        remote_url: Optional[str] = "lm://localhost:65432",
        remote_serde: str = "naive",
        use_layerwise: bool = False,
        save_decode_cache: bool = False,
        enable_blending: bool = False,
        blend_recompute_ratio: float = 0.15,
        blend_min_tokens: int = 256,
        blend_special_str: str = " # # ",
        max_local_disk_size: float = 0.0,
        enable_p2p: bool = False,
        lookup_url: Optional[str] = None,
        distributed_url: Optional[str] = None,
        error_handling: bool = False,
        save_unfull_chunk: bool = True,
    ) -> "LMCacheEngineConfig":
        # TODO (ApostaC): Add nixl config
        config_dict = {
            "chunk_size": chunk_size,
            "remote_url": remote_url,
            "remote_serde": remote_serde,
            "use_layerwise": use_layerwise,
            "save_decode_cache": save_decode_cache,
            "enable_blending": enable_blending,
            "blend_recompute_ratio": blend_recompute_ratio,
            "blend_min_tokens": blend_min_tokens,
            "blend_special_str": blend_special_str,
            "max_local_disk_size": max_local_disk_size,
            "enable_p2p": enable_p2p,
            "lookup_url": lookup_url,
            "distributed_url": distributed_url,
            "error_handling": error_handling,
            "save_unfull_chunk": save_unfull_chunk,
        }

        if backend == "cpu":
            config_dict.update(
                {
                    "local_cpu": True,
                    "max_local_cpu_size": 5,
                    "local_disk": None,
                    "max_local_disk_size": 0,
                    "remote_url": None,
                }
            )
        elif backend == "local_disk":
            config_dict.update(
                {
                    "local_cpu": False,
                    "max_local_cpu_size": 5,
                    "local_disk": "local/disk_test/local_disk/",
                    "max_local_disk_size": 5,
                    "remote_url": None,
                }
            )
        elif backend == "local_cpu_disk":
            config_dict.update(
                {
                    "local_cpu": True,
                    "max_local_cpu_size": 5,
                    "local_disk": "local/disk_test/local_disk/",
                    "max_local_disk_size": 5,
                    "remote_url": None,
                }
            )
        elif backend == "remote":
            config_dict.update(
                {
                    "local_cpu": False,
                    "max_local_cpu_size": 5,
                    "local_disk": None,
                }
            )
        elif backend == "local_cpu_remote":
            config_dict.update(
                {
                    "local_cpu": True,
                    "max_local_cpu_size": 5,
                    "local_disk": None,
                }
            )
        elif backend == "local_disk_remote":
            config_dict.update(
                {
                    "local_cpu": False,
                    "max_local_cpu_size": 5,
                    "local_disk": "local/disk_test/local_disk/",
                    "max_local_disk_size": 5,
                }
            )
        elif backend == "local_cpu_disk_remote":
            config_dict.update(
                {
                    "local_cpu": True,
                    "max_local_cpu_size": 5,
                    "local_disk": "local/disk_test/local_disk/",
                    "max_local_disk_size": 5,
                }
            )
        else:
            raise ValueError(f"Invalid backend: {backend}")

        instance = cls(**config_dict)
        instance.log_config()
        return instance

    def to_original_config(self) -> orig_config.LMCacheEngineConfig:
        # NOTE: This function is purely for UsageContext compatibility
        return orig_config.LMCacheEngineConfig(
            chunk_size=self.chunk_size,
            local_device="cpu" if self.local_cpu else "cuda",
            max_local_cache_size=int(self.max_local_cpu_size),
            remote_url=None,
            remote_serde=None,
            pipelined_backend=False,
            save_decode_cache=self.save_decode_cache,
            enable_blending=self.enable_blending,
            blend_recompute_ratio=self.blend_recompute_ratio,
            blend_min_tokens=self.blend_min_tokens,
            blend_separator="[BLEND_SEP]",
            blend_add_special_in_precomp=False,
        )

    def log_config(self) -> "LMCacheEngineConfig":
        """log the configuration in LMCache"""
        config_dict = self.model_dump(exclude_none=True)
        # Format sizes for readability
        if "max_local_cpu_size" in config_dict:
            config_dict["max_local_cpu_size"] = (
                f"{config_dict['max_local_cpu_size']} GB"
            )
        if "max_local_disk_size" in config_dict:
            config_dict["max_local_disk_size"] = (
                f"{config_dict['max_local_disk_size']} GB"
            )

        logger.info(f"LMCache Configuration: {config_dict}")
        return self
