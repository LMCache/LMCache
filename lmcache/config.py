# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import os
import re

# Third Party
import torch
import yaml

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class LMCacheEngineMetadata:
    """name of the LLM model"""

    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the format of kv tensors """
    fmt: str
    """ the data type of kv tensors """
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    """ whether use MLA"""
    use_mla: bool = False
    """ the first rank of the distributed setting """
    # TODO(baoloongmao): first_rank should be configurable
    first_rank = 0

    def is_first_rank(self) -> bool:
        """Check if the current worker is the first rank"""
        return self.worker_id == self.first_rank


@dataclass
class LMCacheMemPoolMetadata:
    """Subset of `LMCacheEngineMetadata` to initialize MemPool"""

    kv_shape: Tuple[int, int, int, int, int]
    kv_dtype: torch.dtype
    max_local_cache_size: int


blend_default_separator = "[BLEND_SEP]"


@dataclass
class LMCacheEngineConfig:
    chunk_size: int
    local_device: Optional[str]
    max_local_cache_size: int
    remote_url: Optional[str]
    remote_serde: Optional[str]  # Can be "torch" or "cachegen"

    pipelined_backend: bool

    save_decode_cache: bool  # whether to store decode kv cache

    enable_blending: bool  # whether to enable blending
    blend_recompute_ratio: float  # the ratio of blending recompute
    blend_min_tokens: int  # the minimum number of tokens for blending
    blend_separator: str  # the separator for blending
    blend_add_special_in_precomp: bool
    # whether to add special tokens in pre-computations

    @staticmethod
    def from_defaults(
        chunk_size: int = 256,
        local_device: str = "cuda",
        max_local_cache_size: int = 5,
        remote_url: Optional[str] = "redis://localhost:6379",
        remote_serde: Optional[str] = "torch",
        pipelined_backend: bool = False,
        save_decode_cache: bool = False,
        enable_blending: bool = False,
        blend_recompute_ratio: float = 0.15,
        blend_min_tokens: int = 256,
        blend_separator: str = blend_default_separator,
        blend_add_special_in_precomp: bool = False,
    ) -> "LMCacheEngineConfig":
        return LMCacheEngineConfig(
            chunk_size,
            local_device,
            max_local_cache_size,
            remote_url,
            remote_serde,
            pipelined_backend,
            save_decode_cache,
            enable_blending,
            blend_recompute_ratio,
            blend_min_tokens,
            blend_separator,
            blend_add_special_in_precomp,
        )

    @staticmethod
    def from_legacy(
        chunk_size: int = 256,
        backend: str = "cuda",
        max_local_cache_size: int = 5,
        persist_path: Optional[str] = None,
        remote_serde: Optional[str] = "torch",
        pipelined_backend: bool = False,
        save_decode_cache: bool = False,
    ) -> "LMCacheEngineConfig":
        local_device: Optional[str] = None
        remote_url: Optional[str] = None

        match backend:
            case "cpu" | "cuda":
                local_device = backend
                remote_url = None
            case path if re.match(r"file://(.*)/", path):  # local disk directory
                local_device = path[7:]
                remote_url = None
            case url if re.match(r"(.*)://(.*):(\d+)", url):
                local_device = None
                remote_url = url
        return LMCacheEngineConfig(
            chunk_size,
            local_device,
            max_local_cache_size,
            remote_url,
            remote_serde,
            pipelined_backend,
            save_decode_cache,
            enable_blending=False,
            blend_recompute_ratio=0.15,
            blend_min_tokens=256,
            blend_separator=blend_default_separator,
            blend_add_special_in_precomp=False,
        ).log_config()

    @staticmethod
    def from_file(file_path: str) -> "LMCacheEngineConfig":
        """
        Load the config from a yaml file
        """
        with open(file_path, "r") as fin:
            config = yaml.safe_load(fin)

        chunk_size = config.get("chunk_size", 256)
        local_device = config.get("local_device", None)
        max_local_cache_size = config.get("max_local_cache_size", 20)
        remote_url = config.get("remote_url", None)
        remote_serde = config.get("remote_serde", None)
        pipelined_backend = config.get("pipelined_backend", False)
        save_decode_cache = config.get("save_decode_cache", False)
        enable_blending = config.get("enable_blending", False)
        blend_recompute_ratio = config.get("blend_recompute_ratio", 0.15)
        blend_min_tokens = config.get("blend_min_tokens", 256)
        blend_separator = config.get("blend_separator", blend_default_separator)
        blend_add_special_in_precomp = config.get("blend_add_special_in_precomp", False)

        match local_device:
            case "cpu" | "cuda" | None:
                pass
            case path if re.match(r"file://(.*)/", path):  # local disk directory
                local_device = path[7:]
            case _:
                raise ValueError(f"Invalid local storage device: {local_device}")

        match remote_url:
            case None:
                pass
            case url if re.match(r"(.*)://(.*):(\d+)", url):
                pass
            case _:
                raise ValueError(f"Invalid remote storage url: {remote_url}")

        return LMCacheEngineConfig(
            chunk_size,
            local_device,
            max_local_cache_size,
            remote_url,
            remote_serde,
            pipelined_backend,
            save_decode_cache,
            enable_blending,
            blend_recompute_ratio,
            blend_min_tokens,
            blend_separator,
            blend_add_special_in_precomp,
        ).log_config()

    @staticmethod
    def from_env() -> "LMCacheEngineConfig":
        """Load the config from the environment variables

        It will first create a config by `from_defaults` and overwrite
        the configuration values from the environment variables.

        The environment variables should starts with LMCACHE and be in
        uppercase. For example, `LMCACHE_CHUNK_SIZE`.

        :note: the default configuration only uses cpu
        """

        def get_env_name(attr_name: str) -> str:
            return f"LMCACHE_{attr_name.upper()}"

        def parse_env(name: str, default: Optional[Any]):
            if default is not None:
                return os.getenv(name, str(default))
            else:
                return os.getenv(name)

        config = LMCacheEngineConfig.from_defaults(
            local_device="cpu", remote_url=None, remote_serde=None
        )

        config.chunk_size = int(
            parse_env(get_env_name("chunk_size"), config.chunk_size)
        )
        config.local_device = parse_env(
            get_env_name("local_device"), config.local_device
        )
        config.max_local_cache_size = int(
            parse_env(
                get_env_name("max_local_cache_size"),
                config.max_local_cache_size,
            )
        )
        config.remote_url = parse_env(get_env_name("remote_url"), config.remote_url)
        config.remote_serde = parse_env(
            get_env_name("remote_serde"), config.remote_serde
        )
        config.pipelined_backend = parse_env(
            get_env_name("pipelined_backend"), config.pipelined_backend
        )
        config.save_decode_cache = parse_env(
            get_env_name("save_decode_cache"), config.save_decode_cache
        )
        config.enable_blending = parse_env(
            get_env_name("enable_blending"), config.enable_blending
        )
        config.blend_recompute_ratio = float(
            parse_env(
                get_env_name("blend_recompute_ratio"),
                config.blend_recompute_ratio,
            )
        )
        config.blend_min_tokens = int(
            parse_env(get_env_name("blend_min_tokens"), config.blend_min_tokens)
        )
        config.blend_separator = parse_env(
            get_env_name("blend_separator"), config.blend_separator
        )
        config.blend_add_special_in_precomp = bool(
            parse_env(
                get_env_name("blend_add_special_in_precomp"),
                config.blend_add_special_in_precomp,
            )
        )

        return config.log_config()

    def log_config(self) -> "LMCacheEngineConfig":
        """Log all configuration settings"""
        config_dict = {
            "chunk_size": self.chunk_size,
            "local_device": self.local_device,
            "max_local_cache_size": f"{self.max_local_cache_size} GB",
            "remote_url": self.remote_url,
            "remote_serde": self.remote_serde,
            "pipelined_backend": self.pipelined_backend,
            "save_decode_cache": self.save_decode_cache,
            "enable_blending": self.enable_blending,
            "blend_recompute_ratio": self.blend_recompute_ratio,
            "blend_min_tokens": self.blend_min_tokens,
            "blend_separator": self.blend_separator,
            "blend_add_special_in_precomp": self.blend_add_special_in_precomp,
        }
        logger.info(f"LMCache Configuration: {config_dict}")

        return self


### SOME GLOBAL CONFIGS
# TODO: it needs to be manually updated in the code here, but cannot be really
# configured
class GlobalConfig:
    enable_debug: bool = True

    @classmethod
    def set_debug(cls, enable: bool):
        cls.enable_debug = enable

    @classmethod
    def is_debug(cls) -> bool:
        return cls.enable_debug
