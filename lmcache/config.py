import re
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import yaml


@dataclass
class LMCacheEngineMetadata:
    """ name of the LLM model """
    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the format of kv tensors """
    fmt: str
    """ the data type of kv tensors """
    kv_dtype: torch.dtype
    """ the data type of kv tensors """
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]


@dataclass
class LMCacheMemPoolMetadata:
    """ Subset of `LMCacheEngineMetadata` to initialize MemPool"""

    kv_shape: Tuple[int, int, int, int, int]
    kv_dtype: torch.dtype
    max_local_cache_size: int


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

    @staticmethod
    def from_defaults(
        chunk_size: int = 256,
        local_device: str = "cuda",
        max_local_cache_size: int = 5,
        remote_url: str = "redis://localhost:6379",
        remote_serde: str = "torch",
        pipelined_backend: bool = False,
        save_decode_cache: bool = False,
        enable_blending: bool = False,
        blend_recompute_ratio: float = 0.15,
        blend_min_tokens: int = 256,
    ) -> "LMCacheEngineConfig":
        return LMCacheEngineConfig(chunk_size, local_device,
                                   max_local_cache_size, remote_url,
                                   remote_serde, pipelined_backend,
                                   save_decode_cache, enable_blending,
                                   blend_recompute_ratio, blend_min_tokens)

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
            case path if re.match(r"file://(.*)/",
                                  path):  # local disk directory
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
        )

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

        match local_device:
            case "cpu" | "cuda" | None:
                pass
            case path if re.match(r"file://(.*)/",
                                  path):  # local disk directory
                local_device = path[7:]
            case _:
                raise ValueError(
                    f"Invalid local storage device: {local_device}")

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
        )


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
