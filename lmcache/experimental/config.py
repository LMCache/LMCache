import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import yaml


@dataclass
class LMCacheEngineConfig:
    chunk_size: int
    local_cpu: bool
    max_local_cpu_size: float  # in GB
    # need to be assigned a non-zero
    # value even if local_cpu is disabled
    local_disk: Optional[str]
    max_local_disk_size: float  # in GB

    remote_url: Optional[str]
    remote_serde: Optional[str]  # Can be "torch" or "cachegen"

    save_decode_cache: bool  # whether to store decode kv cache

    enable_blending: bool  # whether to enable blending
    blend_recompute_ratio: float  # the ratio of blending recompute
    blend_min_tokens: int  # the minimum number of tokens for blending

    @staticmethod
    def from_defaults(
        chunk_size: int = 256,
        local_cpu: bool = True,
        max_local_cpu_size: int = 5,
        local_disk: Optional[str] = None,
        max_local_disk_size: int = 0,
        remote_url: Optional[str] = "redis://localhost:6379",
        remote_serde: Optional[str] = "torch",
        save_decode_cache: bool = False,
        enable_blending: bool = False,
        blend_recompute_ratio: float = 0.15,
        blend_min_tokens: int = 256,
    ) -> "LMCacheEngineConfig":
        return LMCacheEngineConfig(chunk_size, local_cpu, max_local_cpu_size,
                                   local_disk, max_local_disk_size, remote_url,
                                   remote_serde, save_decode_cache,
                                   enable_blending, blend_recompute_ratio,
                                   blend_min_tokens)

    @staticmethod
    def from_legacy(
        chunk_size: int = 256,
        backend: str = "cpu",
        remote_url: str = "redis://localhost:6379",
        remote_serde: str = "torch",
        save_decode_cache: bool = False,
        enable_blending: bool = False,
        blend_recompute_ratio: float = 0.15,
        blend_min_tokens: int = 256,
    ) -> "LMCacheEngineConfig":
        if backend == "cpu":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = None
            max_local_disk_size = 0
        elif backend == "local_disk":
            local_cpu = False
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
        elif backend == "local_cpu_disk":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
        else:
            raise ValueError(f"Invalid backend: {backend}")
        return LMCacheEngineConfig(chunk_size, local_cpu, max_local_cpu_size,
                                   local_disk, max_local_disk_size, remote_url,
                                   remote_serde, save_decode_cache,
                                   enable_blending, blend_recompute_ratio,
                                   blend_min_tokens)

    @staticmethod
    def from_file(file_path: str) -> "LMCacheEngineConfig":
        """
        Load the config from a yaml file
        """
        with open(file_path, "r") as fin:
            config = yaml.safe_load(fin)

        chunk_size = config.get("chunk_size", 256)

        local_cpu = config.get("local_cpu", True)
        max_local_cpu_size = config.get("max_local_cpu_size", 5)

        local_disk = config.get("local_disk", None)
        max_local_disk_size = config.get("max_local_disk_size", 5)

        remote_url = config.get("remote_url", None)
        remote_serde = config.get("remote_serde", "torch")

        save_decode_cache = config.get("save_decode_cache", False)
        enable_blending = config.get("enable_blending", False)
        blend_recompute_ratio = config.get("blend_recompute_ratio", 0.15)
        blend_min_tokens = config.get("blend_min_tokens", 256)

        match local_disk:
            case None:
                local_disk_path = None
            case path if re.match(r"file://(.*)/",
                                  path):  # local disk directory
                local_disk_path = path[7:]

        match remote_url:
            case None:
                pass
            case url if re.match(r"(.*)://(.*):(\d+)", url):
                pass
            case _:
                raise ValueError(f"Invalid remote storage url: {remote_url}")

        return LMCacheEngineConfig(
            chunk_size,
            local_cpu,
            max_local_cpu_size,
            local_disk_path,
            max_local_disk_size,
            remote_url,
            remote_serde,
            save_decode_cache,
            enable_blending,
            blend_recompute_ratio,
            blend_min_tokens,
        )

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

        config = LMCacheEngineConfig.from_defaults(remote_url=None,
                                                   remote_serde=None)

        config.chunk_size = int(
            parse_env(get_env_name("chunk_size"), config.chunk_size))
        config.local_cpu = parse_env(get_env_name("local_cpu"),
                                     config.local_cpu)
        config.max_local_cpu_size = int(
            parse_env(get_env_name("max_local_cpu_size"),
                      config.max_local_cpu_size))
        config.local_disk = parse_env(get_env_name("local_disk"),
                                      config.local_disk)
        config.max_local_disk_size = int(
            parse_env(get_env_name("max_local_disk_size"),
                      config.max_local_disk_size))
        config.remote_url = parse_env(get_env_name("remote_url"),
                                      config.remote_url)
        config.remote_serde = parse_env(get_env_name("remote_serde"),
                                        config.remote_serde)
        config.save_decode_cache = parse_env(get_env_name("save_decode_cache"),
                                             config.save_decode_cache)
        config.enable_blending = parse_env(get_env_name("enable_blending"),
                                           config.enable_blending)
        config.blend_recompute_ratio = float(
            parse_env(get_env_name("blend_recompute_ratio"),
                      config.blend_recompute_ratio))
        config.blend_min_tokens = int(
            parse_env(get_env_name("blend_min_tokens"),
                      config.blend_min_tokens))

        return config
