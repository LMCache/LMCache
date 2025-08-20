import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import yaml

import lmcache.config as orig_config


@dataclass
class LMCacheEngineConfig:
    chunk_size: int
    local_cpu: bool
    max_local_cpu_size: float  # in GB
    # need to be assigned a non-zero
    # value even if local_cpu is disabled
    local_disk: Optional[str]
    max_local_disk_size: float  # in GB
    remote_disk: Optional[str]
    max_remote_disk_size: float  # in GB

    remote_url: Optional[str]
    remote_serde: Optional[str]  # Can be "naive" or "cachegen"

    save_decode_cache: bool  # whether to store decode kv cache

    enable_blending: bool  # whether to enable blending
    blend_recompute_ratio: float  # the ratio of blending recompute
    blend_min_tokens: int  # the minimum number of tokens for blending

    alpha: float
    policy: str
    rate: float
    compression: str
    dataset_csv: str
    method_output_csv: str

    @staticmethod
    def from_defaults(
        chunk_size: int = 256,
        local_cpu: bool = True,
        max_local_cpu_size: float = 5.0,
        local_disk: Optional[str] = None,
        max_local_disk_size: int = 0,
        remote_disk: Optional[str] = None,
        max_remote_disk_size: int = 0,
        remote_url: Optional[str] = "lm://localhost:65432",
        remote_serde: Optional[str] = "naive",
        save_decode_cache: bool = False,
        enable_blending: bool = False,
        blend_recompute_ratio: float = 0.15,
        blend_min_tokens: int = 256,
        alpha: float = 1.0,
        policy: str = "ours",
        rate: float = 1.0,
        compression: str = "kivi",
        dataset_csv: str = "blablabla",
        method_output_csv: str = "blablabla",
    ) -> "LMCacheEngineConfig":
        return LMCacheEngineConfig(chunk_size, local_cpu, max_local_cpu_size,
                                   local_disk, max_local_disk_size, remote_disk, max_remote_disk_size, remote_url,
                                   remote_serde, save_decode_cache,
                                   enable_blending, blend_recompute_ratio,
                                   blend_min_tokens, alpha, policy, rate, compression, dataset_csv, method_output_csv)

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

        remote_disk = config.get("remote_disk", None)
        max_remote_disk_size = config.get("max_remote_disk_size", 5)

        remote_url = config.get("remote_url", None)
        remote_serde = config.get("remote_serde", "naive")

        save_decode_cache = config.get("save_decode_cache", False)
        enable_blending = config.get("enable_blending", False)
        blend_recompute_ratio = config.get("blend_recompute_ratio", 0.15)
        blend_min_tokens = config.get("blend_min_tokens", 256)

        alpha = config.get("alpha", 1.0)
        policy = config.get("policy", "ours")
        rate = config.get("rate", 1.0)
        compression = config.get("compression", "kivi")
        dataset_csv = config.get("dataset_csv", "blablabla")
        method_output_csv = config.get("method_output_csv", "blablabla")

        match local_disk:
            case None:
                local_disk_path = None
            case path if re.match(r"file://(.*)/",
                                  path):  # local disk directory
                local_disk_path = path[7:]

        match remote_disk:
            case None:
                remote_disk_path = None
            case path if re.match(r"file://(.*)/",
                                  path):  # remote disk directory
                remote_disk_path = path[7:]

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
            remote_disk_path,
            max_remote_disk_size,
            remote_url,
            remote_serde,
            save_decode_cache,
            enable_blending,
            blend_recompute_ratio,
            blend_min_tokens,
            alpha,
            policy,
            rate,
            compression,
            dataset_csv,
            method_output_csv,
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

        def parse_env(name: str, default: Optional[Any]) -> Optional[str]:
            if default is not None:
                return os.getenv(name, str(default))
            else:
                return os.getenv(name)

        def to_bool(value: Optional[str]) -> bool:
            if value is None:
                return False
            return value.lower() in ["true", "1"]

        def to_int(value: Optional[str]) -> int:
            if value is None:
                return 0
            return int(value)

        def to_float(value: Optional[str]) -> float:
            if value is None:
                return 0.0
            return float(value)

        config = LMCacheEngineConfig.from_defaults(remote_url=None,
                                                   remote_serde=None)
        config.chunk_size = to_int(
            parse_env(get_env_name("chunk_size"), config.chunk_size))
        config.local_cpu = to_bool(
            parse_env(get_env_name("local_cpu"), config.local_cpu))
        config.max_local_cpu_size = to_float(
            parse_env(get_env_name("max_local_cpu_size"),
                      config.max_local_cpu_size))
        config.local_disk = parse_env(get_env_name("local_disk"),
                                      config.local_disk)
        config.max_local_disk_size = to_float(
            parse_env(get_env_name("max_local_disk_size"),
                      config.max_local_disk_size))
        config.remote_disk = parse_env(get_env_name("remote_disk"),
                                       config.remote_disk)
        config.max_remote_disk_size = to_float(
            parse_env(get_env_name("max_remote_disk_size"),
                      config.max_remote_disk_size))
        config.remote_url = parse_env(get_env_name("remote_url"),
                                      config.remote_url)
        config.remote_serde = parse_env(get_env_name("remote_serde"),
                                        config.remote_serde)
        config.save_decode_cache = to_bool(
            parse_env(get_env_name("save_decode_cache"),
                      config.save_decode_cache))
        config.enable_blending = to_bool(
            parse_env(get_env_name("enable_blending"), config.enable_blending))
        config.blend_recompute_ratio = to_float(
            parse_env(get_env_name("blend_recompute_ratio"),
                      config.blend_recompute_ratio))
        config.blend_min_tokens = to_int(
            parse_env(get_env_name("blend_min_tokens"),
                      config.blend_min_tokens))
        config.alpha = to_float(parse_env(get_env_name("alpha"), config.alpha))
        config.policy = parse_env(get_env_name("policy"), config.policy)
        config.rate = to_float(parse_env(get_env_name("rate"), config.rate))
        config.compression = parse_env(get_env_name("compression"),
                                        config.compression)
        config.dataset_csv = parse_env(get_env_name("dataset_csv"),
                                       config.dataset_csv)
        config.method_output_csv = parse_env(get_env_name("method_output_csv"),
                                              config.method_output_csv)
        return config

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
            blend_add_special_in_precomp=False
        )
