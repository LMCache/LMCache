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
from typing import Any, Optional
import json
import os
import re

# Third Party
import yaml

# First Party
from lmcache.logging import init_logger
import lmcache.config as orig_config

logger = init_logger(__name__)


def _parse_local_disk(local_disk) -> Optional[str]:
    match local_disk:
        case None:
            local_disk_path = None
        case path if re.match(r"file://(.*)/", path):
            local_disk_path = path[7:]
        case _:
            local_disk_path = local_disk
    return local_disk_path


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
    remote_serde: Optional[str]  # Can be "naive" or "cachegen"

    use_layerwise: bool  # whether to use layerwise pipelining

    save_decode_cache: bool  # whether to store decode kv cache

    # Blending related configurations
    enable_blending: bool  # whether to enable blending
    blend_recompute_ratio: float  # the ratio of blending recompute
    blend_min_tokens: int  # the minimum number of tokens for blending
    blend_special_str: str = " # # "  # the separator for blending

    # P2P related configurations
    enable_p2p: bool = False  # whether to enable peer-to-peer sharing
    lookup_url: Optional[str] = None  # the url of the lookup server
    distributed_url: Optional[str] = None  # the url of the distributed server

    # Error handling related configurations
    error_handling: bool = False  # whether to enable error handling

    # Controller related configurations
    enable_controller: Optional[bool] = False  # whether to enable controller
    # the id of the lmcache instance
    lmcache_instance_id: str = "lmcache_default_instance"
    # controller url
    controller_url: Optional[str] = None
    # lmcache worker url
    # NOTE: port number will add `worker_id`
    lmcache_worker_port: Optional[int] = None

    # (Optional) Nixl configurations
    # whether to enable Nixl
    enable_nixl: Optional[bool] = False
    # Role: sender or receiver
    nixl_role: Optional[str] = None
    # The host of the nixl receiver
    nixl_receiver_host: Optional[str] = None
    # The BASE port of the nixl receiver,
    # real port is nixl_receiver_port + WORKER_RANK
    nixl_receiver_port: Optional[int] = None
    # The transport buffer size of nixl in bytes
    nixl_buffer_size: Optional[int] = None
    # The device that nixl uses
    nixl_buffer_device: Optional[str] = None
    # HACK: explicit option to enable/disable nixl GC before it's mature enough
    nixl_enable_gc: Optional[bool] = False

    # The url of the actual remote lmcache instance for auditing
    audit_actual_remote_url: Optional[str] = None

    # The path under the WekaFS mount that the cache will be stored
    weka_path: Optional[str] = None
    # (Optional) The path under the File-based backend cache will be stored
    gds_path: Optional[str] = None
    # (Optional) GDS/CuFile related configurations
    # Size of CuFile Buffer in MiB
    cufile_buffer_size: Optional[int] = None

    # The extra config
    extra_config: Optional[dict] = None

    # By default, all chunks are saved
    # But in some scenarios, such as reuse, save unfull chunk is unnecessary
    save_unfull_chunk: bool = True

    # Set timeout(seconds) to improve lmcache stability
    # when calling get_blocking method in remote backend
    blocking_timeout_secs: int = 10

    @staticmethod
    def from_defaults(
        chunk_size: int = 256,
        local_cpu: bool = True,
        max_local_cpu_size: float = 5.0,
        local_disk: Optional[str] = None,
        max_local_disk_size: int = 0,
        remote_url: Optional[str] = "lm://localhost:65432",
        remote_serde: Optional[str] = "naive",
        use_layerwise: bool = False,
        save_decode_cache: bool = False,
        enable_blending: bool = False,
        blend_recompute_ratio: float = 0.15,
        blend_min_tokens: int = 256,
        blend_special_str: str = " # # ",
        enable_p2p: bool = False,
        lookup_url: Optional[str] = None,
        distributed_url: Optional[str] = None,
        error_handling: bool = False,
        enable_controller: Optional[bool] = False,
        lmcache_instance_id: str = "lmcache_default_instance",
        controller_url: Optional[str] = None,
        lmcache_worker_port: Optional[int] = None,
        enable_nixl: Optional[bool] = False,
        nixl_role: Optional[str] = None,
        nixl_receiver_host: Optional[str] = None,
        nixl_receiver_port: Optional[int] = None,
        nixl_buffer_size: Optional[int] = None,
        nixl_buffer_device: Optional[str] = None,
        nixl_enable_gc: Optional[bool] = False,
        audit_actual_remote_url: Optional[str] = None,
        weka_path: Optional[str] = None,
        gds_path: Optional[str] = None,
        cufile_buffer_size: Optional[int] = None,
        extra_config: Optional[dict] = None,
        save_unfull_chunk: bool = True,
        blocking_timeout_secs: int = 10,
    ) -> "LMCacheEngineConfig":
        # TODO (ApostaC): Add nixl config
        return LMCacheEngineConfig(
            chunk_size,
            local_cpu,
            max_local_cpu_size,
            local_disk,
            max_local_disk_size,
            remote_url,
            remote_serde,
            use_layerwise,
            save_decode_cache,
            enable_blending,
            blend_recompute_ratio,
            blend_min_tokens,
            blend_special_str,
            enable_p2p,
            lookup_url,
            distributed_url,
            error_handling,
            enable_controller,
            lmcache_instance_id,
            controller_url,
            lmcache_worker_port,
            enable_nixl,
            nixl_role,
            nixl_receiver_host,
            nixl_receiver_port,
            nixl_buffer_size,
            nixl_buffer_device,
            nixl_enable_gc,
            audit_actual_remote_url,
            weka_path,
            gds_path,
            cufile_buffer_size,
            extra_config,
            save_unfull_chunk,
            blocking_timeout_secs,
        ).validate()

    @staticmethod
    def from_legacy(
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
        if backend == "cpu":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = None
            max_local_disk_size = 0
            remote_url = None
        elif backend == "local_disk":
            local_cpu = False
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
            remote_url = None
        elif backend == "local_cpu_disk":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
            remote_url = None
        elif backend == "remote":
            local_cpu = False
            max_local_cpu_size = 5
            local_disk = None
        elif backend == "local_cpu_remote":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = None
        elif backend == "local_disk_remote":
            local_cpu = False
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
        elif backend == "local_cpu_disk_remote":
            local_cpu = True
            max_local_cpu_size = 5
            local_disk = "/local/disk_test/local_disk/"
            max_local_disk_size = 5
        else:
            raise ValueError(f"Invalid backend: {backend}")
        return (
            LMCacheEngineConfig(
                chunk_size=chunk_size,
                local_cpu=local_cpu,
                max_local_cpu_size=max_local_cpu_size,
                local_disk=local_disk,
                max_local_disk_size=max_local_disk_size,
                remote_url=remote_url,
                remote_serde=remote_serde,
                use_layerwise=use_layerwise,
                save_decode_cache=save_decode_cache,
                enable_blending=enable_blending,
                blend_recompute_ratio=blend_recompute_ratio,
                blend_min_tokens=blend_min_tokens,
                blend_special_str=blend_special_str,
                enable_p2p=enable_p2p,
                lookup_url=lookup_url,
                distributed_url=distributed_url,
                error_handling=error_handling,
                save_unfull_chunk=save_unfull_chunk,
            )
            .validate()
            .log_config()
        )

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
        remote_serde = config.get("remote_serde", "naive")

        use_layerwise = config.get("use_layerwise", False)

        save_decode_cache = config.get("save_decode_cache", False)

        enable_blending = config.get("enable_blending", False)
        blend_recompute_ratio = config.get("blend_recompute_ratio", 0.15)
        blend_min_tokens = config.get("blend_min_tokens", 256)
        blend_special_str = config.get("blend_special_str", " # # ")

        enable_p2p = config.get("enable_p2p", False)
        lookup_url = config.get("lookup_url", None)
        distributed_url = config.get("distributed_url", None)

        error_handling = config.get("error_handling", False)

        enable_controller = config.get("enable_controller", False)
        lmcache_instance_id = config.get(
            "lmcache_instance_id", "lmcache_default_instance"
        )
        controller_url = config.get("controller_url", None)
        lmcache_worker_port = config.get("lmcache_worker_port", None)

        enable_nixl = config.get("enable_nixl", False)
        nixl_role = config.get("nixl_role", None)
        nixl_receiver_host = config.get("nixl_receiver_host", None)
        nixl_receiver_port = config.get("nixl_receiver_port", None)
        nixl_buffer_size = config.get("nixl_buffer_size", None)
        nixl_buffer_device = config.get("nixl_buffer_device", None)
        nixl_enable_gc = config.get("nixl_enable_gc", False)

        extra_config = config.get("extra_config", None)
        if extra_config is not None:
            assert isinstance(extra_config, dict), "extra_config must be a dict"

        # Try getting "legacy" nixl config
        if nixl_receiver_host is None:
            nixl_receiver_host = config.get("nixl_peer_host", None)
            if nixl_receiver_host is not None:
                logger.warning(
                    "nixl_peer_host is deprecated, please use "
                    "nixl_receiver_host in the config file instead"
                )

        if nixl_receiver_port is None:
            nixl_receiver_port = config.get("nixl_peer_port", None)
            if nixl_receiver_port is not None:
                logger.warning(
                    "nixl_peer_port is deprecated, please use "
                    "nixl_receiver_port in the config file instead"
                )

        audit_actual_remote_url = config.get("audit_actual_remote_url", None)

        weka_path = config.get("weka_path", None)
        gds_path = config.get("gds_path", None)
        cufile_buffer_size = config.get("cufile_buffer_size", None)

        save_unfull_chunk = config.get("save_unfull_chunk", True)

        blocking_timeout_secs = config.get("blocking_timeout_secs", 10)

        local_disk_path = _parse_local_disk(local_disk)

        match remote_url:
            case None:
                pass
            case url if re.match(r"(.*)://(.*):(\d+)", url):
                pass
            case _:
                raise ValueError(f"Invalid remote storage url: {remote_url}")

        return (
            LMCacheEngineConfig(
                chunk_size,
                local_cpu,
                max_local_cpu_size,
                local_disk_path,
                max_local_disk_size,
                remote_url,
                remote_serde,
                use_layerwise,
                save_decode_cache,
                enable_blending,
                blend_recompute_ratio,
                blend_min_tokens,
                blend_special_str,
                enable_p2p,
                lookup_url,
                distributed_url,
                error_handling,
                enable_controller,
                lmcache_instance_id,
                controller_url,
                lmcache_worker_port,
                enable_nixl,
                nixl_role,
                nixl_receiver_host,
                nixl_receiver_port,
                nixl_buffer_size,
                nixl_buffer_device,
                nixl_enable_gc,
                audit_actual_remote_url,
                weka_path,
                gds_path,
                cufile_buffer_size,
                extra_config,
                save_unfull_chunk,
                blocking_timeout_secs,
            )
            .validate()
            .log_config()
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

        def to_dict(value: Optional[str]) -> Optional[dict]:
            if value is None:
                return None
            res = json.loads(value)
            assert isinstance(res, dict), "value must be a dict"
            return res

        config = LMCacheEngineConfig.from_defaults(remote_url=None, remote_serde=None)
        config.chunk_size = to_int(
            parse_env(get_env_name("chunk_size"), config.chunk_size)
        )
        config.local_cpu = to_bool(
            parse_env(get_env_name("local_cpu"), config.local_cpu)
        )
        config.max_local_cpu_size = to_float(
            parse_env(get_env_name("max_local_cpu_size"), config.max_local_cpu_size)
        )
        config.local_disk = _parse_local_disk(
            parse_env(get_env_name("local_disk"), config.local_disk)
        )
        config.max_local_disk_size = to_float(
            parse_env(get_env_name("max_local_disk_size"), config.max_local_disk_size)
        )
        config.remote_url = parse_env(get_env_name("remote_url"), config.remote_url)
        config.remote_serde = parse_env(
            get_env_name("remote_serde"), config.remote_serde
        )

        config.use_layerwise = to_bool(
            parse_env(get_env_name("use_layerwise"), config.use_layerwise)
        )

        config.save_decode_cache = to_bool(
            parse_env(get_env_name("save_decode_cache"), config.save_decode_cache)
        )

        config.enable_blending = to_bool(
            parse_env(get_env_name("enable_blending"), config.enable_blending)
        )
        config.blend_recompute_ratio = to_float(
            parse_env(
                get_env_name("blend_recompute_ratio"),
                config.blend_recompute_ratio,
            )
        )
        config.blend_min_tokens = to_int(
            parse_env(get_env_name("blend_min_tokens"), config.blend_min_tokens)
        )
        blend_special_str = parse_env(
            get_env_name("blend_special_str"), config.blend_special_str
        )
        assert blend_special_str is not None
        config.blend_special_str = blend_special_str

        config.enable_p2p = to_bool(
            parse_env(get_env_name("enable_p2p"), config.enable_p2p)
        )
        config.lookup_url = parse_env(get_env_name("lookup_url"), config.lookup_url)
        config.distributed_url = parse_env(
            get_env_name("distributed_url"), config.distributed_url
        )

        config.error_handling = to_bool(
            parse_env(get_env_name("error_handling"), config.error_handling)
        )

        config.enable_controller = to_bool(
            parse_env(get_env_name("enable_controller"), config.enable_controller)
        )
        lmcache_instance_id = parse_env(
            get_env_name("lmcache_instance_id"), "lmcache_default_instance"
        )
        assert lmcache_instance_id is not None
        config.lmcache_instance_id = lmcache_instance_id
        config.controller_url = parse_env(
            get_env_name("controller_url"), config.controller_url
        )
        config.lmcache_worker_port = to_int(
            parse_env(get_env_name("lmcache_worker_port"), config.lmcache_worker_port)
        )

        config.enable_nixl = to_bool(
            parse_env(get_env_name("enable_nixl"), config.enable_nixl)
        )
        config.nixl_role = parse_env(get_env_name("nixl_role"), config.nixl_role)
        config.nixl_receiver_host = parse_env(
            get_env_name("nixl_receiver_host"), config.nixl_receiver_host
        )
        config.nixl_receiver_port = to_int(
            parse_env(get_env_name("nixl_receiver_port"), config.nixl_receiver_port)
        )
        config.nixl_buffer_size = to_int(
            parse_env(get_env_name("nixl_buffer_size"), config.nixl_buffer_size)
        )
        config.nixl_buffer_device = parse_env(
            get_env_name("nixl_buffer_device"), config.nixl_buffer_device
        )
        config.nixl_enable_gc = to_bool(
            parse_env(get_env_name("nixl_enable_gc"), config.nixl_enable_gc)
        )

        # Try getting "legacy" nixl config
        if config.nixl_receiver_host is None:
            config.nixl_receiver_host = parse_env(
                get_env_name("nixl_peer_host"), config.nixl_receiver_host
            )
            if config.nixl_receiver_host is not None:
                logger.warning(
                    "LMCACHE_NIXL_PEER_HOST is deprecated, please use "
                    "LMCACHE_NIXL_RECEIVER_HOST environment variable instead"
                )

        if config.nixl_receiver_port is None:
            config.nixl_receiver_port = to_int(
                parse_env(get_env_name("nixl_peer_port"), config.nixl_receiver_port)
            )
            if config.nixl_receiver_port is not None:
                logger.warning(
                    "LMCACHE_NIXL_PEER_PORT is deprecated, please use "
                    "LMCACHE_NIXL_RECEIVER_PORT environment variable instead"
                )

        config.audit_actual_remote_url = parse_env(
            get_env_name("audit_actual_remote_url"),
            config.audit_actual_remote_url,
        )

        config.weka_path = parse_env(
            get_env_name("weka_path"),
            config.weka_path,
        )
        config.gds_path = parse_env(
            get_env_name("gds_path"),
            config.gds_path,
        )
        config.cufile_buffer_size = to_int(
            parse_env(
                get_env_name("cufile_buffer_size"),
                config.cufile_buffer_size,
            )
        )
        config.extra_config = to_dict(parse_env(get_env_name("extra_config"), None))
        config.save_unfull_chunk = to_bool(
            parse_env(get_env_name("save_unfull_chunk"), config.save_unfull_chunk)
        )
        config.blocking_timeout_secs = to_int(
            parse_env(
                get_env_name("blocking_timeout_secs"), config.blocking_timeout_secs
            )
        )
        return config.validate().log_config()

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

    def validate(self) -> "LMCacheEngineConfig":
        """Validate the config"""
        if self.enable_p2p:
            assert self.lookup_url is not None
            assert self.distributed_url is not None

        if self.enable_nixl:
            assert self.nixl_role is not None
            assert self.nixl_receiver_host is not None
            assert self.nixl_receiver_port is not None
            assert self.nixl_buffer_size is not None
            assert self.nixl_buffer_device is not None
            assert self.nixl_enable_gc is not None

            assert self.local_cpu is False, "Nixl only supports local_cpu=False"
            assert self.max_local_cpu_size == 0, (
                "Nixl only supports max_local_cpu_size=0"
            )

            assert self.local_disk is None, "Nixl only supports local_disk=None"

            assert self.remote_url is None, "Nixl only supports remote_url=None"

            assert self.save_decode_cache is False, (
                "Nixl only supports save_decode_cache=False"
            )
            assert self.enable_p2p is False, "Nixl only supports enable_p2p=False"

        return self

    def log_config(self) -> "LMCacheEngineConfig":
        """log the configuration in LMCache"""
        config_dict = {
            "chunk_size": self.chunk_size,
            "cufile_buffer_size": self.cufile_buffer_size,
            "local_cpu": self.local_cpu,
            "max_local_cpu_size": f"{self.max_local_cpu_size} GB",
            "local_disk": self.local_disk,
            "max_local_disk_size": f"{self.max_local_disk_size} GB",
            "remote_url": self.remote_url,
            "remote_serde": self.remote_serde,
            "use_layerwise": self.use_layerwise,
            "save_decode_cache": self.save_decode_cache,
            "enable_blending": self.enable_blending,
            "blend_recompute_ratio": self.blend_recompute_ratio,
            "blend_min_tokens": self.blend_min_tokens,
            "enable_p2p": self.enable_p2p,
            "lookup_url": self.lookup_url,
            "distributed_url": self.distributed_url,
            "error_handling": self.error_handling,
            "enable_controller": self.enable_controller,
            "lmcache_instance_id": self.lmcache_instance_id,
            "enable_nixl": self.enable_nixl,
            "nixl_role": self.nixl_role,
            "nixl_receiver_host": self.nixl_receiver_host,
            "nixl_receiver_port": self.nixl_receiver_port,
            "nixl_buffer_size": self.nixl_buffer_size,
            "nixl_buffer_device": self.nixl_buffer_device,
            "nixl_enable_gc": self.nixl_enable_gc,
            "weka_path": self.weka_path,
            "gds_path": self.gds_path,
            "extra_config": self.extra_config,
            "save_unfull_chunk": self.save_unfull_chunk,
            "blocking_timeout_secs": self.blocking_timeout_secs,
        }
        logger.info(f"LMCache Configuration: {config_dict}")

        return self
