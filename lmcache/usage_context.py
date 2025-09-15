# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urljoin
import dataclasses
import importlib.metadata
import os
import platform
import subprocess
import threading
import time

# Third Party
import cpuinfo
import numpy as np
import psutil
import requests
import torch

# First Party
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.connections import global_http_connection
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.observability import LMCacheStats


logger = init_logger(__name__)


class EnvMessage:
    def __init__(
        self,
        provider,
        num_cpu,
        cpu_type,
        cpu_family_model_stepping,
        total_memory,
        architecture,
        platforms,
        gpu_count,
        gpu_type,
        gpu_memory_per_device,
        source,
    ):
        self.provider = provider
        self.num_cpu = num_cpu
        self.cpu_type = cpu_type
        self.cpu_family_model_stepping = cpu_family_model_stepping
        self.total_memory = total_memory
        self.architecture = architecture
        self.platforms = platforms
        self.gpu_count = gpu_count
        self.gpu_type = gpu_type
        self.gpu_memory_per_device = gpu_memory_per_device
        self.source = source


class EngineMessage:
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.chunksize = config.chunk_size
        self.local_device = config.local_device
        self.max_local_cache_size = config.max_local_cache_size
        self.remote_url = config.remote_url
        self.remote_serde = config.remote_serde
        self.pipelined_backend = config.pipelined_backend
        self.save_decode_cache = config.save_decode_cache
        self.enable_blending = config.enable_blending
        self.blend_recompute_ratio = config.blend_recompute_ratio
        self.blend_min_tokens = config.blend_min_tokens
        self.model_name = metadata.model_name
        self.world_size = metadata.world_size
        self.worker_id = metadata.worker_id
        self.fmt = metadata.fmt
        self.kv_dtype = metadata.kv_dtype
        self.kv_shape = metadata.kv_shape


class MetadataMessage:
    def __init__(self, start_time, duration):
        self.start_time = start_time
        self.duration = duration


@dataclass
# follows naming convention in usage_context.py
class ContinuousContextMessage:
    interval_num_stored_tokens: int
    interval_num_hit_tokens: int
    interval_stored_kv_size: int
    message_type: str = "ContinuousContextMessage"


class UsageContext:
    def __init__(
        self,
        server_url: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        local_log: Optional[str] = None,
    ):
        self.server_url = server_url
        self.config = config
        self.metadata = metadata
        self.start_time = datetime.now()
        self.local_log = local_log

        self.send_env_message()
        self.send_engine_message()
        t = threading.Thread(target=self.send_metadata_message)
        t.start()

    def send_message_server(self, msg, message_type):
        msg.message_type = message_type
        try:
            global_http_client = global_http_connection.get_sync_client()
            data = dict()
            for key, value in msg.__dict__.items():
                if isinstance(value, torch.dtype):
                    data[key] = str(value)
                else:
                    data[key] = value
            if self.server_url is not None:
                logger.debug("context message updated")
                global_http_client.post(self.server_url, json=data, timeout=5)
        except requests.exceptions.RequestException:
            logger.debug("Unable to send lmcache context message")

    def send_message_local(self, msg, message_type):
        if self.local_log is None:
            return
        msg.message_type = message_type
        message = ""
        for key, value in msg.__dict__.items():
            message += "{}: {}\n".format(key, value)
        message += "\n"
        with open(self.local_log, "a") as f:
            f.write(message)

    def send_env_message(self):
        env_message = self.track_env()
        self.send_message_server(env_message, "EnvMessage")
        self.send_message_local(env_message, "EnvMessage")

    def send_engine_message(self):
        engine_message = self.track_engine()
        self.send_message_server(engine_message, "EngineMessage")
        self.send_message_local(engine_message, "EngineMessage")

    def send_metadata_message(self):
        metadata_message = self.track_metadata()
        self.send_message_server(metadata_message, "MetadataMessage")
        self.send_message_local(metadata_message, "MetadataMessage")

    def track_env(self):
        provider = self._get_provider()
        num_cpu, cpu_type, cpu_family_model_stepping = self._get_cpu_info()
        total_memory = psutil.virtual_memory().total
        architecture = platform.architecture()
        platforms = platform.platform()
        gpu_count, gpu_type, gpu_memory_per_device = self._get_gpu_info()
        source = self._get_source()
        env_message = EnvMessage(
            provider,
            num_cpu,
            cpu_type,
            cpu_family_model_stepping,
            total_memory,
            architecture,
            platforms,
            gpu_count,
            gpu_type,
            gpu_memory_per_device,
            source,
        )
        return env_message

    def track_engine(self):
        engine_message = EngineMessage(self.config, self.metadata)
        return engine_message

    def track_metadata(self):
        start_time = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        interval = datetime.now() - self.start_time
        duration = interval.total_seconds()
        return MetadataMessage(start_time, duration)

    def _get_provider(self):
        vendor_files = [
            "/sys/class/dmi/id/product_version",
            "/sys/class/dmi/id/bios_vendor",
            "/sys/class/dmi/id/product_name",
            "/sys/class/dmi/id/chassis_asset_tag",
            "/sys/class/dmi/id/sys_vendor",
        ]
        # Mapping of identifiable strings to cloud providers
        cloud_identifiers = {
            "amazon": "AWS",
            "microsoft corporation": "AZURE",
            "google": "GCP",
            "oraclecloud": "OCI",
        }

        for vendor_file in vendor_files:
            path = Path(vendor_file)
            if path.is_file():
                file_content = path.read_text().lower()
                for identifier, provider in cloud_identifiers.items():
                    if identifier in file_content:
                        return provider

        # Try detecting through environment variables
        env_to_cloud_provider = {
            "RUNPOD_DC_ID": "RUNPOD",
        }
        for env_var, provider in env_to_cloud_provider.items():
            if os.environ.get(env_var):
                return provider

        return "UNKNOWN"

    def _get_cpu_info(self):
        info = cpuinfo.get_cpu_info()
        num_cpu = info.get("count", None)
        cpu_type = info.get("brand_raw", "")
        cpu_family_model_stepping = ",".join(
            [
                str(info.get("family", "")),
                str(info.get("model", "")),
                str(info.get("stepping", "")),
            ]
        )
        return num_cpu, cpu_type, cpu_family_model_stepping

    def _get_gpu_info(self):
        if torch.cuda.is_available():
            device_property = torch.cuda.get_device_properties(0)
            gpu_count = torch.cuda.device_count()
            gpu_type = device_property.name
            gpu_memory_per_device = device_property.total_memory
        else:
            gpu_count = psutil.cpu_count(logical=False)
            gpu_type = platform.processor()
            gpu_memory_per_device = psutil.virtual_memory()
        return gpu_count, gpu_type, gpu_memory_per_device

    def _get_source(self):
        path = "/proc/1/cgroup"
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if "docker" in line:
                        return "DOCKER"
        try:
            _ = importlib.metadata.distribution("LMCache")
            return "PIP"
        except importlib.metadata.PackageNotFoundError:
            pass
        try:
            result = subprocess.run(
                ["conda", "list", "LMCache"], capture_output=True, text=True
            )
            if "LMCache" in result.stdout:
                return "CONDA"
        except FileNotFoundError:
            pass

        return "UNKNOWN"


class ContinuousUsageContext:
    _instance = None

    def __init__(self, metadata: LMCacheEngineMetadata):
        self.metadata: LMCacheEngineMetadata = metadata
        self.server_url: str = urljoin(
            os.getenv("LMCACHE_USAGE_TRACK_URL", "http://stats.lmcache.ai:8080"),
            "cache-usage",
        )
        logger.info(f"sending cache usage stats to {self.server_url}")
        self.min_logging_interval: int = int(
            os.getenv("LMCACHE_USAGE_TRACK_INTERVAL", "600")
        )
        # send the first message immediately after init
        self.last_logged_ts: float = -1

        self.interval_num_hit_tokens: int = 0
        self.interval_num_stored_tokens: int = 0
        self.kv_sz_per_token_bytes: int = int(
            np.prod(self.metadata.kv_shape)
            * self.metadata.kv_dtype.itemsize
            / self.metadata.kv_shape[2]
        )

    @staticmethod
    def GetOrCreate(metadata: LMCacheEngineMetadata) -> "ContinuousUsageContext":
        if ContinuousUsageContext._instance is None:
            ContinuousUsageContext._instance = ContinuousUsageContext(metadata)
        if ContinuousUsageContext._instance.metadata != metadata:
            logger.error(
                "ContinuousUsageContext instance already created with"
                "different metadata. This should not happen except "
                "in test."
            )
        return ContinuousUsageContext._instance

    def send_caching_message(self):
        msg: ContinuousContextMessage = ContinuousContextMessage(
            interval_stored_kv_size=int(
                self.kv_sz_per_token_bytes * self.interval_num_stored_tokens
            ),
            interval_num_hit_tokens=int(self.interval_num_hit_tokens),
            interval_num_stored_tokens=int(self.interval_num_stored_tokens),
        )
        try:
            global_http_client = global_http_connection.get_sync_client()
            if self.server_url is not None:
                logger.debug("caching usage message sent.")
                global_http_client.post(
                    f"{self.server_url}", json=dataclasses.asdict(msg), timeout=5
                )
            self.interval_num_hit_tokens = 0
            self.interval_num_stored_tokens = 0
        except requests.exceptions.RequestException:
            logger.debug("Unable to send lmcache caching usage message...")

    def incr_or_send_stats(self, stats: "LMCacheStats"):
        # no-ops when user disable usage tracking
        if os.getenv("LMCACHE_TRACK_USAGE") == "false":
            return None

        self.interval_num_hit_tokens += stats.interval_hit_tokens
        self.interval_num_stored_tokens += stats.interval_stored_tokens

        cur_ts: float = time.monotonic()
        if cur_ts - self.last_logged_ts >= self.min_logging_interval:
            self.send_caching_message()
            self.last_logged_ts = cur_ts


def InitializeUsageContext(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    local_log: Optional[str] = None,
):
    server_url = urljoin(
        os.getenv("LMCACHE_USAGE_TRACK_URL", "http://stats.lmcache.ai:8080"), "context"
    )
    if os.getenv("LMCACHE_TRACK_USAGE") == "false":
        return None
    else:
        logger.info("Initializing usage context.")
        return UsageContext(server_url, config, metadata, local_log)
