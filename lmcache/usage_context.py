# SPDX-License-Identifier: Apache-2.0
# Standard
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import platform
import subprocess
import threading

# Third Party
import cpuinfo
import pkg_resources
import psutil
import requests
import torch

# First Party
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.connections import global_http_connection
from lmcache.logging import init_logger

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
        device_property = torch.cuda.get_device_properties(0)
        gpu_count = torch.cuda.device_count()
        gpu_type = device_property.name
        gpu_memory_per_device = device_property.total_memory
        return gpu_count, gpu_type, gpu_memory_per_device

    def _get_source(self):
        path = "/proc/1/cgroup"
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if "docker" in line:
                        return "DOCKER"
        try:
            _ = pkg_resources.get_distribution("LMCache")
            return "PIP"
        except pkg_resources.DistributionNotFound:
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


def InitializeUsageContext(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    local_log: Optional[str] = None,
):
    server_url = "http://stats.lmcache.ai:8080/endpoint"
    if os.getenv("LMCACHE_TRACK_USAGE") == "false":
        return None
    else:
        logger.info("Initializing usage context.")
        return UsageContext(server_url, config, metadata, local_log)
