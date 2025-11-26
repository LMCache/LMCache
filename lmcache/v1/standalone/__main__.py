# SPDX-License-Identifier: Apache-2.0
"""
LMCache Standalone Starter

A standalone starter for LMCacheEngine that:
- Loads configuration from YAML file or environment variables
- Supports command-line parameter overrides
- Starts a real LMCacheEngine instance
- Works without vLLM or GPU
- Supports all backend types (CPU, Disk, P2P, Remote, etc.)
- Optionally starts internal API server for remote access
"""

# Standard
from typing import Any, Dict, Optional, Tuple
import argparse
import asyncio
import os
import signal
import sys

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.internal_api_server.api_server import InternalAPIServer

logger = init_logger(__name__)


class LMCacheStandaloneStarter:
    """Standalone starter for LMCacheEngine"""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ):
        self.config = config
        self.metadata = metadata

        # Create objects in constructor for better error handling
        instance_id = self.config.lmcache_instance_id
        self.lmcache_engine = LMCacheEngineBuilder.get_or_create(
            instance_id=instance_id,
            config=self.config,
            metadata=self.metadata,
            gpu_connector=None,
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
        )

        # Create API server in constructor
        self.api_server = InternalAPIServer(self)  # type: ignore[arg-type]

        self.running = False

    def start(self) -> LMCacheEngine:
        """Start the LMCache engine"""
        logger.info("=" * 80)
        logger.info("Starting LMCache Standalone Engine")
        logger.info("=" * 80)

        logger.info(f"Configuration: {self.config}")
        logger.info(f"Metadata: {self.metadata}")

        instance_id = self.config.lmcache_instance_id
        logger.info(f"Starting LMCache engine with instance ID: {instance_id}")

        # Initialize the engine
        self.lmcache_engine.post_init()
        logger.info("LMCache engine post-initialized")

        # Start internal API server
        self.api_server.start()

        self.running = True
        return self.lmcache_engine

    def stop(self):
        """Stop the LMCache engine"""
        if not self.running:
            return

        logger.info("Stopping LMCache engine...")
        self.running = False

        if self.api_server:
            logger.info("Stopping internal API server...")
            self.api_server.stop()

        if self.lmcache_engine:
            logger.info("Closing LMCache engine...")
            self.lmcache_engine.close()

            instance_id = self.config.lmcache_instance_id
            LMCacheEngineBuilder.destroy(instance_id)
            logger.info(f"Engine instance {instance_id} destroyed")

        logger.info("LMCache engine stopped")

    async def run_forever(self):
        """Keep the engine running"""
        logger.info("=" * 80)
        logger.info("LMCache engine is running")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)

        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()


def load_config(config_file: Optional[str] = None) -> LMCacheEngineConfig:
    """Load configuration from file or environment"""
    config_file = config_file or os.getenv("LMCACHE_CONFIG_FILE")

    if config_file:
        logger.info(f"Loading configuration from file: {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)
    else:
        logger.info("No config file specified, loading from environment variables")
        config = LMCacheEngineConfig.from_env()

    config.validate()
    config.log_config()

    return config


def override_config_from_dict(config: LMCacheEngineConfig, overrides: Dict[str, Any]):
    """Override configuration with dictionary"""
    for key, value in overrides.items():
        if hasattr(config, key):
            old_value = getattr(config, key)
            setattr(config, key, value)
            if old_value != value:
                logger.info(f"Override config: {key} = {value} (was {old_value})")
        else:
            logger.warning(f"Unknown config key: {key}, ignoring")


def parse_kv_shape(shape_str: str) -> Tuple[int, int, int, int, int]:
    """Parse KV shape from string like '32,2,256,32,128'"""
    try:
        parts = tuple(int(x.strip()) for x in shape_str.split(","))
        if len(parts) != 5:
            raise ValueError(
                f"kv_shape must have exactly 5 dimensions, got {len(parts)}"
            )
        return parts  # type: ignore[return-value]
    except ValueError as e:
        raise ValueError(f"Invalid kv_shape format: {shape_str}. Error: {e}") from e


def create_metadata(
    model_name: str,
    worker_id: int,
    world_size: int,
    kv_dtype: torch.dtype,
    kv_shape: Tuple[int, int, int, int, int],
    use_mla: bool,
    fmt: str = "vllm",
) -> LMCacheEngineMetadata:
    """Create engine metadata"""
    metadata = LMCacheEngineMetadata(
        model_name=model_name,
        world_size=world_size,
        worker_id=worker_id,
        fmt=fmt,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
        role="worker",
    )

    return metadata


def parse_extra_params(extra_args: list) -> Dict[str, Any]:
    """Parse extra parameters in key=value format"""
    params = {}
    for arg in extra_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
            try:
                if value.lower() in ("true", "false"):
                    params[key] = value.lower() == "true"
                elif value.isdigit():
                    params[key] = int(value)
                elif value.replace(".", "", 1).isdigit():
                    params[key] = float(value)
                else:
                    params[key] = value
            except ValueError:
                params[key] = value
            logger.info(f"Extra parameter: {key} = {params[key]}")
    return params


def setup_signal_handlers(starter: LMCacheStandaloneStarter):
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        starter.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="LMCache Standalone Starter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to LMCache configuration file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="standalone_model",
        help="Model name for cache identification",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker ID for distributed setup",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total number of workers",
    )
    parser.add_argument(
        "--kv-dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="KV cache data type",
    )
    parser.add_argument(
        "--kv-shape",
        type=str,
        default="32,2,256,32,128",
        help="KV cache shape as comma-separated integers (e.g., '32,2,256,32,128')",
    )
    parser.add_argument(
        "--use-mla",
        action="store_true",
        help="Enable MLA (Multi-Level Attention)",
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default="vllm",
        help="Cache format (default: vllm)",
    )

    args, extra = parser.parse_known_args()
    args.extra_params = parse_extra_params(extra)

    return args


def main():
    """Main entry point"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("LMCache Standalone Starter")
    logger.info("=" * 80)

    try:
        config_path = args.config or os.getenv("LMCACHE_CONFIG_FILE")
        if config_path:
            logger.info(f"Loading LMCache config file: {config_path}")
            config = LMCacheEngineConfig.from_file(config_path)
            # Allow environment variables to override file settings
            config.update_config_from_env()
        else:
            logger.info("No config file specified, loading from environment variables.")
            config = LMCacheEngineConfig.from_env()
        # Override with any extra command-line parameters
        if args.extra_params:
            override_config_from_dict(config, args.extra_params)

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        kv_dtype = dtype_map.get(args.kv_dtype, torch.float16)

        kv_shape = parse_kv_shape(args.kv_shape)
        logger.info(f"Using KV shape: {kv_shape}")

        metadata = create_metadata(
            model_name=args.model_name,
            worker_id=args.worker_id,
            world_size=args.world_size,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=args.use_mla,
            fmt=args.fmt,
        )

        starter = LMCacheStandaloneStarter(config, metadata)
        setup_signal_handlers(starter)

        starter.start()
        asyncio.run(starter.run_forever())

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("LMCache Standalone Starter stopped")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
