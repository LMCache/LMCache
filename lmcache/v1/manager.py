# SPDX-License-Identifier: Apache-2.0
"""
LMCacheManager: A unified manager for LMCache internal components.

This module provides a clean interface to manage LMCache components lifecycle,
decoupling the vLLM adapter from internal LMCache implementation details.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Union
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import CreateGPUConnector
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.health_monitor.constants import (
    DEFAULT_PING_INTERVAL,
    PING_INTERVAL_CONFIG_KEY,
)
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

    # First Party
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)

# Engine name constant
ENGINE_NAME = "LMCacheEngine"


class LMCacheManager:
    """
    LMCacheManager manages the lifecycle of LMCache internal components.

    This class encapsulates the initialization and shutdown of:
    - LMCacheEngine
    - LookupClient / LookupServer
    - OffloadServer
    - InternalAPIServer
    - RuntimePluginLauncher
    - HealthMonitor

    This manager supports two main modes:
    - vLLM integration mode (requires vllm_config)
    - Standalone mode (requires metadata and GPU connector)
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        vllm_config: Optional["VllmConfig"] = None,
        role: str = "worker",
        connector: Optional[Any] = None,
    ):
        """
        Initialize LMCacheManager for vLLM integration mode.

        Args:
            config: LMCache engine configuration
            vllm_config: vLLM configuration (required for vLLM integration mode)
            role: The role string ("scheduler" or "worker")
            connector: Reference to LMCacheConnectorV1Impl for internal API server
        """
        self._config = config
        self._vllm_config: Optional["VllmConfig"] = vllm_config
        self._role = role
        self._connector: Any = connector

        # Components (initialized later)
        self._lmcache_engine: Optional[LMCacheEngine] = None
        self._lmcache_engine_metadata: Optional[LMCacheMetadata] = None
        self._lookup_client: Optional[LookupClientInterface] = None
        self._lookup_server: Optional[
            Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]
        ] = None
        self._offload_server: Optional[ZMQOffloadServer] = None
        self._api_server: Optional[InternalAPIServer] = None
        self._runtime_plugin_launcher: Optional[RuntimePluginLauncher] = None
        self._health_monitor: Optional[HealthMonitor] = None

        # Flag to track if initialization failed
        self._init_failed = False
        self._init_failed_reason: str = ""

        # Initialize components based on role
        try:
            self._init_components()
        except Exception as e:
            self._init_failed = True
            self._init_failed_reason = str(e)
            logger.error(
                "Failed to initialize LMCacheManager components: %s. "
                "System will operate in degraded mode (recompute).",
                e,
            )

    def _init_components(self) -> None:
        """Initialize components based on the role for vLLM mode."""
        if self._role == "scheduler":
            # Initialize vLLM scheduler components
            self._init_scheduler_components()
        else:
            # Initialize vLLM worker components
            self._init_worker_components()
        # Initialize API server and plugin launcher only on DP rank 0
        assert self._vllm_config is not None
        if self._vllm_config.parallel_config.data_parallel_rank_local == 0:
            self._init_dp_rank0_components()

    def _init_scheduler_components(self) -> None:
        """Initialize components for scheduler role."""
        # First Party
        from lmcache.integration.vllm.utils import create_lmcache_metadata
        from lmcache.observability import PrometheusLogger
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        assert self._vllm_config is not None, "vllm_config required for vLLM mode"

        if self._config.enable_scheduler_bypass_lookup:
            # Create LMCacheEngine for scheduler when bypass is enabled
            self._lmcache_engine = self._create_lmcache_engine(role="scheduler")
            self._lmcache_engine_metadata = self._lmcache_engine.metadata
        else:
            self._lmcache_engine = None
            # Create a dummy metadata for prometheus logger
            self._lmcache_engine_metadata, _ = create_lmcache_metadata(
                self._vllm_config, role="scheduler"
            )
            PrometheusLogger.GetOrCreate(
                self._lmcache_engine_metadata,
                config=self._config,
            )

        # Create lookup client
        self._lookup_client = LookupClientFactory.create_lookup_client(
            self._config,
            self._lmcache_engine_metadata,
            self._lmcache_engine,
        )

    def _init_worker_components(self) -> None:
        """Initialize components for worker role."""
        # Third Party
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        assert self._vllm_config is not None, "vllm_config required for vLLM mode"

        # Create LMCacheEngine
        self._lmcache_engine = self._create_lmcache_engine(role="worker")
        self._lmcache_engine_metadata = self._lmcache_engine.metadata

        # Create lookup server
        self._lookup_server = LookupClientFactory.create_lookup_server(
            self._lmcache_engine, self._lmcache_engine_metadata
        )

        # Create offload server
        self._offload_server = ZMQOffloadServer(
            self._lmcache_engine,
            get_tensor_model_parallel_rank(),
        )

    def _init_dp_rank0_components(self) -> None:
        """Initialize components that only run on DP rank 0."""
        assert self._vllm_config is not None, "vllm_config required for vLLM mode"

        # Start internal API server
        self._api_server = InternalAPIServer(self)

        # Create plugin launcher
        worker_id = (
            -1
            if self._lmcache_engine is None
            else self._lmcache_engine.metadata.worker_id
        )
        self._runtime_plugin_launcher = RuntimePluginLauncher(
            self._config,
            self._role,
            self._vllm_config.parallel_config.tensor_parallel_size,
            worker_id,
        )

    def _init_health_monitor(self) -> None:
        """
        Initialize the health monitor for the LMCacheManager.

        This is called during post_init after all components are initialized.
        The HealthMonitor automatically discovers and instantiates all
        HealthCheck subclasses based on the manager's role and components.
        """
        # First Party
        from lmcache.observability import PrometheusLogger
        from lmcache.v1.periodic_thread import (
            PeriodicThreadRegistry,
            ThreadLevel,
        )

        # Get ping interval from config
        ping_interval = self._config.get_extra_config_value(
            PING_INTERVAL_CONFIG_KEY, DEFAULT_PING_INTERVAL
        )

        # Create health monitor with manager - it will auto-discover health checks
        self._health_monitor = HealthMonitor(
            manager=self,
            ping_interval=ping_interval,
        )

        # Inject health monitor into engine (if exists)
        if self._lmcache_engine is not None:
            self._lmcache_engine.set_health_monitor(self._health_monitor)

        # Start the health monitor
        self._health_monitor.start()
        logger.info(
            "Health monitor initialized and started at manager level (role=%s)",
            self._role,
        )

        # Setup metrics callback for health status
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.lmcache_is_healthy.set_function(
                lambda: 1 if self.is_healthy() else 0
            )

            # Setup PeriodicThread metrics callbacks
            registry = PeriodicThreadRegistry.get_instance()

            prometheus_logger.periodic_threads_total_count.set_function(
                lambda: len(registry.get_all())
            )
            prometheus_logger.periodic_threads_running_count.set_function(
                lambda: registry.get_running_count()
            )
            prometheus_logger.periodic_threads_active_count.set_function(
                lambda: registry.get_active_count()
            )

            # Per-level metrics
            for level in ThreadLevel:
                level_name = level.value
                total_attr = f"periodic_threads_{level_name}_total"
                running_attr = f"periodic_threads_{level_name}_running"
                active_attr = f"periodic_threads_{level_name}_active"

                if hasattr(prometheus_logger, total_attr):
                    getattr(prometheus_logger, total_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["total"]
                    )
                if hasattr(prometheus_logger, running_attr):
                    getattr(prometheus_logger, running_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["running"]
                    )
                if hasattr(prometheus_logger, active_attr):
                    getattr(prometheus_logger, active_attr).set_function(
                        lambda lvl=level: registry.get_count_by_level(lvl)["active"]
                    )

    def _create_lmcache_engine(self, role: str) -> LMCacheEngine:
        """
        Create and return an LMCacheEngine instance.

        Args:
            role: The role string ("scheduler" or "worker")

        Returns:
            LMCacheEngine instance
        """
        # First Party
        from lmcache.integration.vllm.utils import (
            ENGINE_NAME,
            calculate_local_rank_and_world_size,
            mla_enabled,
        )

        if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
            return curr_engine

        assert self._vllm_config is not None, "vllm_config required for vLLM mode"

        try:
            # Third Party
            from vllm.utils.torch_utils import get_kv_cache_torch_dtype
        except ImportError:
            # Third Party
            from vllm.utils import get_kv_cache_torch_dtype
        # Third Party
        from vllm.distributed.parallel_state import get_tp_group

        model_config = self._vllm_config.model_config
        parallel_config = self._vllm_config.parallel_config
        cache_config = self._vllm_config.cache_config

        kv_dtype = get_kv_cache_torch_dtype(
            cache_config.cache_dtype, model_config.dtype
        )

        use_mla = mla_enabled(model_config)
        self._validate_mla_config(use_mla)

        # Construct kv shape
        num_layer = model_config.get_num_layers(parallel_config)
        num_draft_layers = self._calculate_draft_layers()
        num_layer += num_draft_layers
        chunk_size = self._config.chunk_size
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)

        logger.info(
            "num_layer: %d, chunk_size: %d, num_kv_head (per gpu): %d, "
            "head_size: %d, hidden_dim (D) for KV (per gpu): %d, "
            "use mla: %s, kv shape: %s, num_draft_layers: %d",
            num_layer,
            chunk_size,
            num_kv_head,
            head_size,
            num_kv_head * head_size,
            use_mla,
            kv_shape,
            num_draft_layers,
        )

        # Extract engine_id and kv_connector_extra_config from vllm_config
        engine_id = None
        kv_connector_extra_config = None
        if hasattr(self._vllm_config, "kv_transfer_config"):
            kv_transfer_config = self._vllm_config.kv_transfer_config
            if kv_transfer_config is not None:
                engine_id = getattr(kv_transfer_config, "engine_id", None)
                kv_connector_extra_config = getattr(
                    kv_transfer_config, "kv_connector_extra_config", None
                )

        # Create metadata
        local_worker_id, local_world_size = calculate_local_rank_and_world_size(
            self._vllm_config
        )
        metadata = LMCacheMetadata(
            model_name=model_config.model,
            world_size=parallel_config.world_size,
            local_world_size=local_world_size,
            worker_id=parallel_config.rank,
            local_worker_id=local_worker_id,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
            role=role,
            served_model_name=model_config.served_model_name,
            chunk_size=self._config.chunk_size,
            engine_id=engine_id,
            kv_connector_extra_config=kv_connector_extra_config,
        )

        # Get tensor parallel group
        if role == "scheduler":
            tpg = SimpleNamespace()
            tpg.broadcast = lambda tensor, src: tensor
            tpg.broadcast_object = lambda obj, src: obj
            vllm_gpu_connector = None
        else:
            tpg = get_tp_group()
            vllm_gpu_connector = CreateGPUConnector(
                self._config, metadata, EngineType.VLLM
            )

        engine = LMCacheEngineBuilder.get_or_create(
            ENGINE_NAME,
            self._config,
            metadata,
            vllm_gpu_connector,
            tpg.broadcast,
            tpg.broadcast_object,
        )

        if role == "scheduler" and self._config.enable_scheduler_bypass_lookup:
            assert engine.save_only_first_rank or self._config.get_extra_config_value(
                "remote_enable_mla_worker_id_as0", metadata.use_mla
            ), (
                "enable_scheduler_bypass_lookup is only supported with "
                "save_only_first_rank or remote_enable_mla_worker_id_as0"
            )

        return engine

    def _validate_mla_config(self, use_mla: bool) -> None:
        """Validate MLA-related configuration."""
        if use_mla and (
            self._config.remote_serde != "naive"
            and self._config.remote_serde is not None
        ):
            raise ValueError("MLA only works with naive serde mode..")

        if use_mla and self._config.use_layerwise and self._config.enable_blending:
            raise ValueError(
                "We haven't supported MLA with Cacheblend yet. Please disable blending."
            )

    def _calculate_draft_layers(self) -> int:
        """Calculate the number of draft layers for speculative decoding."""
        assert self._vllm_config is not None, "vllm_config required for vLLM mode"

        num_draft_layers = 0
        vllm_config = self._vllm_config
        model_config = vllm_config.model_config

        if vllm_config.speculative_config is not None:
            logger.info(
                "vllm_config.speculative_config: %s", vllm_config.speculative_config
            )
            if vllm_config.speculative_config.method == "deepseek_mtp":
                num_draft_layers = getattr(
                    model_config.hf_config, "num_nextn_predict_layers", 0
                )
            elif vllm_config.speculative_config.use_eagle():
                try:
                    draft_model_config = (
                        vllm_config.speculative_config.draft_model_config
                    )
                    num_draft_layers = draft_model_config.get_num_layers(
                        vllm_config.parallel_config
                    )
                    logger.info("EAGLE detected %d extra layer(s)", num_draft_layers)
                except Exception:
                    logger.info(
                        "EAGLE detected, but failed to get the number of extra layers"
                        "falling back to 1"
                    )
                    num_draft_layers = 1
        return num_draft_layers

    def start_services(self) -> None:
        """
        Start all managed services.

        Managed services include:
        - InternalAPIServer: HTTP server exposing internal APIs for
          monitoring and management (e.g., cache stats, flush operations).
        - RuntimePluginLauncher: Launches external plugin processes defined
          in the configuration (e.g., custom telemetry, cache warming scripts).
        """
        if self._api_server is not None:
            self._api_server.start()

        if self._runtime_plugin_launcher is not None:
            self._runtime_plugin_launcher.launch_plugins()

    def _handle_post_init_failure(self, e: Exception) -> None:
        """
        Handle initialization failure during post_init.

        This method is shared between LMCacheManager and subclasses to avoid
        code duplication.

        Args:
            e: The exception that caused the failure
        """
        self._init_failed = True
        self._init_failed_reason = str(e)
        if self._lmcache_engine is not None:
            self._lmcache_engine.mark_init_failed(str(e))
        logger.error(
            "Failed during post_init: %s. "
            "System will operate in degraded mode (recompute).",
            e,
        )

    def post_init(self) -> None:
        """
        Post-initialization after KV caches are registered.
        """
        # If initialization already failed, mark engine and return early
        if self._init_failed:
            if self._lmcache_engine is not None:
                self._lmcache_engine.mark_init_failed(self._init_failed_reason)
            logger.warning("Skipping post_init due to previous initialization failure")
            return

        if self._lmcache_engine is None:
            # Initialize health monitor for scheduler (even without engine)
            self._init_health_monitor()
            return

        try:
            # vLLM mode post-init
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupServer,
            )

            async_lookup_server = None
            if self._config.enable_async_loading and self._lookup_server is not None:
                assert isinstance(self._lookup_server, LMCacheAsyncLookupServer)
                async_lookup_server = self._lookup_server

            self._lmcache_engine.post_init(async_lookup_server=async_lookup_server)

            # Initialize health monitor after engine post_init completes
            self._init_health_monitor()
        except Exception as e:
            self._handle_post_init_failure(e)

    def stop_services(self) -> None:
        """Stop all managed components gracefully."""
        logger.info("Stopping LMCacheManager services...")
        start_time = time.time()
        errors: list[tuple[str, Union[str, Exception]]] = []

        def _safe_close(name: str, close_fn, timeout: float = 10.0):
            """Helper to close a resource with timeout protection."""
            try:
                logger.info("Closing %s...", name)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(close_fn)
                    try:
                        future.result(timeout=timeout)
                        logger.info("%s closed successfully", name)
                    except TimeoutError:
                        logger.error(
                            "%s close operation timed out after %ss. "
                            "Continuing with shutdown...",
                            name,
                            timeout,
                        )
                        errors.append((name, "Timeout"))
            except Exception as e:
                logger.error("Error closing %s: %s", name, e)
                errors.append((name, e))

        # Stop health monitor first
        if self._health_monitor is not None:
            _safe_close("health_monitor", self._health_monitor.stop, timeout=5.0)

        # Close offload server
        if self._offload_server is not None:
            _safe_close("offload_server", self._offload_server.close, timeout=10.0)

        # Stop plugins
        if self._runtime_plugin_launcher is not None:
            _safe_close(
                "runtime_plugin_launcher",
                self._runtime_plugin_launcher.stop_plugins,
                timeout=10.0,
            )

        # Stop API server
        if self._api_server is not None:
            _safe_close("api_server", self._api_server.stop, timeout=10.0)

        # Close lookup server
        if self._lookup_server is not None:
            _safe_close("lookup_server", self._lookup_server.close, timeout=10.0)

        # Close lookup client
        if self._lookup_client is not None:
            _safe_close("lookup_client", self._lookup_client.close, timeout=10.0)

        # Destroy cache engine
        try:
            # In vLLM mode, use ENGINE_NAME constant
            logger.info("Destroying LMCache engine: %s", ENGINE_NAME)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(LMCacheEngineBuilder.destroy, ENGINE_NAME)
                try:
                    future.result(timeout=15.0)
                    logger.info("LMCache engine destroyed successfully")
                except TimeoutError:
                    logger.error(
                        "Cache engine destroy timed out after 15s. "
                        "Continuing with shutdown..."
                    )
                    errors.append(("cache_engine", "Timeout"))
        except Exception as e:
            logger.error("Error destroying cache engine: %s", e)
            errors.append(("cache_engine", e))

        elapsed = time.time() - start_time
        if errors:
            logger.warning(
                "Shutdown completed with %d errors in %.2fs: %s",
                len(errors),
                elapsed,
                errors,
            )
        else:
            logger.info(
                "LMCacheManager services stopped successfully in %.2fs", elapsed
            )

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance."""
        return self._lmcache_engine

    @property
    def lmcache_engine_metadata(self) -> Optional[LMCacheMetadata]:
        """Get the LMCache engine metadata."""
        return self._lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional[LookupClientInterface]:
        """Get the lookup client instance."""
        return self._lookup_client

    @property
    def lookup_server(
        self,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        """Get the lookup server instance."""
        return self._lookup_server

    def close_lookup_client(self) -> dict:
        """
        Close the current lookup client.

        Returns:
            dict: Result with old client type info.
        """
        old_type = self._get_lookup_type_str(self._lookup_client)
        if self._lookup_client is not None:
            try:
                self._lookup_client.close()
                self._lookup_client = None
                logger.info("Closed lookup client: %s", old_type)
            except Exception as e:
                logger.warning("Error closing lookup client: %s", e)
        return {"old": old_type}

    def create_lookup_client(self, dryrun: bool = False) -> dict:
        """
        Create a new lookup client using the current configuration.

        Args:
            dryrun: If True, only return what would be created without
                actually creating it.

        Returns:
            dict: Result with new client type info.
        """
        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        if self._lmcache_engine_metadata is None:
            return {"error": "metadata not available"}

        if dryrun:
            # Create temporarily to get type info, then discard
            client = LookupClientFactory.create_lookup_client(
                self._config,
                self._lmcache_engine_metadata,
                self._lmcache_engine,
            )
            new_type = self._get_lookup_type_str(client)
            client.close()
            return {"new": new_type, "dryrun": True}

        self._lookup_client = LookupClientFactory.create_lookup_client(
            self._config,
            self._lmcache_engine_metadata,
            self._lmcache_engine,
        )
        new_type = self._get_lookup_type_str(self._lookup_client)
        logger.info("Created lookup client: %s", new_type)
        return {"new": new_type}

    def recreate_lookup_client(self) -> dict:
        """
        Recreate the lookup client (close + create).

        Returns:
            dict: Result with old and new client type info.
        """
        if self._role != "scheduler":
            return {"error": "only supported for scheduler role"}

        result = self.close_lookup_client()
        create_result = self.create_lookup_client()
        result.update(create_result)
        return result

    def close_lookup_server(self) -> dict:
        """
        Close the current lookup server.

        Returns:
            dict: Result with old server type info.
        """
        old_type = self._get_lookup_type_str(self._lookup_server)
        if self._lookup_server is not None:
            try:
                self._lookup_server.close()
                self._lookup_server = None
                logger.info("Closed lookup server: %s", old_type)
            except Exception as e:
                logger.warning("Error closing lookup server: %s", e)
        return {"old": old_type}

    def create_lookup_server(self, dryrun: bool = False) -> dict:
        """
        Create a new lookup server using the current configuration.

        Args:
            dryrun: If True, only return what would be created without
                actually creating it.

        Returns:
            dict: Result with new server type info.
        """
        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        if self._lmcache_engine is None:
            return {"error": "engine not available"}
        if self._lmcache_engine_metadata is None:
            return {"error": "metadata not available"}

        if dryrun:
            # Create temporarily to get type info, then discard
            server = LookupClientFactory.create_lookup_server(
                self._lmcache_engine,
                self._lmcache_engine_metadata,
            )
            new_type = self._get_lookup_type_str(server)
            if server is not None:
                server.close()
            return {"new": new_type, "dryrun": True}

        self._lookup_server = LookupClientFactory.create_lookup_server(
            self._lmcache_engine,
            self._lmcache_engine_metadata,
        )
        new_type = self._get_lookup_type_str(self._lookup_server)
        logger.info("Created lookup server: %s", new_type)
        return {"new": new_type}

    def recreate_lookup_server(self) -> dict:
        """
        Recreate the lookup server (close + create).

        Returns:
            dict: Result with old and new server type info.
        """
        if self._role != "worker":
            return {"error": "only supported for worker role"}

        result = self.close_lookup_server()
        create_result = self.create_lookup_server()
        result.update(create_result)
        return result

    def _get_lookup_type_str(self, obj) -> str:
        """
        Get type string for lookup client/server, including wrapper info.

        Returns format: "OuterWrapper(InnerWrapper(CoreType))" or "None"
        """
        if obj is None:
            return "None"

        # First Party
        parts = []
        current = obj
        while True:
            parts.append(type(current).__name__)
            # Check for wrapper by looking for 'actual_lookup_client' attribute
            if hasattr(current, "actual_lookup_client"):
                current = current.actual_lookup_client
            else:
                break

        if len(parts) == 1:
            return parts[0]
        # Format: Outer(Inner(Core))
        result = parts[-1]
        for wrapper in reversed(parts[:-1]):
            result = "%s(%s)" % (wrapper, result)
        return result

    def get_lookup_info(self) -> dict:
        """
        Get information about the current lookup client and server.

        Returns:
            dict: Information about lookup client/server types and role.
        """
        return {
            "client": self._get_lookup_type_str(self._lookup_client),
            "server": self._get_lookup_type_str(self._lookup_server),
            "role": self._role,
        }

    @property
    def offload_server(self) -> Optional[ZMQOffloadServer]:
        """Get the offload server instance."""
        return self._offload_server

    @property
    def api_server(self) -> Optional[InternalAPIServer]:
        """Get the API server instance."""
        return self._api_server

    @property
    def health_monitor(self) -> Optional[HealthMonitor]:
        """Get the health monitor instance."""
        return self._health_monitor

    @property
    def role(self) -> str:
        """Get the role of this manager (scheduler or worker)."""
        return self._role

    @property
    def kv_caches(self) -> dict[str, torch.Tensor]:
        if self._connector is not None and hasattr(self._connector, "kv_caches"):
            return self._connector.kv_caches
        return {}

    def is_healthy(self) -> bool:
        """
        Check if the LMCacheManager is healthy.

        Returns False if:
        - Initialization failed
        - HealthMonitor reports unhealthy
        - Engine reports unhealthy

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._lmcache_engine is not None and not self._lmcache_engine.is_healthy():
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    @property
    def config(self) -> LMCacheEngineConfig:
        """Get the LMCache engine configuration."""
        return self._config

    def get_inference_info(self) -> dict:
        """Get inference information by delegating to the connector.

        Returns:
            dict: Dictionary containing inference information,
                  or empty dict if connector is not available.
        """
        if self._connector is not None and hasattr(
            self._connector, "get_inference_info"
        ):
            return self._connector.get_inference_info()
        return {}
