# SPDX-License-Identifier: Apache-2.0
"""One-shot usage reporting for the multiprocess (MP) cache server."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.env_probe import collect_env_message
from lmcache.usage_telemetry.identity import is_usage_tracking_enabled
from lmcache.usage_telemetry.one_shot import UsageContextBase
from lmcache.usage_telemetry.transport import UsageMessageSender, usage_server_url

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import StorageManagerConfig
    from lmcache.v1.multiprocess.config import MPServerConfig

logger = init_logger(__name__)


@dataclass
class MPServerMessage:
    """Configuration snapshot of a multiprocess (MP) cache server.

    Note:
        The server's operator-facing ``instance_id`` is deliberately not
        included: it can be operator-chosen and therefore identifying.
        Usage messages are correlated through
        :class:`lmcache.usage_telemetry.identity.UsageIdentity` only.
    """

    lmcache_version: str
    chunk_size: int
    hash_algorithm: str
    engine_type: str
    supported_transfer_mode: str
    separate_object_groups: bool
    max_gpu_workers: int
    max_cpu_workers: int
    p2p_enabled: bool
    l1_size_bytes: int
    l1_medium: str
    l1_shm_enabled: bool
    eviction_policy: str
    l2_adapter_types: str
    l2_store_policy: str
    l2_prefetch_policy: str

    @classmethod
    def from_configs(
        cls,
        mp_config: MPServerConfig,
        storage_manager_config: StorageManagerConfig,
    ) -> MPServerMessage:
        """Build the message from the MP server and storage configurations.

        Args:
            mp_config: The MP server configuration.
            storage_manager_config: The storage manager configuration.

        Returns:
            The populated message. ``l1_medium`` is one of ``"dram"``,
            ``"gds"``, or ``"dram+devdax"``; ``l2_adapter_types`` is a
            comma-joined list of adapter type names (empty when no L2 is
            configured).
        """
        # First Party
        from lmcache import __version__
        from lmcache.v1.distributed.l2_adapters.config import (
            get_type_name_for_config,
        )

        l1_config = storage_manager_config.l1_manager_config
        memory_config = l1_config.memory_config
        if l1_config.gds_l1_config is not None:
            l1_size_bytes = l1_config.gds_l1_config.size_in_bytes
            l1_medium = "gds"
        else:
            l1_size_bytes = (
                memory_config.size_in_bytes + memory_config.devdax_size_in_bytes
            )
            l1_medium = "dram+devdax" if memory_config.devdax_path else "dram"
        l2_adapter_types = ",".join(
            get_type_name_for_config(adapter_config)
            for adapter_config in storage_manager_config.l2_adapter_config.adapters
        )
        return cls(
            lmcache_version=__version__,
            chunk_size=mp_config.chunk_size,
            hash_algorithm=mp_config.hash_algorithm,
            engine_type=mp_config.engine_type,
            supported_transfer_mode=mp_config.supported_transfer_mode,
            separate_object_groups=mp_config.separate_object_groups,
            max_gpu_workers=mp_config.max_gpu_workers,
            max_cpu_workers=mp_config.max_cpu_workers,
            p2p_enabled=mp_config.p2p_config.enabled,
            l1_size_bytes=l1_size_bytes,
            l1_medium=l1_medium,
            l1_shm_enabled=bool(memory_config.shm_name),
            eviction_policy=storage_manager_config.eviction_config.eviction_policy,
            l2_adapter_types=l2_adapter_types,
            l2_store_policy=storage_manager_config.store_policy,
            l2_prefetch_policy=storage_manager_config.prefetch_policy,
        )


class MPUsageContext(UsageContextBase):
    """One-shot usage reporter for the multiprocess cache server.

    Sends an :class:`lmcache.usage_telemetry.env_probe.EnvMessage` and an
    :class:`MPServerMessage` to the stats server. Model information is not
    known at server startup (vLLM instances register later), so it is
    intentionally absent here; it rides the continuous messages instead.
    """

    def __init__(
        self,
        server_url: str,
        mp_config: MPServerConfig,
        storage_manager_config: StorageManagerConfig,
        local_log: str | None = None,
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            server_url: Full URL of the one-shot context endpoint.
            mp_config: The MP server configuration to snapshot.
            storage_manager_config: The storage manager configuration to
                snapshot.
            local_log: Path of a local log of sent payloads; ``None``
                disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        super().__init__(server_url, local_log, sender)
        self._mp_config = mp_config
        self._storage_manager_config = storage_manager_config

    def _collect_messages(self) -> list[tuple[object, str]]:
        mp_server_message = MPServerMessage.from_configs(
            self._mp_config, self._storage_manager_config
        )
        return [
            (collect_env_message(), "EnvMessage"),
            (mp_server_message, "MPServerMessage"),
        ]


def InitializeMPUsageContext(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    local_log: str | None = None,
    sender: UsageMessageSender | None = None,
) -> MPUsageContext | None:
    """Start one-shot usage reporting for a multiprocess cache server.

    The report is sent from a daemon thread so server startup never blocks
    on the stats server.

    Args:
        mp_config: The MP server configuration to snapshot.
        storage_manager_config: The storage manager configuration to
            snapshot.
        local_log: Path of a local log of sent payloads; ``None`` disables
            local logging.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The usage context, or ``None`` when usage tracking is disabled.
    """
    if not is_usage_tracking_enabled():
        return None
    logger.info("Initializing MP usage context.")
    context = MPUsageContext(
        usage_server_url("context"),
        mp_config,
        storage_manager_config,
        local_log,
        sender,
    )
    threading.Thread(
        target=context.report_once, daemon=True, name="lmcache-usage-report"
    ).start()
    return context
