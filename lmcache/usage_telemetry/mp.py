# SPDX-License-Identifier: Apache-2.0
"""Usage reporting for the multiprocess (MP) cache server.

Two reporting mechanisms:

- One-shot at server startup (:class:`MPUsageContext` via
  :func:`InitializeMPUsageContext`): environment and server configuration.
- Registration hook (:func:`report_kv_cache_registered`): model and KV
  layout information, which only become visible when a serving engine
  registers its KV caches.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.context import UsageContextBase
from lmcache.usage_telemetry.env_probe import collect_env_message
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import is_usage_tracking_enabled
from lmcache.usage_telemetry.messages import (
    DeploymentMode,
    MPInstanceMessage,
    MPServerMessage,
    UsageMessage,
)
from lmcache.usage_telemetry.transport import (
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import StorageManagerConfig
    from lmcache.v1.multiprocess.config import MPServerConfig

logger = init_logger(__name__)


class MPUsageContext(UsageContextBase):
    """Usage reporter for the multiprocess cache server.

    At startup it sends an ``EnvMessage`` and an ``MPServerMessage``;
    :meth:`report_instance` sends per-instance model/KV information.
    """

    def __init__(
        self,
        mp_config: MPServerConfig,
        storage_manager_config: StorageManagerConfig,
        local_log: str | None = None,
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            mp_config: The MP server configuration to snapshot.
            storage_manager_config: The storage manager configuration to
                snapshot.
            local_log: Path of a local log of sent payloads; ``None``
                disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        super().__init__(DeploymentMode.MP_SERVER, local_log, sender)
        self._mp_config = mp_config
        self._storage_manager_config = storage_manager_config
        self._reported_instances: set[tuple[str, int]] = set()
        self._reported_instances_lock = threading.Lock()

    def _collect_messages(self) -> list[UsageMessage]:
        return [
            collect_env_message(),
            MPServerMessage.from_configs(self._mp_config, self._storage_manager_config),
        ]

    @swallow_telemetry_errors
    def report_instance(self, message: MPInstanceMessage) -> None:
        """Send *message* once per ``(model_name, world_size)`` pair.

        Duplicate registrations (multiple workers of one engine, worker
        restarts and re-registrations) are dropped. Safe to call from any
        thread; never raises.

        Args:
            message: The instance snapshot to report.
        """
        key = (message.model_name, message.world_size)
        with self._reported_instances_lock:
            if key in self._reported_instances:
                return
            self._reported_instances.add(key)
        payload = build_usage_payload(message, self._identity, self._mode)
        self._sender.send(usage_server_url(message.ENDPOINT), payload)
        self._write_local(payload)


_mp_usage_context: MPUsageContext | None = None
"""Context created by :func:`InitializeMPUsageContext`; consumed by
:func:`report_kv_cache_registered`. ``None`` when tracking is disabled."""


@swallow_telemetry_errors
def InitializeMPUsageContext(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    local_log: str | None = None,
    sender: UsageMessageSender | None = None,
) -> MPUsageContext | None:
    """Start usage reporting for a multiprocess cache server.

    Returns immediately; the startup report is sent in the background.
    Never blocks or raises.

    Args:
        mp_config: The MP server configuration to snapshot.
        storage_manager_config: The storage manager configuration to
            snapshot.
        local_log: Path of a local log of sent payloads; ``None`` disables
            local logging.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The usage context, or ``None`` when usage tracking is disabled or
        initialization failed.
    """
    global _mp_usage_context
    if not is_usage_tracking_enabled():
        _mp_usage_context = None
        return None
    logger.info("Initializing MP usage context.")
    context = MPUsageContext(mp_config, storage_manager_config, local_log, sender)
    threading.Thread(
        target=context.report_once, daemon=True, name="lmcache-usage-report"
    ).start()
    _mp_usage_context = context
    return context


@swallow_telemetry_errors
def report_kv_cache_registered(
    model_name: str,
    world_size: int,
    kv_dtypes: list[str],
    kv_shapes: list[list[int]],
    attn_windows: list[int],
) -> None:
    """Report one registered KV-cache instance.

    Sends an :class:`MPInstanceMessage` in the background, once per
    ``(model_name, world_size)`` pair per process; duplicates are dropped.
    A no-op when usage tracking is disabled or MP usage reporting was
    never initialized. Never blocks or raises.

    Args:
        model_name: Model whose KV caches were registered.
        world_size: World size of the registering engine.
        kv_dtypes: Torch dtype name per KV object group.
        kv_shapes: Tensor dimensions per KV object group.
        attn_windows: Attention window per object group, in chunks;
            ``-1`` means full attention.
    """
    context = _mp_usage_context
    if context is None:
        return
    message = MPInstanceMessage(
        model_name=model_name,
        world_size=world_size,
        kv_dtypes=",".join(kv_dtypes),
        kv_shapes=";".join("x".join(str(dim) for dim in shape) for shape in kv_shapes),
        attn_windows=",".join(str(window) for window in attn_windows),
    )
    threading.Thread(
        target=context.report_instance,
        args=(message,),
        daemon=True,
        name="lmcache-usage-report",
    ).start()
