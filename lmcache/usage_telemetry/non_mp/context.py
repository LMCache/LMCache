# SPDX-License-Identifier: Apache-2.0
"""One-shot context reporting for the single-process LMCacheEngine path."""

# Future
from __future__ import annotations

# Standard
from datetime import datetime
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.base import UsageContextBase
from lmcache.usage_telemetry.env_probe import collect_env_message
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import is_usage_tracking_enabled
from lmcache.usage_telemetry.messages import (
    DeploymentMode,
    EngineMessage,
    MetadataMessage,
    UsageMessage,
)
from lmcache.usage_telemetry.transport import UsageMessageSender

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class UsageContext(UsageContextBase):
    """One-shot usage reporter for the single-process LMCacheEngine path.

    Sends an ``EnvMessage``, an ``EngineMessage``, and a
    ``MetadataMessage`` to the stats server.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        local_log: str | None = None,
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            config: The engine configuration to snapshot.
            metadata: The engine metadata (model, world size, kv layout).
            local_log: Path of a local log of sent payloads; ``None``
                disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        super().__init__(DeploymentMode.SINGLE_PROCESS, local_log, sender)
        self._config = config
        self._metadata = metadata

    def _collect_messages(self) -> list[UsageMessage]:
        metadata_message = MetadataMessage(
            start_time=self._start_time.strftime("%Y-%m-%d %H:%M:%S"),
            duration=(datetime.now() - self._start_time).total_seconds(),
        )
        return [
            collect_env_message(),
            EngineMessage.from_config(self._config, self._metadata),
            metadata_message,
        ]


@swallow_telemetry_errors
def InitializeUsageContext(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    local_log: str | None = None,
    sender: UsageMessageSender | None = None,
) -> UsageContext | None:
    """Start one-shot usage reporting for a single-process engine.

    Returns immediately; the report is sent in the background. Never
    blocks or raises.

    Args:
        config: The engine configuration to snapshot.
        metadata: The engine metadata (model, world size, kv layout).
        local_log: Path of a local log of sent payloads; ``None`` disables
            local logging.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The usage context, or ``None`` when usage tracking is disabled or
        initialization failed.
    """
    if not is_usage_tracking_enabled():
        return None
    logger.info("Initializing usage context.")
    context = UsageContext(config, metadata, local_log, sender)
    threading.Thread(
        target=context.report_once, daemon=True, name="lmcache-usage-report"
    ).start()
    return context
