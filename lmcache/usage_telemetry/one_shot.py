# SPDX-License-Identifier: Apache-2.0
"""One-shot usage reporting: a startup snapshot of environment and config."""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
import threading

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.usage_telemetry.env_probe import collect_env_message
from lmcache.usage_telemetry.identity import (
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.transport import (
    DEFAULT_SENDER,
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


@dataclass
class EngineMessage:
    """Configuration snapshot of a single-process LMCacheEngine."""

    chunksize: int
    local_device: str
    max_local_cache_size: int
    remote_url: str | None
    remote_serde: str | None
    pipelined_backend: bool
    save_decode_cache: bool
    enable_blending: bool
    blend_recompute_ratio: float
    blend_min_tokens: int
    model_name: str
    world_size: int
    worker_id: int
    kv_dtype: torch.dtype
    kv_shape: tuple[int, int, int, int, int]

    @classmethod
    def from_config(
        cls, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> EngineMessage:
        """Build the message from the engine configuration and metadata.

        Args:
            config: The engine configuration to snapshot.
            metadata: The engine metadata (model, world size, kv layout).

        Returns:
            The populated message.
        """
        return cls(
            chunksize=config.chunk_size,
            local_device="cpu" if config.local_cpu else torch_device_type,
            max_local_cache_size=int(config.max_local_cpu_size),
            remote_url=config.remote_url,
            remote_serde=config.remote_serde,
            pipelined_backend=False,
            save_decode_cache=config.save_decode_cache,
            enable_blending=config.enable_blending,
            blend_recompute_ratio=0.15,
            blend_min_tokens=config.blend_min_tokens,
            model_name=metadata.model_name,
            world_size=metadata.world_size,
            worker_id=metadata.worker_id,
            kv_dtype=metadata.kv_dtype,
            kv_shape=metadata.kv_shape,
        )


@dataclass
class MetadataMessage:
    """Process start time and uptime at send time."""

    start_time: str
    duration: float


class UsageContextBase(ABC):
    """Shared plumbing for one-shot usage reporting.

    Subclasses define which messages the report contains via
    :meth:`_collect_messages`; this base owns the identity, transport, and
    optional local logging.
    """

    def __init__(
        self,
        server_url: str,
        local_log: str | None,
        sender: UsageMessageSender | None,
    ) -> None:
        """Initialize shared reporting state.

        Args:
            server_url: Full URL of the one-shot context endpoint.
            local_log: Path of a human-readable local log of every sent
                payload; ``None`` disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        self._server_url = server_url
        self._local_log = local_log
        self._sender = sender if sender is not None else DEFAULT_SENDER
        self._identity = get_usage_identity()
        self._start_time = datetime.now()

    @abstractmethod
    def _collect_messages(self) -> list[tuple[object, str]]:
        """Return the ``(message, message_type)`` pairs of the report."""
        raise NotImplementedError

    def report_once(self) -> None:
        """Collect and send every one-shot message synchronously.

        The factory functions run this on a daemon thread so that slow or
        unreachable stats servers never delay startup; tests may call it
        directly. Send failures are swallowed.
        """
        for message, message_type in self._collect_messages():
            payload = build_usage_payload(message, message_type, self._identity)
            self._sender.send(self._server_url, payload)
            self._write_local(payload)

    def _write_local(self, payload: dict[str, object]) -> None:
        """Append *payload* to the local log file, if one is configured."""
        if self._local_log is None:
            return
        text = "".join(f"{key}: {value}\n" for key, value in payload.items()) + "\n"
        try:
            with open(self._local_log, "a") as f:
                f.write(text)
        except OSError:
            logger.debug("Unable to write usage log to %s", self._local_log)


class UsageContext(UsageContextBase):
    """One-shot usage reporter for the single-process LMCacheEngine path.

    Sends an :class:`EnvMessage`, an :class:`EngineMessage`, and a
    :class:`MetadataMessage` to the stats server.
    """

    def __init__(
        self,
        server_url: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        local_log: str | None = None,
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            server_url: Full URL of the one-shot context endpoint.
            config: The engine configuration to snapshot.
            metadata: The engine metadata (model, world size, kv layout).
            local_log: Path of a local log of sent payloads; ``None``
                disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        super().__init__(server_url, local_log, sender)
        self._config = config
        self._metadata = metadata

    def _collect_messages(self) -> list[tuple[object, str]]:
        metadata_message = MetadataMessage(
            start_time=self._start_time.strftime("%Y-%m-%d %H:%M:%S"),
            duration=(datetime.now() - self._start_time).total_seconds(),
        )
        return [
            (collect_env_message(), "EnvMessage"),
            (EngineMessage.from_config(self._config, self._metadata), "EngineMessage"),
            (metadata_message, "MetadataMessage"),
        ]


def InitializeUsageContext(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    local_log: str | None = None,
    sender: UsageMessageSender | None = None,
) -> UsageContext | None:
    """Start one-shot usage reporting for a single-process engine.

    The report is sent from a daemon thread so startup never blocks on the
    stats server.

    Args:
        config: The engine configuration to snapshot.
        metadata: The engine metadata (model, world size, kv layout).
        local_log: Path of a local log of sent payloads; ``None`` disables
            local logging.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The usage context, or ``None`` when usage tracking is disabled.
    """
    if not is_usage_tracking_enabled():
        return None
    logger.info("Initializing usage context.")
    context = UsageContext(
        usage_server_url("context"), config, metadata, local_log, sender
    )
    threading.Thread(
        target=context.report_once, daemon=True, name="lmcache-usage-report"
    ).start()
    return context
