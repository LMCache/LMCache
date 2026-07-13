# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for one-shot (context) usage reporting.

Mode-specific reporters live in :mod:`lmcache.usage_telemetry.mp` and
:mod:`lmcache.usage_telemetry.non_mp`; they subclass
:class:`UsageContextBase` and only define which messages to send.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from datetime import datetime

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import get_usage_identity
from lmcache.usage_telemetry.messages import DeploymentMode, UsageMessage
from lmcache.usage_telemetry.transport import (
    DEFAULT_SENDER,
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)

logger = init_logger(__name__)


class UsageContextBase(ABC):
    """Shared plumbing for one-shot usage reporting.

    Subclasses define which messages the report contains via
    :meth:`_collect_messages`; this base owns the identity, transport, and
    optional local logging. Each message is POSTed to the endpoint it
    declares in the schema (:mod:`lmcache.usage_telemetry.messages`),
    routed through the mode's URL prefix.
    """

    def __init__(
        self,
        mode: DeploymentMode,
        local_log: str | None,
        sender: UsageMessageSender | None,
    ) -> None:
        """Initialize shared reporting state.

        Args:
            mode: Deployment mode stamped on every payload this reporter
                sends and selecting the endpoint prefix.
            local_log: Path of a human-readable local log of every sent
                payload; ``None`` disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        self._mode = mode
        self._local_log = local_log
        self._sender = sender if sender is not None else DEFAULT_SENDER
        self._identity = get_usage_identity()
        self._start_time = datetime.now()

    @abstractmethod
    def _collect_messages(self) -> list[UsageMessage]:
        """Return the messages of the one-shot report."""
        raise NotImplementedError

    @swallow_telemetry_errors
    def report_once(self) -> None:
        """Collect and send every one-shot message on the calling thread.

        Failures are swallowed; never raises.
        """
        for message in self._collect_messages():
            payload = build_usage_payload(message, self._identity, self._mode)
            self._sender.send(usage_server_url(message.ENDPOINT, self._mode), payload)
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
