# SPDX-License-Identifier: Apache-2.0
"""Payload construction and HTTP transport for usage telemetry."""

# Future
from __future__ import annotations

# Standard
from urllib.parse import urljoin
import json
import os

# Third Party
import torch

# First Party
from lmcache.connections import global_http_connection
from lmcache.logging import init_logger
from lmcache.usage_telemetry.identity import UsageIdentity

logger = init_logger(__name__)


USAGE_SCHEMA_VERSION = 1
"""Version stamped on every usage payload; bump when fields change meaning."""


def usage_server_url(endpoint: str) -> str:
    """Build the stats-server URL for *endpoint*.

    Args:
        endpoint: Path component on the stats server, e.g. ``"context"``.

    Returns:
        The full URL, honoring the ``LMCACHE_USAGE_TRACK_URL`` override.
    """
    base = os.getenv("LMCACHE_USAGE_TRACK_URL", "http://stats.lmcache.ai:8080")
    return urljoin(base, endpoint)


class UsageMessageSender:
    """Default HTTP transport for usage messages.

    Posts JSON payloads with a short timeout and swallows every failure
    (telemetry must never disturb serving). Inject a stub instance into the
    usage contexts to capture payloads in tests.
    """

    def send(self, url: str, payload: dict[str, object]) -> None:
        """POST one JSON payload to *url*; never raises.

        Args:
            url: Full endpoint URL on the stats server.
            payload: Flat JSON-serializable payload.
        """
        try:
            client = global_http_connection.get_sync_client()
            client.post(
                url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except Exception as e:
            logger.debug("Unable to send lmcache usage message: %s", e)


DEFAULT_SENDER = UsageMessageSender()
"""Shared default sender; stateless, safe to reuse across contexts."""


def build_usage_payload(
    message: object, message_type: str, identity: UsageIdentity
) -> dict[str, object]:
    """Flatten *message* into a payload stamped with identity and schema info.

    Args:
        message: A message dataclass instance; its fields become payload keys.
        message_type: Discriminator stored under the ``message_type`` key.
        identity: Identifiers stamped on the payload.

    Returns:
        A flat JSON-serializable dict (``torch.dtype`` values stringified).
    """
    payload: dict[str, object] = {
        "message_type": message_type,
        "schema_version": USAGE_SCHEMA_VERSION,
        "session_id": identity.session_id,
        "machine_id": identity.machine_id,
    }
    for key, value in vars(message).items():
        payload[key] = str(value) if isinstance(value, torch.dtype) else value
    return payload
