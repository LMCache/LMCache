# SPDX-License-Identifier: Apache-2.0
"""Coordinator -> mp server command channel over HTTP.

The coordinator pushes commands (future quota broadcast, KV-op fan-out) by
POSTing to each mp server's existing HTTP server. Reach is derived from the
registry (an instance's ``ip`` + ``http_port``), so there is no per-instance
connection state to manage.
"""

# Standard
from typing import Any
import asyncio

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.registry import InstanceRegistry

logger = init_logger(__name__)

# Path on the mp server's HTTP API that receives pushed commands.
COMMAND_PATH = "/coordinator/command"


class UnknownInstanceError(KeyError):
    """A command was addressed to an instance not in the registry."""


class PushError(Exception):
    """A pushed command failed at the HTTP layer (e.g. timeout, 5xx)."""


class HttpPushClient:
    """Sends command payloads to registered mp servers over HTTP.

    Args:
        registry: The shared instance registry (provides reach: ip + http_port).
        client: An ``httpx.AsyncClient`` used for all requests.
        timeout_s: Per-request timeout in seconds.
    """

    def __init__(
        self,
        registry: InstanceRegistry,
        client: httpx.AsyncClient,
        timeout_s: float = 10.0,
    ) -> None:
        """Initialize with a registry and HTTP client."""
        self._registry = registry
        self._client = client
        self._timeout_s = timeout_s

    async def send_command(
        self, instance_id: str, command: dict[str, Any]
    ) -> dict[str, Any]:
        """POST a command to one mp server and return its JSON reply.

        Args:
            instance_id: Target mp server.
            command: JSON-serializable command body.

        Returns:
            The mp server's JSON response as a dict.

        Raises:
            UnknownInstanceError: If the instance is not registered.
            PushError: If the request fails or returns a non-2xx status.
        """
        instance = self._registry.get(instance_id)
        if instance is None:
            raise UnknownInstanceError(f"Instance {instance_id} is not registered")
        url = f"http://{instance.ip}:{instance.http_port}{COMMAND_PATH}"
        try:
            response = await self._client.post(
                url, json=command, timeout=self._timeout_s
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise PushError(f"Command to {instance_id} failed: {e}") from e

    async def broadcast(self, command: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """POST a command to every registered mp server, skipping failures.

        Args:
            command: JSON-serializable command body.

        Returns:
            A mapping of instance id to JSON reply for instances that replied.
        """
        instances = self._registry.all_instances()
        if not instances:
            return {}

        async def _send_one(instance_id: str) -> tuple[str, dict[str, Any]] | None:
            try:
                return instance_id, await self.send_command(instance_id, command)
            except (UnknownInstanceError, PushError) as e:
                logger.warning("Broadcast to instance %s failed: %s", instance_id, e)
                return None

        results = await asyncio.gather(*(_send_one(n.instance_id) for n in instances))
        return {item[0]: item[1] for item in results if item is not None}
