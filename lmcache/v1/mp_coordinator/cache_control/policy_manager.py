# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side proxy for node-local runtime policy APIs."""

# Standard
from typing import Any

# Third Party
import httpx

# First Party
from lmcache.v1.mp_coordinator.registry import MPInstance


class RuntimePolicyManager:
    """Send runtime-policy requests to registered MP servers."""

    async def request(
        self,
        target: MPInstance,
        http_client: httpx.AsyncClient,
        method: str,
        body: dict[str, Any] | None = None,
        endpoint: str = "/config/policies",
    ) -> tuple[int, Any]:
        """Proxy one policy request and decode its JSON response.

        Args:
            target: The registered MP server to call.
            http_client: Coordinator-owned outbound HTTP client.
            method: HTTP method for the node-local policy endpoint.
            body: Optional JSON request body.
            endpoint: Node-local policy endpoint path.

        Returns:
            The node's status code and decoded JSON body.

        Raises:
            httpx.HTTPError: If the coordinator cannot reach the target.
        """
        url = f"http://{target.ip}:{target.http_port}{endpoint}"
        response = await http_client.request(method, url, json=body)
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text}
        return response.status_code, payload
