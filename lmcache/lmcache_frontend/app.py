# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import resources
from urllib.parse import unquote
import argparse
import asyncio
import json
import os
import threading
import time

# Third Party
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, PlainTextResponse
import httpx
import uvicorn

_PACKAGE = "lmcache.lmcache_frontend"


def _package_resource_path(relative: str) -> str:
    """Return absolute filesystem path for a file shipped inside the package.

    Replacement for the deprecated ``pkg_resources.resource_filename``;
    works for regular (non-zipped) installs, which is how LMCache ships.
    """
    return str(resources.files(_PACKAGE).joinpath(relative))


try:
    # Local
    from .heartbeat import HeartbeatService  # import as module
except ImportError:
    # Third Party
    from heartbeat import HeartbeatService  # type: ignore  # import as script


# Create router
router = APIRouter()


class _NodeRegistry:
    """Encapsulates the mutable proxy/node list used by the frontend.

    Replacing the list is done in-place via :py:meth:`replace` so that
    aliases (the module-level ``target_nodes`` reference and the list
    handed to ``HeartbeatService``) stay in sync.
    """

    def __init__(self) -> None:
        self._nodes: list[dict] = []

    @property
    def nodes(self) -> list[dict]:
        """Return the underlying list (mutated in place)."""
        return self._nodes

    def replace(self, new_nodes: list[dict]) -> None:
        """Swap the registry content in place with ``new_nodes``."""
        self._nodes[:] = new_nodes

    def is_allowed(self, host: str, port: str) -> bool:
        """Return True if ``host:port`` matches any registered node.

        Used by the SSRF guard in :func:`proxy_request` so the proxy
        only forwards to pre-registered destinations.
        """
        return self.resolve(host, port) is not None

    def resolve(self, host: str, port: str) -> tuple[str, str] | None:
        """Return the registry-owned ``(host, port)`` matching the input.

        The returned tuple is taken from the registry itself, not from
        the caller-supplied arguments.  Using this value to build the
        outbound URL breaks the SSRF taint flow for static analysers.
        """
        port = str(port)
        for node in self._nodes:
            n_host, n_port = str(node.get("host")), str(node.get("port"))
            if n_host == host and n_port == port:
                return n_host, n_port
        return None


_node_registry = _NodeRegistry()
# ``target_nodes`` is a module-level alias to the list owned by the
# registry.  External readers and in-place mutations (append / element
# update) keep working unchanged; whole-list replacement MUST go
# through ``_node_registry.replace`` so all aliases stay in sync.
target_nodes = _node_registry.nodes

# Initialize heartbeat service with app context
heartbeat_service: HeartbeatService = HeartbeatService()

global args
args = None


async def fetch_nodes_from_coordinator(url: str) -> list[dict]:
    """Fetch fleet membership from the MP coordinator's instance registry.

    Queries the coordinator's ``GET /instances`` (the single source of
    truth for fleet membership) and returns the registered mp servers as
    a flat list of node dicts.

    Args:
        url: Base URL of the MP coordinator.

    Returns:
        ``[{"name", "host", "port"}, ...]`` for the registered mp
        servers, or ``[]`` when the coordinator is unreachable.
    """
    base_url = url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/instances")
            response.raise_for_status()
            instances = response.json().get("instances") or []
        return [
            {
                "name": f"mp_{instance['instance_id']}",
                "host": instance["ip"],
                "port": str(instance["http_port"]),
            }
            for instance in instances
        ]
    except Exception as e:
        print(f"Failed to fetch nodes from coordinator: {e}")
        return []


def load_config(config_path: str | None = None) -> None:
    """Load the flat node list from a JSON config file.

    Args:
        config_path: Optional path to the JSON file.  When ``None`` the
            packaged ``config.json`` is used.
    """
    try:
        # Prioritize user-specified configuration file
        if config_path:
            with open(config_path, "r") as f:
                _node_registry.replace(json.load(f))
            print(
                f"Loaded {len(target_nodes)} target nodes from specified path: "
                f"{config_path}"
            )
        else:
            # Use package resource path as default configuration
            default_config_path = _package_resource_path("config.json")
            with open(default_config_path, "r") as f:
                _node_registry.replace(json.load(f))
            print(f"Loaded default configuration with {len(target_nodes)} target nodes")
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        _node_registry.replace([])


def validate_node(node: dict) -> bool:
    """Validate a single node configuration dict.

    Args:
        node: Candidate node dict.

    Returns:
        True when ``node`` has the required ``name``/``host``/``port`` keys.
    """
    if not isinstance(node, dict):
        return False

    required_keys = {"name", "host", "port"}
    return required_keys.issubset(node.keys())


def validate_nodes(nodes: list) -> bool:
    """Validate a list of node dicts; see :func:`validate_node`."""
    if not isinstance(nodes, list):
        return False

    return all(validate_node(node) for node in nodes)


@router.get("/api/nodes")
async def get_all_nodes() -> dict:
    """Return the flat list of mp-server nodes sourced from the coordinator.

    Returns:
        ``{"nodes": [{"name", "host", "port"}, ...]}``.
    """
    return {"nodes": list(target_nodes)}


@router.api_route(
    "/proxy2/{node_name}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_request_by_name(request: Request, node_name: str, path: str):
    """Proxy requests using node name as identifier.

    Resolves ``node_name`` against the flat node registry and forwards
    with a single /proxy2/{name}/{path} call.
    """
    node = next((n for n in target_nodes if n["name"] == node_name), None)
    if not node:
        raise HTTPException(
            status_code=404, detail=f"Node with name '{node_name}' not found"
        )

    return await proxy_request(
        request, target_host=node["host"], target_port_or_socket=node["port"], path=path
    )


@router.api_route(
    "/proxy/{target_host}/{target_port_or_socket}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_request(
    request: Request, target_host: str, target_port_or_socket: str | int, path: str
):
    """Proxy requests to the specified target host and port or socket path.

    For security, non-socket targets must match a host/port already
    registered in :data:`_node_registry`; this prevents the endpoint
    from being used as an open relay (SSRF).  Socket paths are
    accepted as-is because they are local UDS endpoints.
    """
    target_port_or_socket = unquote(str(target_port_or_socket))
    # Check if target_port_or_socket is a socket path (contains '/')
    is_socket_path = "/" in target_port_or_socket

    if is_socket_path:
        # For socket paths, use UDS transport
        socket_path = target_port_or_socket
        target_url = f"http://localhost/{path}"

        # Create UDS transport
        transport = httpx.AsyncHTTPTransport(uds=socket_path)
    else:
        port = target_port_or_socket
        # SSRF guard: resolve against the registry and reuse the
        # trusted host/port from there when building the outbound URL.
        # This keeps user-controlled values out of the URL sink.
        resolved = _node_registry.resolve(target_host, port)
        if resolved is None:
            raise HTTPException(
                status_code=403,
                detail=("Target %s:%s is not a registered node" % (target_host, port)),
            )
        safe_host, safe_port = resolved
        target_url = f"http://{safe_host}:{safe_port}/{path}"
        transport = None  # Use default transport

    headers = {}
    for key, value in request.headers.items():
        if key.lower() in [
            "host",
            "content-length",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        ]:
            continue
        headers[key] = value

    body = await request.body()

    # Create client with appropriate transport
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=request.query_params,
                timeout=60.0,
            )

            response_headers = {}
            for key, value in response.headers.items():
                if key.lower() in [
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "transfer-encoding",
                    "upgrade",
                ]:
                    continue
                response_headers[key] = value

            return PlainTextResponse(
                content=response.content,
                headers=response_headers,
                media_type=response.headers.get("content-type", "text/plain"),
                status_code=response.status_code,
            )

        except httpx.ConnectError as e:
            if is_socket_path:
                detail = f"Failed to connect to socket: {socket_path}"
            else:
                detail = f"Failed to connect to target service {target_host}:{port}"
            raise HTTPException(status_code=502, detail=detail) from e
        except httpx.TimeoutException as e:
            if is_socket_path:
                detail = f"Connection to socket {socket_path} timed out"
            else:
                detail = f"Connection to target service {target_host}:{port} timed out"
            raise HTTPException(status_code=504, detail=detail) from e
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error communicating with target service: {str(e)}",
            ) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}") from e


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "lmcache-monitor"}


@router.get("/api/heartbeat/status")
async def get_heartbeat_status():
    """Get heartbeat status"""
    return heartbeat_service.status()


@router.post("/api/heartbeat/start")
async def start_heartbeat_api(request: Request):
    """Start heartbeat service"""
    try:
        data = await request.json()
        heartbeat_url = data.get("heartbeat_url")
        initial_delay = data.get("initial_delay", 0)
        interval = data.get("interval", 30)

        if not heartbeat_url:
            raise HTTPException(status_code=400, detail="heartbeat_url is required")

        heartbeat_service.start(heartbeat_url, initial_delay, interval)
        return {"status": "success", "message": "Heartbeat service started"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/heartbeat/stop")
async def stop_heartbeat_api():
    """Stop heartbeat service"""
    try:
        heartbeat_service.stop()
        return {"status": "success", "message": "Heartbeat service stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def initialize_nodes(coordinator_url: str | None = None) -> None:
    """Initialize node configuration from CLI args or coordinator URL.

    Args:
        coordinator_url: Optional coordinator base URL.  When set,
            fleet membership is sourced from the coordinator's registry
            via ``fetch_nodes_from_coordinator``. Otherwise, falls
            back to ``args.nodes`` or ``args.config``.
    """
    global args

    if args is None:
        raise ValueError("args is not initialized")

    if coordinator_url:
        print(f"Fetching nodes from coordinator: {coordinator_url}")
        nodes = await fetch_nodes_from_coordinator(coordinator_url)
        if nodes:
            _node_registry.replace(nodes)
            print(f"Loaded {len(nodes)} mp servers from coordinator")
        else:
            print(f"Warning: coordinator {coordinator_url} returned no instances")
    elif args.nodes:
        try:
            nodes = json.loads(args.nodes)
            if validate_nodes(nodes):
                _node_registry.replace(nodes)
                print(f"Loaded {len(nodes)} target nodes from command line arguments")
            else:
                print(
                    "Failed to validate nodes parameter: missing required keys "
                    "('name', 'host', 'port')"
                )
        except json.JSONDecodeError:
            print("Failed to parse nodes JSON parameter")
    elif args.config:
        load_config(args.config)


# Minimum seconds between two coordinator refreshes triggered by ``/``.
# Each refresh issues one ``GET /instances`` to the coordinator, so an
# unthrottled ``/`` would hammer it. 30s matches the default heartbeat
# interval.
_COORDINATOR_REFRESH_INTERVAL_SEC = 30.0
_coordinator_last_refresh: float = 0.0
_coordinator_refresh_lock = asyncio.Lock()


async def _maybe_refresh_from_coordinator(coordinator_url: str) -> None:
    """Refresh the node registry from coordinator, at most once per interval.

    The first caller within the interval performs the refresh; other
    concurrent callers return immediately (stale-on-read). This keeps
    the homepage responsive even under high traffic.
    """
    global _coordinator_last_refresh
    now = time.monotonic()
    if now - _coordinator_last_refresh < _COORDINATOR_REFRESH_INTERVAL_SEC:
        return
    if _coordinator_refresh_lock.locked():
        return
    async with _coordinator_refresh_lock:
        now = time.monotonic()
        if now - _coordinator_last_refresh < _COORDINATOR_REFRESH_INTERVAL_SEC:
            return
        await initialize_nodes(coordinator_url)
        _coordinator_last_refresh = time.monotonic()


@router.get("/")
async def serve_frontend():
    """Return frontend homepage.

    When a coordinator URL is configured, trigger a throttled
    background-style refresh so opening the homepage repeatedly does
    not hammer the coordinator or every proxy's ``/api/nodes``.
    """
    if args.coordinator_url:
        await _maybe_refresh_from_coordinator(args.coordinator_url)

    try:
        # Use package resource path
        index_path = _package_resource_path("static/index.html")
        return FileResponse(index_path)
    except Exception:
        # Development environment uses local files
        return FileResponse("static/index.html")


# Helper function to fetch metrics from a single node
async def _fetch_node_metrics(node):
    """Fetch metrics from a single node"""
    try:
        # Check if port is a socket path
        is_socket_path = "/" in node["port"]

        if is_socket_path:
            # Use UDS transport for socket paths
            transport = httpx.AsyncHTTPTransport(uds=node["port"])
            # Use localhost as host
            url = "http://localhost/metrics"
            async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        else:
            # Build URL for regular port
            url = f"http://{node['host']}:{node['port']}/metrics"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
    except Exception as e:
        return f"# ERROR: Failed to get metrics from {node['name']}: {str(e)}\n"


@router.get("/metrics")
async def aggregated_metrics():
    """Aggregate /metrics from every registered mp-server node."""
    if not target_nodes:
        return PlainTextResponse("# No nodes configured\n", status_code=404)

    # Snapshot the registry to avoid mid-iteration mutation
    nodes = list(target_nodes)
    metrics_results = await asyncio.gather(
        *[_fetch_node_metrics(node) for node in nodes]
    )

    # Combine all metrics with node name as comment header
    aggregated = ""
    for i, metrics in enumerate(metrics_results):
        node = nodes[i]
        aggregated += (
            f"# Metrics from node: {node['name']} ({node['host']}:{node['port']})\n"
        )
        aggregated += metrics
        aggregated += "\n\n"

    return PlainTextResponse(aggregated)


def create_app():
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Flexible Proxy Server",
        description="HTTP proxy service supporting specified target hosts and ports",
    )
    app.include_router(router)

    # Get static file path (prefer package resources)
    try:
        static_path = _package_resource_path("static")
    except Exception:
        static_path = os.path.join(os.path.dirname(__file__), "static")

    # Mount static file service
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    return app


def main():
    global args
    parser = argparse.ArgumentParser(description="LMCache Cluster Monitoring Tool")
    parser.add_argument(
        "--port", type=int, default=8000, help="Service port, default 8000"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Bind host address, default 0.0.0.0"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Specify configuration file path, default uses internal config.json",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Directly specify target nodes as a JSON string. "
        'Example: \'[{"name":"node1","host":"127.0.0.1","port":8001}]\'',
    )
    parser.add_argument(
        "--heartbeat-url",
        type=str,
        default=None,
        help="Heartbeat service URL, e.g.: http://example.com/heartbeat",
    )
    parser.add_argument(
        "--report-host",
        type=str,
        default=None,
        help="Host to report in heartbeat api_address. When set, bypasses "
        "get_local_ip() auto-detection. Useful for local dev or multi-NIC "
        "hosts where the auto-detected IP is not reachable from the "
        "discovery service side.",
    )
    parser.add_argument(
        "--heartbeat-initial-delay",
        type=int,
        default=0,
        help="Initial delay before starting heartbeat (seconds), default 0",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Heartbeat interval (seconds), default 30",
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default=None,
        help="MP coordinator base URL to source fleet membership from, default None",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warning",
        choices=["critical", "error", "warning", "warn", "info", "debug", "trace"],
        help="Uvicorn log level, default: warn",
    )
    parser.add_argument(
        "--no-http",
        action="store_true",
        default=False,
        help="Disable HTTP server startup (heartbeat still runs)",
    )

    args = parser.parse_args()

    # Initialize node configuration
    asyncio.run(initialize_nodes(args.coordinator_url))

    app = create_app()
    print(f"Monitoring service running at http://{args.host}:{args.port}")
    print(f"Node management: http://{args.host}:{args.port}/static/index.html")

    # Start heartbeat service if URL is configured
    if args.heartbeat_url:
        # Set application configuration for heartbeat service
        heartbeat_service.set_app_config(
            args.host, args.port, target_nodes, args.report_host
        )

        print("Starting heartbeat service...")
        print(f"Heartbeat URL: {args.heartbeat_url}")
        print(f"Initial delay: {args.heartbeat_initial_delay}s")
        print(f"Interval: {args.heartbeat_interval}s")
        reported_host = args.report_host or heartbeat_service.get_local_ip()
        print(f"API Address: http://{reported_host}:{args.port}")
        print(f"Target nodes count: {len(target_nodes)}")

        heartbeat_service.start(
            args.heartbeat_url, args.heartbeat_initial_delay, args.heartbeat_interval
        )
    else:
        print("Heartbeat URL not configured, heartbeat disabled")

    if args.no_http:
        print("HTTP server disabled (--no-http), running heartbeat only")
        try:
            stop_event = threading.Event()
            stop_event.wait()
        finally:
            print("Shutting down application...")
            heartbeat_service.stop()
        return

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        # Stop heartbeat service when app closes
        print("Shutting down application...")
        heartbeat_service.stop()


if __name__ == "__main__":
    main()
