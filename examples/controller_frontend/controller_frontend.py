# SPDX-License-Identifier: Apache-2.0
"""
Controller Frontend Server

This server provides a web interface for managing LMCache Controller.
It serves static files and proxies API requests to the actual Controller.
"""

# Standard
from typing import Any, Dict
import os
import time

# Third Party
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import uvicorn

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

app = FastAPI(
    title="LMCache Controller Frontend",
    description="Web interface for LMCache Controller",
)

# Configuration
CONTROLLER_HOST = os.getenv("CONTROLLER_HOST", "localhost")
CONTROLLER_PORT = os.getenv("CONTROLLER_PORT", "9000")
CONTROLLER_BASE_URL = f"http://{CONTROLLER_HOST}:{CONTROLLER_PORT}"

# Serve static files from the static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# HTTP client for proxying requests
client = httpx.AsyncClient(timeout=30.0)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint for the frontend server."""
    return {"status": "healthy", "service": "controller_frontend"}


# Proxy endpoints for Controller API
# These endpoints forward requests to the actual Controller


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(request: Request, path: str):
    """
    Proxy all API requests to the Controller.

    This handles all HTTP methods and forwards them to the Controller.
    """
    # Build the target URL
    target_url = f"{CONTROLLER_BASE_URL}/{path}"

    # Get request data
    method = request.method
    headers = dict(request.headers)

    # Remove host header to avoid confusion
    headers.pop("host", None)

    # Get request body if present
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
        except Exception:
            body = None

    try:
        # Make request to Controller
        response = await client.request(
            method=method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params,
        )

        # Return the response
        return JSONResponse(
            content=response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {"text": response.text},
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    except httpx.ConnectError as err:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Controller at {CONTROLLER_BASE_URL}. "
            "Make sure the Controller is running.",
        ) from err
    except httpx.TimeoutException as err:
        raise HTTPException(
            status_code=504, detail="Request to Controller timed out."
        ) from err
    except Exception as err:
        logger.error("Error proxying request to Controller", exc_info=err)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request. "
            "Please try again later.",
        ) from err


# Additional endpoints for frontend-specific functionality
# These endpoints provide data that might not be available directly from Controller


@app.get("/api/frontend/instances")
async def get_frontend_instances() -> Dict[str, Any]:
    """
    Get instance information for the frontend.

    This endpoint aggregates data from Controller endpoints
    to provide a comprehensive view for the frontend.
    """
    try:
        # Get worker info from Controller
        response = await client.post(
            f"{CONTROLLER_BASE_URL}/query_worker_info", json={"instance_id": "all"}
        )

        if response.status_code != 200:
            return {"instances": [], "error": "Failed to get worker info"}

        data = response.json()
        worker_infos = data.get("worker_infos", [])

        # Group workers by instance
        instances_map = {}
        for worker in worker_infos:
            instance_id = worker.get("instance_id")
            if instance_id not in instances_map:
                instances_map[instance_id] = {
                    "instance_id": instance_id,
                    "ip": worker.get("ip", ""),
                    "workers": [],
                    "worker_count": 0,
                    "last_heartbeat": worker.get("last_heartbeat_time", 0),
                }

            instances_map[instance_id]["workers"].append(worker)
            instances_map[instance_id]["worker_count"] += 1

            # Update latest heartbeat
            if (
                worker.get("last_heartbeat_time", 0)
                > instances_map[instance_id]["last_heartbeat"]
            ):
                instances_map[instance_id]["last_heartbeat"] = worker.get(
                    "last_heartbeat_time", 0
                )

        instances = list(instances_map.values())

        # Add status based on heartbeat
        current_time = time.time()
        for instance in instances:
            time_diff = current_time - instance["last_heartbeat"]
            if time_diff < 60:
                instance["status"] = "active"
            elif time_diff < 300:
                instance["status"] = "warning"
            else:
                instance["status"] = "inactive"

        return {"instances": instances}

    except Exception as e:
        logger.error("Error getting frontend instances", exc_info=e)
        return {
            "instances": [],
            "error": "Failed to retrieve instance information. Please try again later.",
        }


@app.get("/api/frontend/stats")
async def get_frontend_stats() -> Dict[str, Any]:
    """
    Get frontend statistics.
    """
    try:
        # Get worker info for instance and worker counts
        response = await client.post(
            f"{CONTROLLER_BASE_URL}/query_worker_info", json={"instance_id": "all"}
        )

        if response.status_code != 200:
            return {"stats": {}, "error": "Failed to get worker info"}

        data = response.json()
        worker_infos = data.get("worker_infos", [])

        # Calculate stats
        instance_ids = set(worker.get("instance_id") for worker in worker_infos)

        stats = {
            "instance_count": len(instance_ids),
            "worker_count": len(worker_infos),
            "active_workers": sum(
                1
                for w in worker_infos
                if w.get("last_heartbeat_time", 0) > time.time() - 60
            ),
            "timestamp": time.time(),
        }

        return {"stats": stats}

    except Exception as e:
        logger.error("Error getting frontend stats", exc_info=e)
        return {
            "stats": {},
            "error": "Failed to retrieve statistics. Please try again later.",
        }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize HTTP client on startup."""
    global client
    client = httpx.AsyncClient(timeout=30.0)
    logger.info(f"Controller Frontend starting. Controller URL: {CONTROLLER_BASE_URL}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up HTTP client on shutdown."""
    await client.aclose()
    logger.info("Controller Frontend shutting down.")


def main():
    """Main entry point."""
    # Standard
    import argparse

    parser = argparse.ArgumentParser(description="LMCache Controller Frontend Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8500, help="Port to bind to")
    parser.add_argument(
        "--controller-host", type=str, default="localhost", help="Controller host"
    )
    parser.add_argument(
        "--controller-port", type=int, default=9000, help="Controller port"
    )

    args = parser.parse_args()

    # Update configuration based on arguments
    global CONTROLLER_HOST, CONTROLLER_PORT, CONTROLLER_BASE_URL
    CONTROLLER_HOST = args.controller_host
    CONTROLLER_PORT = args.controller_port
    CONTROLLER_BASE_URL = f"http://{CONTROLLER_HOST}:{CONTROLLER_PORT}"

    logger.info(f"Starting Controller Frontend on {args.host}:{args.port}")
    logger.info(f"Proxying to Controller at {CONTROLLER_BASE_URL}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
