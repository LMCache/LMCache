#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Disaggregated prefill / decode proxy server.

Supports multiple prefiller and decoder instances with independent
round-robin load balancing across each pool.

Flow for /v1/chat/completions and /v1/completions:
  1. Send prefill request (max_tokens=1) to prefiller with X-Request-Id
  2. Wait for telemetry notification (KV cache ready)
  3. Forward original request to decoder

Usage (single instance):
    python one_proxy.py \
        --port 10001 \
        --prefiller-host localhost --prefiller-port 8100 \
        --decoder-host localhost   --decoder-port 8200 \
        --telemetry-port 5768

Usage (multiple instances):
    python one_proxy.py \
        --port 10001 \
        --prefiller-host localhost --prefiller-port 8100,8101 \
        --decoder-host localhost   --decoder-port 8200,8201,8202 \
        --telemetry-port 5768

    # Or auto-expand from a base port:
    python one_proxy.py \
        --port 10001 \
        --prefiller-host localhost --prefiller-port 8100 --num-prefillers 2 \
        --decoder-host localhost   --decoder-port 8200 --num-decoders 3 \
        --telemetry-port 5768
"""

# Standard
from contextlib import asynccontextmanager
from typing import Optional
import argparse
import asyncio
import itertools
import os
import threading
import time
import uuid

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Do not return exception text to clients (avoids leaking stack paths and internals).
_PROXY_CLIENT_ERROR_MESSAGE = "Internal server error"

# ---------------------------------------------------------------------------
# Pending-request tracking (shared between proxy app and telemetry app)
# ---------------------------------------------------------------------------
pending_requests: dict[str, asyncio.Event] = {}
# TP worker arrivals per request: world_size and which ranks have checked in.
pending_tp_state: dict[str, dict] = {}
pending_requests_lock = threading.Lock()
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:16]


def create_pending_request(request_id: str) -> asyncio.Event:
    """Create an asyncio.Event for a request and store it."""
    event = asyncio.Event()
    with pending_requests_lock:
        pending_requests[request_id] = event
    return event


def remove_pending_request(request_id: str):
    """Remove a pending request and its TP state from the dictionary."""
    with pending_requests_lock:
        pending_requests.pop(request_id, None)
        pending_tp_state.pop(request_id, None)


def notify_request(
    chatcmpl_request_id: str,
    world_size: int = 1,
    kv_rank: int = 0,
) -> bool:
    """
    Record that one TP worker has finished storing KV cache.
    Only signals the event when all ``world_size`` workers have reported.
    Returns True if all workers are done and the event was signaled.
    """
    with pending_requests_lock:
        # vLLM wraps the request ID as "chatcmpl-{uuid}-{suffix}",
        # so strip the first and last segments to recover the original UUID.
        request_id = "-".join(chatcmpl_request_id.split("-")[1:])
        request_id = request_id[:16]
        event = pending_requests.get(request_id, None)
        if event is None:
            return False

        if request_id not in pending_tp_state:
            pending_tp_state[request_id] = {
                "world_size": world_size,
                "received_ranks": set(),
            }
        state = pending_tp_state[request_id]
        state["received_ranks"].add(kv_rank)
        logger.info(
            f"Request {request_id}: TP worker kv_rank={kv_rank} reported "
            f"({len(state['received_ranks'])}/{state['world_size']})"
        )

        if len(state["received_ranks"]) >= state["world_size"]:
            pending_tp_state.pop(request_id, None)
            if main_event_loop is not None:
                main_event_loop.call_soon_threadsafe(event.set)
            return True
    return False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

    # --- Prefiller client(s) ------------------------------------------------
    pref_hosts = global_args.prefiller_host
    pref_ports = global_args.prefiller_port

    app.state.prefill_clients = []
    pref_pairs = _pair_hosts_and_ports(
        pref_hosts, pref_ports, global_args.num_prefillers
    )
    for host, port in pref_pairs:
        base = f"http://{host}:{int(port)}"
        client = httpx.AsyncClient(timeout=None, base_url=base)
        app.state.prefill_clients.append(client)
        logger.info(f"Prefiller client: {base}")

    # --- Decoder client(s) --------------------------------------------------
    dec_hosts = global_args.decoder_host
    dec_ports = global_args.decoder_port

    app.state.decode_clients = []
    dec_pairs = _pair_hosts_and_ports(dec_hosts, dec_ports, global_args.num_decoders)
    for host, port in dec_pairs:
        base = f"http://{host}:{int(port)}"
        client = httpx.AsyncClient(timeout=None, base_url=base)
        app.state.decode_clients.append(client)
        logger.info(f"Decoder client: {base}")

    yield

    # Shutdown: close all clients
    for c in app.state.prefill_clients:
        await c.aclose()
    for c in app.state.decode_clients:
        await c.aclose()
    logger.info("All clients closed")


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------
def csv_ints(s):
    return [int(x) for x in s.split(",")]


def csv_strs(s):
    return [x.strip() for x in s.split(",")]


def _pair_hosts_and_ports(hosts, ports, count=None):
    """Flexible host-port pairing with expansion strategies."""
    if not isinstance(hosts, list):
        hosts = [hosts]
    if not isinstance(ports, list):
        ports = [ports]
    if len(hosts) == 1 and len(ports) == 1:
        if count is None or count <= 1:
            return [(hosts[0], ports[0])]
        else:
            return [(hosts[0], ports[0] + i) for i in range(count)]
    if len(hosts) == 1:
        return [(hosts[0], p) for p in ports]
    if len(ports) == 1:
        return [(h, ports[0]) for h in hosts]
    if len(hosts) != len(ports):
        raise ValueError("Length mismatch between hosts and ports")
    return list(zip(hosts, ports, strict=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disaggregated prefill/decode proxy server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10001,
        help="Port to run the proxy server on (default: 10001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the proxy server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--prefiller-host",
        type=csv_strs,
        default=["localhost"],
        help="Prefiller host(s) (default: localhost)",
    )
    parser.add_argument(
        "--prefiller-port",
        type=csv_ints,
        default=[8100],
        help="Prefiller port(s) (default: 8100)",
    )
    parser.add_argument(
        "--num-prefillers",
        type=int,
        default=1,
        help="Number of prefillers (default: 1)",
    )
    parser.add_argument(
        "--decoder-host",
        type=csv_strs,
        default=["localhost"],
        help="Decoder host(s) (default: localhost)",
    )
    parser.add_argument(
        "--decoder-port",
        type=csv_ints,
        default=[8200],
        help="Decoder port(s) (default: 8200)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=1,
        help="Number of decoders (default: 1)",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=5768,
        help="Port for telemetry server (default: 5768)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Round-robin helpers
# ---------------------------------------------------------------------------
_prefill_rr = itertools.count()
_decode_rr = itertools.count()


def _pick_prefill_client() -> tuple[int, httpx.AsyncClient]:
    """Pick the next prefiller client using round-robin. Returns (index, client)."""
    clients = app.state.prefill_clients
    idx = next(_prefill_rr) % len(clients)
    return idx, clients[idx]


def _pick_decode_client() -> tuple[int, httpx.AsyncClient]:
    """Pick the next decoder client using round-robin. Returns (index, client)."""
    clients = app.state.decode_clients
    idx = next(_decode_rr) % len(clients)
    return idx, clients[idx]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _build_headers(**extra: str) -> dict[str, str]:
    """Build common HTTP headers, including auth if OPENAI_API_KEY is set."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra)
    return headers


async def _send_prefill(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
):
    """Send a request to prefiller with X-Request-Id header."""
    headers = _build_headers(**{"X-Request-Id": request_id})
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def _send_decode(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
):
    """Send a non-streaming request to decoder."""
    headers = _build_headers()
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def _stream_decode(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
):
    """Stream the response from decoder."""
    headers = _build_headers()
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


# ---------------------------------------------------------------------------
# Disaggregated prefill/decode handler
# ---------------------------------------------------------------------------
async def _handle_disagg_request(request: Request, endpoint: str):
    """
    Common handler for disaggregated prefill/decode requests.
    Works for both /v1/completions and /v1/chat/completions.

    Steps:
      1. Send prefill request (max_tokens=1) to prefiller with X-Request-Id
      2. Wait for telemetry notification (KV cache ready)
      3. Forward original request to decoder
    """
    try:
        req_data = await request.json()
        is_stream = req_data.get("stream", False)

        logger.info(f"Received {endpoint} request (stream={is_stream})")

        pf_idx, prefill_client = _pick_prefill_client()
        dc_idx, decode_client = _pick_decode_client()
        logger.info(f"Routing to prefiller[{pf_idx}] / decoder[{dc_idx}]")

        # -- Step 1: Disaggregated prefill -> prefiller -----------------------
        request_id = generate_request_id()
        logger.info(f"Received {endpoint} request with generated ID: {request_id}")

        event = create_pending_request(request_id)
        try:
            prefill_req_data = req_data.copy()
            prefill_req_data["max_tokens"] = 1
            if "max_completion_tokens" in prefill_req_data:
                prefill_req_data["max_completion_tokens"] = 1
            prefill_req_data["stream"] = False
            prefill_req_data.pop("stream_options", None)

            prefill_send_time = time.monotonic()
            await _send_prefill(prefill_client, endpoint, prefill_req_data, request_id)
            prefill_first_response_time = time.monotonic()
            prefill_duration = prefill_first_response_time - prefill_send_time
            logger.info(
                f"Request {request_id}: prefill request"
                f" duration = {prefill_duration:.4f}s"
            )

            # Wait for telemetry notification (KV cache ready)
            await event.wait()
            notify_time = time.monotonic()
            notify_wait_duration = notify_time - prefill_first_response_time
            logger.info(
                f"Request {request_id}: finished saving KV caches after prefill"
                f" response = {notify_wait_duration * 1000:.2f}ms"
            )
            logger.debug(f"Event signaled for {request_id}, forwarding to decoder")

        finally:
            remove_pending_request(request_id)

        # -- Step 2: Decode -> decoder ----------------------------------------
        if is_stream:

            async def generate_stream():
                first_chunk = True
                async for chunk in _stream_decode(decode_client, endpoint, req_data):
                    if first_chunk:
                        decode_first_response_time = time.monotonic()
                        latency = (
                            decode_first_response_time - prefill_first_response_time
                        )
                        logger.info(
                            f"Request {request_id}: latency between prefill first "
                            f"response and decode first response = "
                            f"{latency * 1000:.2f}ms"
                        )
                        first_chunk = False
                    yield chunk

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            response = await _send_decode(decode_client, endpoint, req_data)
            return JSONResponse(content=response.json())

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Backend returned error: {e.response.status_code} - {e.response.text}"
        )
        return JSONResponse(
            content={
                "error": {
                    "message": f"Backend error: {e.response.text}",
                    "type": "backend_error",
                    "code": e.response.status_code,
                }
            },
            status_code=e.response.status_code,
        )
    except Exception as e:
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        logger.error(f"Error in {endpoint} endpoint")
        logger.error(str(e))
        logger.error("".join(traceback.format_exception(*exc_info)))
        return JSONResponse(
            content={
                "error": {
                    "message": _PROXY_CLIENT_ERROR_MESSAGE,
                    "type": "proxy_error",
                    "code": 500,
                }
            },
            status_code=500,
        )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    """Handle /v1/chat/completions requests."""
    return await _handle_disagg_request(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def handle_completions(request: Request):
    """Handle /v1/completions requests."""
    return await _handle_disagg_request(request, "/v1/completions")


@app.get("/v1/models")
async def list_models(request: Request):
    """Forward model list request to any reachable prefiller."""
    headers = _build_headers()
    last_err: Exception | None = None
    for client in app.state.prefill_clients:
        try:
            response = await client.get("/v1/models", headers=headers)
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except Exception as e:
            last_err = e
            logger.warning(
                "Prefiller %s unreachable for /v1/models: %s", client.base_url, e
            )
    if last_err is not None:
        logger.error(
            "All prefillers unreachable for /v1/models",
            exc_info=last_err,
        )
    else:
        logger.error("All prefillers unreachable for /v1/models")
    return JSONResponse(
        content={
            "error": {
                "message": _PROXY_CLIENT_ERROR_MESSAGE,
                "type": "proxy_error",
            }
        },
        status_code=500,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint -- probes every prefiller and decoder."""
    headers = _build_headers()
    prefillers_ok: list[bool] = []
    decoders_ok: list[bool] = []

    for client in app.state.prefill_clients:
        try:
            resp = await client.get("/v1/models", headers=headers)
            resp.raise_for_status()
            prefillers_ok.append(True)
        except Exception:
            prefillers_ok.append(False)

    for client in app.state.decode_clients:
        try:
            resp = await client.get("/v1/models", headers=headers)
            resp.raise_for_status()
            decoders_ok.append(True)
        except Exception:
            decoders_ok.append(False)

    all_healthy = all(prefillers_ok) and all(decoders_ok)
    body = {
        "status": "healthy" if all_healthy else "degraded",
        "prefillers": {
            "total": len(prefillers_ok),
            "healthy": sum(prefillers_ok),
        },
        "decoders": {
            "total": len(decoders_ok),
            "healthy": sum(decoders_ok),
        },
    }
    return JSONResponse(content=body, status_code=200 if all_healthy else 503)


# ---------------------------------------------------------------------------
# Telemetry FastAPI app (runs on a separate port in a background thread)
# ---------------------------------------------------------------------------
telemetry_app = FastAPI()


@telemetry_app.post("/api/v1/telemetry")
async def handle_telemetry(request: Request):
    """
    Handle telemetry POST requests.

    Expected payload format (from FastAPIRequestTelemetry):
    {
        "event": "request_store_finished",
        "request_ids_set": ["id1", "id2", ...],
        "model_name": "...",
        "world_size": N,
        "kv_rank": K,
    }
    """
    try:
        payload = await request.json()
        event_type = payload.get("event")
        request_ids = payload.get("request_ids_set", [])
        world_size = payload.get("world_size", 1)
        kv_rank = payload.get("kv_rank", 0)

        for request_id in request_ids:
            logger.info(
                f"Received telemetry event: {event_type} "
                f"for request: {request_id} "
                f"(kv_rank={kv_rank}, world_size={world_size})"
            )

        notified_count = 0
        for request_id in request_ids:
            if notify_request(request_id, world_size=world_size, kv_rank=kv_rank):
                notified_count += 1
                logger.info(
                    f"All {world_size} TP worker(s) done for request: {request_id}"
                )

        return JSONResponse(
            content={
                "status": "ok",
                "notified": notified_count,
                "total": len(request_ids),
            }
        )

    except Exception as e:
        logger.error("Error processing telemetry", exc_info=e)
        return JSONResponse(
            content={
                "status": "error",
                "message": _PROXY_CLIENT_ERROR_MESSAGE,
            },
            status_code=500,
        )


def _run_telemetry_server(host: str, port: int):
    """Run the telemetry server in a separate thread."""
    # Third Party
    import uvicorn

    uvicorn.run(telemetry_app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    # Third Party
    import uvicorn

    # Start telemetry server in a background thread
    telemetry_thread = threading.Thread(
        target=_run_telemetry_server,
        args=(global_args.host, global_args.telemetry_port),
        daemon=True,
    )
    telemetry_thread.start()
    logger.info(
        f"Telemetry server started on {global_args.host}:{global_args.telemetry_port}"
    )

    logger.info(f"Starting proxy on {global_args.host}:{global_args.port}")

    uvicorn.run(
        app,
        host=global_args.host,
        port=global_args.port,
        log_level="info",
    )
