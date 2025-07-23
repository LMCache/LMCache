# SPDX-License-Identifier: Apache-2.0

"""
Disaggregated Prefill Proxy Server (Decoder to generate First Token)

This proxy server supports two deployment modes:

1. Multi-host mode (different machines, same ports):
   --prefiller-hosts "host1,host2,host3" --prefiller-port 8100
   --decoder-hosts "host4,host5" --decoder-port 8200

2. Single-host multi-port mode (same machine, different ports):
   --prefiller-hosts "localhost" --num-prefillers 3 --prefiller-port 8100
   --decoder-hosts "localhost" --num-decoders 2 --decoder-port 8200
   (This will use ports 8100,8101,8102 for prefillers and 8200,8201 for decoders)

The ZMQ server (--zmq-host/--zmq-port) is used for internal communication
between prefiller and decoder nodes, separate from the main proxy server.
"""

# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional
import argparse
import asyncio
import os
import sys
import time
import traceback

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import msgspec
import numpy as np
import uvicorn
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector.nixl_connector_v3 import (
    NixlMsg,
)

logger = init_logger(__name__)


@dataclass
class ClientInfo:
    client: httpx.AsyncClient
    host: Optional[str] = None
    init_port: Optional[list[int]] = None
    alloc_port: Optional[list[int]] = None

    async def aclose(self):
        await self.client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize clients
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.total_clients = []
    app.state.finished_reqs = set()

    # Initialize prefiller clients
    if "," in global_args.prefiller_hosts:
        # Multi-host mode: each host uses the same port
        prefiller_hosts = [
            host.strip() for host in global_args.prefiller_hosts.split(",")
        ]
        for host in prefiller_hosts:
            prefiller_base_url = f"http://{host}:{global_args.prefiller_port}/v1"
            prefill_client = httpx.AsyncClient(
                timeout=None, base_url=prefiller_base_url
            )
            app.state.prefill_clients.append(
                ClientInfo(
                    prefill_client,
                )
            )
    else:
        # Single-host multi-port mode: use num_prefillers with incremental ports
        for i in range(global_args.num_prefillers):
            port = global_args.prefiller_port + i
            prefiller_base_url = f"http://{global_args.prefiller_hosts}:{port}/v1"
            prefill_client = httpx.AsyncClient(
                timeout=None, base_url=prefiller_base_url
            )
            app.state.prefill_clients.append(
                ClientInfo(
                    prefill_client,
                )
            )

    # Initialize decoder clients
    if "," in global_args.decoder_hosts:
        # Multi-host mode: each host uses the same port
        decoder_hosts = [host.strip() for host in global_args.decoder_hosts.split(",")]
        for host in decoder_hosts:
            decoder_base_url = f"http://{host}:{global_args.decoder_port}/v1"
            decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)
            init_ports = global_args.nixl_init_channel_port
            alloc_ports = global_args.nixl_alloc_channel_port

            app.state.decode_clients.append(
                ClientInfo(
                    decode_client,
                    host,
                    init_ports,
                    alloc_ports,
                )
            )
    else:
        # Single-host multi-port mode: use num_decoders with incremental ports
        for i in range(global_args.num_decoders):
            port = global_args.decoder_port + i
            decoder_base_url = f"http://{global_args.decoder_hosts}:{port}/v1"
            decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)
            init_ports = [p + i for p in global_args.nixl_init_channel_port]
            alloc_ports = [p + i for p in global_args.nixl_alloc_channel_port]

            app.state.decode_clients.append(
                ClientInfo(
                    decode_client,
                    global_args.decoder_hosts,
                    init_ports,
                    alloc_ports,
                )
            )

    app.state.total_clients = app.state.prefill_clients + app.state.decode_clients

    # Only start ZMQ server if there are multiple decoders
    if len(app.state.decode_clients) > 1:
        app.state.zmq_task = asyncio.create_task(zmq_pull_server())
    else:
        app.state.zmq_task = None

    yield

    # Shutdown: Close clients
    for client in app.state.prefill_clients:
        await client.aclose()
    for client in app.state.decode_clients:
        await client.aclose()

    global run_proxy
    run_proxy = False
    if app.state.zmq_task is not None:
        try:
            await asyncio.wait_for(app.state.zmq_task, timeout=5.0)  # 5 seconds timeout
        except asyncio.TimeoutError:
            logger.warning("ZMQ server shutdown timed out after 5 seconds")


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


class StatsCalculator:
    def __init__(self):
        self._stats = []
        self._last_log_time = time.time()

    def add(self, value):
        self._stats.append(value)
        if time.time() - self._last_log_time > 5:
            self._log_stats()
            self._last_log_time = time.time()

    def _log_stats(self):
        # Print average, median, and 99th percentile
        np_arr = np.array(self._stats)
        output_str = (
            f"\nNum requests: {len(self._stats)}"
            + "\nPrefill node TTFT stats:"
            + f"\n - Average (ms): {np.mean(np_arr)}"
            + f"\n - Median (ms): {np.median(np_arr)}"
            + f"\n - 99th Percentile (ms): {np.percentile(np_arr, 99)}\n"
        )
        print(
            "===============================",
            output_str,
            "===============================",
        )


stats_calculator = StatsCalculator()
counter = 0


def csv_ints(s):
    return [int(x) for x in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--prefiller-hosts",
        type=str,
        default="localhost",
        help="Comma-separated list of prefiller hosts "
        "(e.g., 'host1,host2,host3') for multi-host mode",
    )
    parser.add_argument("--prefiller-port", type=int, default=7100)
    parser.add_argument(
        "--num-prefillers",
        type=int,
        default=1,
        help="Number of prefiller instances for "
        "single-host multi-port setup "
        "(mutually exclusive with comma-separated hosts)",
    )
    parser.add_argument(
        "--decoder-hosts",
        type=str,
        default="localhost",
        help="Comma-separated list of decoder hosts "
        "(e.g., 'host1,host2,host3') for multi-host mode",
    )
    parser.add_argument("--decoder-port", type=int, default=7200)
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=1,
        help="Number of decoder instances for "
        "single-host multi-port setup "
        "(mutually exclusive with comma-separated hosts)",
    )
    parser.add_argument("--nixl-init-channel-port", type=csv_ints, default=[7300])
    parser.add_argument("--nixl-alloc-channel-port", type=csv_ints, default=[7400])

    parser.add_argument(
        "--nixl-proxy-host", type=str, default="0.0.0.0", help="NIXL proxy host"
    )
    parser.add_argument(
        "--nixl-proxy-port", type=int, default=7500, help="NIXL proxy port"
    )

    args = parser.parse_args()

    # Validate mutually exclusive parameters
    if "," in args.prefiller_hosts and args.num_prefillers > 1:
        parser.error(
            "--prefiller-hosts with comma-separated values is mutually "
            "exclusive with --num-prefillers > 1"
        )

    if "," in args.decoder_hosts and args.num_decoders > 1:
        parser.error(
            "--decoder-hosts with comma-separated values is mutually "
            "exclusive with --num-decoders > 1"
        )

    return args


# ZMQ communication
zmq_ctx = zmq.asyncio.Context()
run_proxy = True  # Shutdown flag


async def zmq_pull_server():
    socket = zmq_ctx.socket(zmq.PULL)
    nixl_proxy_url = f"{global_args.nixl_proxy_host}:{global_args.nixl_proxy_port}"
    socket.bind(f"tcp://{nixl_proxy_url}")
    logger.info(f"NIXL proxy server started on {nixl_proxy_url}")

    while run_proxy:
        try:
            msg_bytes = await socket.recv()
            msg = msgspec.msgpack.decode(msg_bytes, type=NixlMsg)
            req_id = msg.req_id
            app.state.finished_reqs.add(req_id)
            logger.debug(f"Prefill of req {req_id} done.")
        except zmq.Again:
            await asyncio.sleep(0.01)  # Avoid busy loop
        except Exception as e:
            print("ZMQ Error:", e)
            break

    socket.close()
    logger.info("ZMQ PULL server stopped.")


async def send_request_to_service(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Send a request to a service using a persistent client.
    """
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def stream_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Asynchronously stream the response from a service using a persistent client.
    """
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


def round_robin_pick_client(clients, idx):
    return clients[idx % len(clients)]


async def wait_decode_kv_ready(req_id: str):
    wait_count = 0
    while req_id not in app.state.finished_reqs:
        await asyncio.sleep(0.0001)  # sleep for 0.1 ms
        wait_count += 1
    logger.debug(f"Prefill node signaled kv ready for req {req_id}")
    app.state.finished_reqs.remove(req_id)


@app.post("/v1/completions")
async def handle_completions(request: Request):
    global counter, stats_calculator
    counter += 1
    req_id = str(counter)  # we use counter as req_id

    st = time.time()
    try:
        req_data = await request.json()

        # Pick decode client first
        decode_client = round_robin_pick_client(app.state.decode_clients, counter)

        # Prepare prefill request
        prefill_req_data = req_data.copy()
        prefill_req_data["max_tokens"] = 1
        if "max_completion_tokens" in prefill_req_data:
            prefill_req_data["max_completion_tokens"] = 1

        # Only set disagg_spec if there are multiple decoders
        if len(app.state.decode_clients) > 1:
            disagg_spec = {
                "req_id": req_id,
                "receiver_host": decode_client.host,
                "receiver_init_port": decode_client.init_port,
                "receiver_alloc_port": decode_client.alloc_port,
            }
            prefill_req_data["kv_transfer_params"] = {
                "disagg_spec": disagg_spec,
            }
        else:
            # For single decoder, no need for disagg_spec
            prefill_req_data["kv_transfer_params"] = {}

        # Send request to prefill service round robin
        prefill_client = round_robin_pick_client(app.state.prefill_clients, counter)
        await send_request_to_service(
            prefill_client.client, "/completions", prefill_req_data
        )

        et = time.time()
        stats_calculator.add((et - st) * 1000)  # Convert to milliseconds

        # Stream response from decode service
        async def generate_stream():
            # Wait until decode node signals that kv is ready
            # (only for multiple decoders)
            if len(app.state.decode_clients) > 1:
                await wait_decode_kv_ready(req_id)

            async for chunk in stream_service_response(
                decode_client.client, "/completions", req_data
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server - completions endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    global counter, stats_calculator
    counter += 1
    req_id = str(counter)  # we use counter as req_id

    st = time.time()
    try:
        req_data = await request.json()

        # Pick decode client first
        decode_client = round_robin_pick_client(app.state.decode_clients, counter)

        # Prepare prefill request
        prefill_req_data = req_data.copy()
        prefill_req_data["max_tokens"] = 1
        if "max_completion_tokens" in prefill_req_data:
            prefill_req_data["max_completion_tokens"] = 1

        # Only set disagg_spec if there are multiple decoders
        if len(app.state.decode_clients) > 1:
            disagg_spec = {
                "req_id": req_id,
                "receiver_host": decode_client.host,
                "receiver_init_port": decode_client.init_port,
                "receiver_alloc_port": decode_client.alloc_port,
            }
            prefill_req_data["kv_transfer_params"] = {
                "disagg_spec": disagg_spec,
            }
        else:
            # For single decoder, no need for disagg_spec
            prefill_req_data["kv_transfer_params"] = {}

        # Send request to prefill service
        prefill_client = round_robin_pick_client(app.state.prefill_clients, counter)
        await send_request_to_service(
            prefill_client.client, "/chat/completions", prefill_req_data
        )

        et = time.time()
        stats_calculator.add((et - st) * 1000)  # Convert to milliseconds

        # Stream response from decode service
        async def generate_stream():
            # Wait until decode node signals that kv is ready
            # (only for multiple decoders)
            if len(app.state.decode_clients) > 1:
                await wait_decode_kv_ready(req_id)

            async for chunk in stream_service_response(
                decode_client.client, "/chat/completions", req_data
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        exc_info = sys.exc_info()
        print(
            "Error occurred in disagg prefill proxy server - chat completions endpoint"
        )
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    uvicorn.run(app, host=global_args.host, port=global_args.port)
