# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import asynccontextmanager
import argparse
import json
import os
import time

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import numpy as np


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize clients
    decoder_base_url = f"http://{global_args.decoder_host}:{global_args.decoder_port}"

    app.state.prefill_clients = []
    for i in range(global_args.num_prefillers):
        port = int(global_args.prefiller_port) + i
        prefiller_base_url = f"http://{global_args.prefiller_host}:{port}"
        prefill_client = httpx.AsyncClient(timeout=None, base_url=prefiller_base_url)
        app.state.prefill_clients.append(prefill_client)

    app.state.decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)

    yield

    # Shutdown: Close clients
    for client in app.state.prefill_clients:
        await client.aclose()
    await app.state.decode_client.aclose()


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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-host", type=str, default="localhost")
    parser.add_argument("--prefiller-port", type=int, default=8100)
    parser.add_argument("--num-prefillers", type=int, default=1)
    parser.add_argument("--decoder-host", type=str, default="localhost")
    parser.add_argument("--decoder-port", type=int, default=8200)
    args = parser.parse_args()
    return args


# Initialize variables to hold the persistent clients
app.state.prefill_clients = []
app.state.decode_client = None


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


@app.post("/v1/completions")
async def handle_completions(request: Request):
    global counter, stats_calculator
    counter += 1

    st = time.time()
    try:
        req_data = await request.json()

        stream = req_data.get("stream", False)
        media_type = "text/event-stream" if stream else "application/json"

        tokenization_client = round_robin_pick_client(
            app.state.prefill_clients + [app.state.decode_client], counter
        )

        tokenize_output = await send_request_to_service(
            tokenization_client, "/tokenize", {"prompt": req_data["prompt"]}
        )
        tokenize_output = tokenize_output.json()

        org_max_tokens = req_data["max_tokens"]
        req_data["prompt"] = tokenize_output["tokens"]
        req_data["max_tokens"] = 1
        req_data["kv_transfer_params"] = {"ret_first_tok": True}
        req_data["stream"] = False
        stream_options = req_data.pop("stream_options", None)

        # Send request to prefill service round robin, ignore the response
        client = round_robin_pick_client(app.state.prefill_clients, counter)
        prefill_output = await send_request_to_service(
            client, "/v1/completions", req_data
        )

        prefill_output = prefill_output.json()

        et = time.time()
        stats_calculator.add(et - st)

        req_data["max_tokens"] = org_max_tokens - 1
        req_data["prompt"].append(prefill_output["kv_transfer_params"]["first_tok"])
        req_data.pop("kv_transfer_params")
        req_data["stream"] = True
        if stream_options is not None:
            req_data["stream_options"] = stream_options

        # Stream response from decode service
        async def generate_stream():
            head_chunk = {
                "id": prefill_output["id"],
                "object": "text_completion",
                "created": prefill_output["created"],
                "model": prefill_output["model"],
                "choices": [
                    {
                        "index": 0,
                        "text": prefill_output["choices"][0]["text"],
                        "logprobs": None,
                        "finish_reason": None,
                        "stop_reason": None,
                    }
                ],
                "usage": None,
            }
            yield (
                "data: " + json.dumps(head_chunk, separators=(",", ":")) + "\n\n"
            ).encode()

            async for chunk in stream_service_response(
                app.state.decode_client, "/v1/completions", req_data
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type=media_type)

    except Exception as e:
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server - completions endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    global counter, stats_calculator
    counter += 1

    st = time.time()
    try:
        req_data = await request.json()

        stream = req_data.get("stream", False)
        media_type = "text/event-stream" if stream else "application/json"

        org_max_tokens = req_data["max_tokens"]
        req_data["max_tokens"] = 1

        org_max_completion_tokens = None
        if "max_completion_tokens" in req_data:
            org_max_completion_tokens = req_data["max_completion_tokens"]
            req_data["max_completion_tokens"] = 1

        # Send request to prefill service, ignore the response
        client = round_robin_pick_client(app.state.prefill_clients, counter)
        await send_request_to_service(client, "/v1/chat/completions", req_data)

        et = time.time()
        stats_calculator.add(et - st)

        req_data["max_tokens"] = org_max_tokens
        if org_max_completion_tokens is not None:
            req_data["max_completion_tokens"] = org_max_completion_tokens

        # Stream response from decode service
        async def generate_stream():
            async for chunk in stream_service_response(
                app.state.decode_client, "/v1/chat/completions", req_data
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type=media_type)

    except Exception as e:
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(
            "Error occurred in disagg prefill proxy server  - chat completions endpoint"
        )
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    # Third Party
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
