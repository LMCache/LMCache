#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import time
from datetime import datetime
from typing import Any

import httpx


BASE_TEXT = (
    "A systems researcher is studying how an inference cache changes the latency profile of a long language model prompt. The notes discuss attention keys, attention values, memory tiers, token chunks, and the careful measurement of cold and warm requests.\n\n"
)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def make_prompt(request_id: int, prompt_repeats: int = 90) -> str:
    prompt = f"Request {request_id}: " + (BASE_TEXT * prompt_repeats) + "A systems researcher is studying how an inference cache changes the latency profile of a long language model prompt. The" # , attention values, memory tiers, token chunks, and the careful measurement of cold
    return prompt


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    phase: str,
    request_id: int,
    repeat: int,
    start_gate: asyncio.Event | None = None,
) -> dict[str, Any] | None:
    if start_gate is not None:
        await start_gate.wait()

    try:
        resp = await client.post(url, json=payload)
        text = resp.text
        resp.raise_for_status()

        try:
            body = resp.json()
        except json.JSONDecodeError:
            body = {"raw_text": text}

        print(
            f"[{timestamp()}] phase={phase} "
            f"request={request_id} repeat={repeat} {json.dumps(body)}",
            flush=True,
        )
        return body

    except Exception as exc:
        print(
            f"[{timestamp()}] phase={phase} "
            f"request={request_id} repeat={repeat} ERROR {exc!r}",
            flush=True,
        )
        return None


async def run_phase(
    client: httpx.AsyncClient,
    app_url: str,
    endpoint: str,
    phase: str,
    num_requests: int,
    compression: float,
    max_tokens: int,
    prompt_repeats: int,
    repeat: int = 0,
    start_together: bool = False,
) -> list[dict[str, Any] | None]:
    url = f"{app_url}{endpoint}"

    start_gate = asyncio.Event() if start_together else None
    tasks = []

    for request_id in range(0, num_requests):
        payload = {
            "prompt": make_prompt(request_id, prompt_repeats),
            "drop_algorithm": "random",
            "drop_remap_pe": True,
            "drop_compression": compression,
            "max_tokens": max_tokens,
            "id": request_id,
        }

        tasks.append(
            asyncio.create_task(
                post_json(
                    client=client,
                    url=url,
                    payload=payload,
                    phase=phase,
                    request_id=request_id,
                    repeat=repeat,
                    start_gate=start_gate,
                )
            )
        )

    if start_gate is not None:
        await asyncio.sleep(0)
        start_gate.set()

    return await asyncio.gather(*tasks)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-host", default="127.0.0.1")
    parser.add_argument("--app-port", type=int, default=9000)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--compression", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-repeats", type=int, default=90)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sleep-between-repeats", type=float, default=0.0)
    parser.add_argument(
        "--separate-repeats",
        action="store_true",
        help="Wait after each repeat instead of launching all repeats together.",
    )
    args = parser.parse_args()

    app_url = f"http://{args.app_host}:{args.app_port}"

    limits = httpx.Limits(
        max_connections=max(args.num_requests * max(args.repeat, 1) + 32, 256),
        max_keepalive_connections=max(args.num_requests + 32, 128),
    )

    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        start_ns = time.time_ns()
        print(f"[{timestamp()}] phase=prefill start", flush=True)
        await run_phase(
            client,
            app_url,
            "/api/prefill",
            "prefill",
            args.num_requests,
            args.compression,
            args.max_tokens,
            args.prompt_repeats,
            repeat=0,
            start_together=True,
        )
        print(f"[{timestamp()}] phase=prefill done", flush=True)

        await asyncio.sleep(10)

        print(f"[{timestamp()}] phase=retrieve_drop start", flush=True)
        await run_phase(
            client,
            app_url,
            "/api/retrieve_drop",
            "retrieve_drop",
            args.num_requests,
            args.compression,
            args.max_tokens,
            args.prompt_repeats,
            repeat=0,
            start_together=True,
        )
        print(f"[{timestamp()}] phase=retrieve_drop done", flush=True)

        start_ns = time.time_ns()

        if args.separate_repeats:
            for repeat in range(args.repeat):
                print(f"[{timestamp()}] phase=decode repeat={repeat} start", flush=True)

                await run_phase(
                    client,
                    app_url,
                    "/api/chat_completion",
                    "decode",
                    args.num_requests,
                    args.compression,
                    args.max_tokens,
                    args.prompt_repeats,
                    repeat=repeat,
                    start_together=True,
                )
                print(f"[{timestamp()}] phase=decode repeat={repeat} done", flush=True)

                if repeat < args.repeat - 1 and args.sleep_between_repeats > 0:
                    await asyncio.sleep(args.sleep_between_repeats)
        else:
            tasks = []
            for repeat in range(args.repeat):

                tasks.append(
                    run_phase(
                        client,
                        app_url,
                        "/api/chat_completion",
                        "decode",
                        args.num_requests,
                        args.compression,
                        args.max_tokens,
                        args.prompt_repeats,
                        repeat=repeat,
                        start_together=True,
                    )
                )
            await asyncio.gather(*tasks)

        end_ns = time.time_ns()
        elapsed_ms = (end_ns - start_ns) // 1_000_000
        total_requests = args.num_requests * args.repeat
        total_tokens = total_requests * args.max_tokens

        print(
            f"Total elapsed time for {total_requests} decode requests: "
            f"{elapsed_ms} ms",
            flush=True,
        )
        print(
            f"Decode throughput for {total_tokens} requested output tokens: "
            f"{total_tokens * 1000 / elapsed_ms:.2f} tokens/s",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())