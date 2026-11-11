# SPDX-License-Identifier: Apache-2.0
"""Measure time-to-first-token (TTFT) for a single OpenAI-style completion.

This is a deliberately small, dependency-light helper for the cpu_hello_world
demo: it sends one streaming ``/v1/completions`` request and records the wall
clock time until the first token arrives. The demo driver calls it once per
request (cold, warm, negative) and compares the numbers.

For a fuller benchmarking client (context sweeps, tokenizer-aware truncation,
cache flushing) see ``examples/online_session/``.
"""

# Standard
import argparse
import json
import time
import urllib.request


def measure_ttft(
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[float, int]:
    """Send one streaming completion and measure time to first token.

    Args:
        api_base: Base URL of the OpenAI-compatible server, e.g.
            ``http://127.0.0.1:18000/v1``.
        model: Model id the server was launched with.
        prompt: The full prompt text to send.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A tuple ``(ttft_seconds, streamed_chunks)`` where ``ttft_seconds`` is
        the wall-clock time until the first non-empty streamed chunk and
        ``streamed_chunks`` is the number of SSE data chunks received.

    Raises:
        RuntimeError: If the server never streams a first token.
    """
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    ttft_seconds = -1.0
    streamed_chunks = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            streamed_chunks += 1
            if ttft_seconds < 0:
                ttft_seconds = time.perf_counter() - start

    if ttft_seconds < 0:
        raise RuntimeError("server did not stream any tokens")
    return ttft_seconds, streamed_chunks


def main() -> None:
    """Parse args, measure one request's TTFT, print it, and append JSONL."""
    parser = argparse.ArgumentParser(description="Measure TTFT for one request.")
    parser.add_argument("--api-base", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Path to a UTF-8 file containing the full prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--label",
        default="run",
        help="Label recorded in the JSONL output (e.g. cold/warm/negative).",
    )
    parser.add_argument(
        "--out",
        default="ttft.jsonl",
        help="JSONL file to append the result to.",
    )
    args = parser.parse_args()

    with open(args.prompt_file, encoding="utf-8") as handle:
        prompt = handle.read()

    ttft_seconds, streamed_chunks = measure_ttft(
        api_base=args.api_base,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
    )

    record = {
        "label": args.label,
        "prompt_chars": len(prompt),
        "ttft_seconds": round(ttft_seconds, 4),
        "streamed_chunks": streamed_chunks,
    }
    with open(args.out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    print(f"{args.label}: TTFT = {ttft_seconds:.4f}s")


if __name__ == "__main__":
    main()
