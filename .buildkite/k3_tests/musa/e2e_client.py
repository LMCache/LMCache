# SPDX-License-Identifier: Apache-2.0
"""Small dependency-free OpenAI-compatible client used by MUSA CI E2E."""

# Standard
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import argparse
import json
import sys
import time


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one JSON HTTP request and return its decoded response."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        body = ""
        if isinstance(exc, HTTPError):
            body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP request to {url} failed: {exc}; body={body}") from exc


def _model(args: argparse.Namespace) -> None:
    """Print the first model identifier exposed by a serving endpoint."""
    response = _request(args.url)
    models = response.get("data", [])
    if not models or not models[0].get("id"):
        raise RuntimeError(f"no model identifier returned by {args.url}: {response}")
    print(models[0]["id"])


def _completion(args: argparse.Namespace) -> None:
    """Send a deterministic completion request and write a normalized result."""
    prompt = Path(args.prompt_file).read_text()
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": False,
        "top_k": args.top_k,
    }
    started = time.monotonic()
    response = _request(args.url, method="POST", payload=payload)
    elapsed = time.monotonic() - started
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError(f"completion response has no choices: {response}")
    choice = choices[0]
    text = choice.get("text")
    if text is None:
        message = choice.get("message", {})
        text = message.get("content")
    if not isinstance(text, str):
        raise RuntimeError(f"completion response has no text content: {response}")
    result = {
        "text": text,
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
        "response_id": response.get("id"),
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"completion ok: {args.output} elapsed={elapsed:.3f}s")


def _chat_completion(args: argparse.Namespace) -> None:
    """Send a deterministic chat completion and write a normalized result."""
    prompt = Path(args.prompt_file).read_text()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": False,
        "top_k": args.top_k,
    }
    started = time.monotonic()
    response = _request(args.url, method="POST", payload=payload)
    elapsed = time.monotonic() - started
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError(f"chat response has no choices: {response}")
    choice = choices[0]
    message = choice.get("message", {})
    text = message.get("content")
    if not isinstance(text, str):
        raise RuntimeError(f"chat response has no message content: {response}")
    result = {
        "text": text,
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
        "response_id": response.get("id"),
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"chat completion ok: {args.output} elapsed={elapsed:.3f}s")


def _compare(args: argparse.Namespace) -> None:
    """Compare normalized completion text from two result files."""
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())
    if left.get("text") != right.get("text"):
        print(
            json.dumps(
                {"left": left.get("text"), "right": right.get("text")},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise RuntimeError(f"completion text differs: {args.left} != {args.right}")
    print(f"completion text matches: {args.left} == {args.right}")


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model")
    model.add_argument("--url", required=True)
    model.set_defaults(function=_model)
    completion = subparsers.add_parser("completion")
    completion.add_argument("--url", required=True)
    completion.add_argument("--model", required=True)
    completion.add_argument("--prompt-file", required=True)
    completion.add_argument("--max-tokens", type=int, required=True)
    completion.add_argument("--seed", type=int, required=True)
    completion.add_argument("--temperature", type=float, required=True)
    completion.add_argument("--top-k", type=int, required=True)
    completion.add_argument("--output", required=True)
    completion.set_defaults(function=_completion)
    chat_completion = subparsers.add_parser("chat-completion")
    chat_completion.add_argument("--url", required=True)
    chat_completion.add_argument("--model", required=True)
    chat_completion.add_argument("--prompt-file", required=True)
    chat_completion.add_argument("--max-tokens", type=int, required=True)
    chat_completion.add_argument("--seed", type=int, required=True)
    chat_completion.add_argument("--temperature", type=float, required=True)
    chat_completion.add_argument("--top-k", type=int, required=True)
    chat_completion.add_argument("--output", required=True)
    chat_completion.set_defaults(function=_chat_completion)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.set_defaults(function=_compare)
    return parser


def main() -> int:
    """Parse arguments, run the selected operation, and return an exit code."""
    args = _parser().parse_args()
    try:
        args.function(args)
    except RuntimeError as exc:
        print(f"e2e-client: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
