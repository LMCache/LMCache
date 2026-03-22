# SPDX-License-Identifier: Apache-2.0
"""Stream one OpenAI-compatible completion and emit token/latency metrics."""

# Standard
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request

# Third Party
from transformers import AutoTokenizer  # type: ignore[import-untyped]

# First Party
from lmcache.cli.commands.base import BaseCommand

_MAX_ERR = 65536
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_LATENCY_METRIC_ROWS = (
    ("ttft_ms", "TTFT (ms)"),
    ("tpot_ms_per_token", "TPOT (ms/token)"),
    ("total_latency_ms", "Total latency (ms)"),
    ("throughput_tokens_per_s", "Throughput (tokens/s)"),
)
_BUILTIN_CORPORA = {
    "ffmpeg": (
        "ffmpeg — multimedia framework. Example: ffmpeg -i in.mp4 "
        "-c:v libx264 out.mk4\n"
    ),
}


def _clip(text: str, limit: int = _MAX_ERR) -> str:
    return (
        text
        if len(text) <= limit
        else text[: max(0, limit - 24)] + "\n...(message truncated)..."
    )


def _info(msg: str) -> None:
    print(f"lmcache query: {msg}", file=sys.stderr)


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def _openai_error(obj: dict[str, Any]) -> Optional[str]:
    err = obj.get("error")
    if err is None:
        return None
    if isinstance(err, str):
        return err.strip() or None
    if not isinstance(err, dict):
        return _clip(str(err))
    for key in ("message", "detail"):
        val = err.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        typ = err.get("type") or err.get("code")
        if key == "message" and isinstance(typ, str) and typ.strip():
            return f"{typ.strip()}: {val.strip()}"
        return val.strip()
    try:
        return _clip(json.dumps(err, ensure_ascii=False))
    except Exception:
        return _clip(str(err))


def _raise_openai_error(obj: dict[str, Any]) -> None:
    msg = _openai_error(obj)
    if msg:
        raise RuntimeError(_clip(msg))


def _raise_json_blob_error(blob: str) -> None:
    s = blob.strip()
    if not s.startswith("{"):
        return
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return
    if isinstance(obj, dict):
        _raise_openai_error(obj)


def _api_url(base: str, path: str) -> str:
    base = base.strip()
    if "://" not in base:
        base = f"http://{base}"
    base = base.rstrip("/")
    return f"{base if base.endswith('/v1') else base + '/v1'}/{path}"


def _read_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="GET"), timeout=max(timeout + 2.0, 5.0)
        ) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:512]
        raise RuntimeError(
            f"GET {url} failed (HTTP {e.code}): {body or 'no body'}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GET {url} failed: {getattr(e, 'reason', e)}") from e
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from GET {url}: {e}") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"GET {url}: expected a JSON object")
    return obj


def _first_model_id(base: str, timeout: float) -> str:
    obj = _read_json(_api_url(base, "models"), timeout)
    data = obj.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(
            "GET /v1/models returned no models; pass --model explicitly."
        )
    first = data[0]
    if not isinstance(first, dict) or "id" not in first:
        raise RuntimeError("GET /v1/models: first entry missing 'id'.")
    return str(first["id"])


def _merge_corpora(corpus_args: list[str]) -> dict[str, str]:
    corpora = dict(_BUILTIN_CORPORA)
    for item in corpus_args:
        if "=" not in item:
            raise ValueError(f"Invalid --corpus {item!r}; expected name=path")
        name, path = [x.strip() for x in item.split("=", 1)]
        if not name:
            raise ValueError(f"Invalid --corpus {item!r}; empty name")
        p = Path(path).expanduser()
        if not p.is_file():
            raise ValueError(f"Corpus file not found for {name!r}: {p}")
        corpora[name] = p.read_text(encoding="utf-8", errors="replace")
    return corpora


def _unknown_corpus(key: str) -> None:
    raise ValueError(
        f"Unknown corpus {key!r}. Define it with --corpus {key}=PATH "
        f"or use a built-in: {', '.join(sorted(_BUILTIN_CORPORA))}."
    )


def expand_prompt_with_breakdown(
    prompt: str, corpus_args: list[str]
) -> tuple[str, Optional[tuple[str, ...]]]:
    corpora = _merge_corpora(corpus_args)
    seen: set[str] = set()
    order: list[str] = []
    for m in _PLACEHOLDER.finditer(prompt):
        key = m.group(1)
        if key not in corpora:
            _unknown_corpus(key)
        if key not in seen:
            seen.add(key)
            order.append(key)
    filled = _PLACEHOLDER.sub(lambda m: corpora[m.group(1)], prompt)
    return filled, (tuple(order) if order else None)


@lru_cache(maxsize=8)
def _load_tokenizer(model_id: str) -> Optional[Any]:
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None


def _split_ints(total: int, weights: list[int]) -> list[int]:
    if total <= 0 or not weights or sum(weights) <= 0:
        return [0] * len(weights)
    exact = [total * w / sum(weights) for w in weights]
    base = [math.floor(x) for x in exact]
    for i in sorted(
        range(len(weights)), key=lambda i: exact[i] - base[i], reverse=True
    )[: total - sum(base)]:
        base[i] += 1
    return base


def _token_weights(
    prompt_template: str, corpus_args: list[str], model_id: str
) -> Optional[tuple[list[tuple[str, int]], int]]:
    tok = _load_tokenizer(model_id)
    if tok is None or not _PLACEHOLDER.search(prompt_template):
        return None
    corpora = _merge_corpora(corpus_args)
    counts: dict[str, int] = {}
    order: list[str] = []
    literal, pos = 0, 0

    def enc(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    for m in _PLACEHOLDER.finditer(prompt_template):
        literal += enc(prompt_template[pos : m.start()])
        key = m.group(1)
        if key not in corpora:
            _unknown_corpus(key)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += enc(corpora[key])
        pos = m.end()
    literal += enc(prompt_template[pos:])
    return [(k, counts[k]) for k in order], literal


def _add_prompt_metrics(
    metrics: Any,
    breakdown: Optional[tuple[str, ...]],
    prompt_tokens: int,
    *,
    prompt_template: str,
    corpus_args: list[str],
    model_id: str,
) -> None:
    metrics.add("prompt_tokens", "Prompt tokens", prompt_tokens)
    if not breakdown or prompt_tokens <= 0:
        return
    try:
        r = _token_weights(prompt_template, corpus_args, model_id)
    except Exception:
        return
    if not r:
        return
    parts, literal = r
    weights = [w for _, w in parts] + [literal]
    if not any(weights):
        return
    alloc = _split_ints(prompt_tokens, weights)
    for i, (name, _) in enumerate(parts):
        metrics.add(f"prompt_corpus_{name}", f"  Corpus '{name}'", alloc[i])
    metrics.add("prompt_query", "  Query", alloc[-1])


def _sse_piece(obj: dict[str, Any], chat: bool) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    c0 = choices[0]
    return (
        str((c0.get("delta") or {}).get("content") or "")
        if chat
        else str(c0.get("text") or "")
    )


def _trim_misc_buffer(misc: list[str], limit: int = _MAX_ERR) -> None:
    while misc and sum(map(len, misc)) > limit:
        misc.pop(0)


def _stream(
    url: str,
    body: dict[str, Any],
    timeout: float,
    *,
    chat: bool,
    max_tokens: int,
) -> dict[str, Any]:
    """POST with ``stream: true``; parse SSE; return TTFT/TPOT and token metrics."""
    payload = {
        **body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0, first_token_t, pieces, usage, misc = time.time(), None, [], None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    misc.append(line)
                    _trim_misc_buffer(misc)
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    misc.append(chunk)
                    _trim_misc_buffer(misc)
                    continue
                if not isinstance(obj, dict):
                    continue
                _raise_openai_error(obj)
                piece = _sse_piece(obj, chat)
                if piece:
                    first_token_t = first_token_t or time.time()
                    pieces.append(piece)

                u_chunk = obj.get("usage")
                if u_chunk is not None:
                    usage = u_chunk
            t1 = time.time()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        _raise_json_blob_error(err_body)
        raise RuntimeError(
            _clip(f"POST {url} failed (HTTP {e.code}):\n{_clip(err_body)}")
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"POST {url} failed: {getattr(e, 'reason', e)}") from e

    misc_text = "\n".join(misc).strip()
    _raise_json_blob_error(misc_text)
    joined = "".join(pieces)
    if not joined and usage is None:
        raise RuntimeError(
            _clip(f"No completion output from engine. Captured response:\n{misc_text}")
            if misc_text
            else "Empty response from engine (no SSE chunks parsed)."
        )

    u = usage or {}
    prompt_tokens = int(u.get("prompt_tokens") or 0)
    num_completion = int(u.get("completion_tokens") or 0)
    # Match V2RequestSender: server count if present, else max_tokens cap.
    num_generated = num_completion if num_completion > 0 else max_tokens
    if first_token_t is None:
        ttft_s = -1.0
        decode_time = 0.0
    else:
        ttft_s = first_token_t - t0
        decode_time = t1 - first_token_t
    dt = t1 - t0
    decoding_speed = (num_generated / decode_time) if decode_time > 0 else 0.0
    tpot_s = (
        (decode_time / num_generated) if num_generated > 0 and decode_time > 0 else 0.0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": num_generated,
        "ttft_ms": ttft_s * 1000.0,
        "tpot_ms_per_token": tpot_s * 1000.0,
        "total_latency_ms": dt * 1000.0,
        "throughput_tokens_per_s": decoding_speed,
    }


def _query_once(
    base: str, model: str, prompt: str, max_tokens: int, timeout: float, *, chat: bool
) -> dict[str, Any]:
    path = "chat/completions" if chat else "completions"
    body = (
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if chat
        else {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    )
    return _stream(
        _api_url(base, path), body, timeout, chat=chat, max_tokens=max_tokens
    )


def _missing_chat_template(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "chat template",
            "chat_template",
            "chattemplate",
            "template resolution",
            "must provide a chat template",
            "default chat template is no longer allowed",
        )
    )


def _weak_completions_error(msg: str) -> bool:
    msg = msg.lower()
    return any(
        s in msg
        for s in (
            "empty response from engine",
            "no completion output from engine",
            "no sse chunks parsed",
        )
    )


def _query_with_fallback(
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    *,
    completions_only: bool,
    chat_first: bool,
) -> dict[str, Any]:
    if completions_only:
        return _query_once(base, model, prompt, max_tokens, timeout, chat=False)
    try:
        return _query_once(base, model, prompt, max_tokens, timeout, chat=chat_first)
    except RuntimeError as first_err:
        if chat_first:
            if not _missing_chat_template(first_err):
                raise
            _info("chat API failed (no chat template); retrying with /v1/completions")
            return _query_once(base, model, prompt, max_tokens, timeout, chat=False)
        _info("/v1/completions failed; retrying with /v1/chat/completions")
        try:
            return _query_once(base, model, prompt, max_tokens, timeout, chat=True)
        except RuntimeError as second_err:
            if _weak_completions_error(str(first_err)) and _missing_chat_template(
                second_err
            ):
                _info(
                    "base / completion-only models: try `--completions` or an instruct "
                    "model with a chat template."
                )
                raise second_err
            raise RuntimeError(f"{first_err}; then {second_err}") from second_err


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--format",
        type=str,
        default=None,
        metavar="FORMAT",
        help="Stdout output format (default: terminal). Available: terminal, json.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Save metrics to a file at PATH (format chosen by --format).",
    )


class QueryCommand(BaseCommand):
    def name(self) -> str:
        return "query"

    def help(self) -> str:
        return "Run one inference request and report TTFT/TPOT metrics."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        pass

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description=(
                "Run one OpenAI-compatible inference request and report metrics."
            ),
        )
        inner = p.add_subparsers(dest="query_target", required=True, metavar="{engine}")
        eng = inner.add_parser(
            "engine",
            help="Send one completion request to the engine OpenAI-compatible HTTP API",
        )
        eng.add_argument(
            "--url",
            required=True,
            help=(
                "Engine HTTP base (e.g. http://localhost:8000 or .../v1). "
                "Scheme defaults to http:// if omitted."
            ),
        )
        eng.add_argument(
            "--prompt",
            required=True,
            help=(
                "Text with optional {name} placeholders "
                "(built-ins: ffmpeg; or define via --corpus NAME=PATH)."
            ),
        )
        eng.add_argument(
            "--model",
            default=None,
            metavar="ID",
            help=(
                "Model id for the engine API. If omitted, GET /v1/models chooses the "
                "first listed model."
            ),
        )
        eng.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Maximum completion tokens (default: 128).",
        )
        eng.add_argument(
            "--timeout",
            type=float,
            default=30.0,
            help="HTTP timeout in seconds (default: 30).",
        )
        eng.add_argument(
            "--corpus",
            action="append",
            default=[],
            metavar="NAME=PATH",
            help="Load file text for {NAME} in --prompt (repeatable).",
        )
        eng.add_argument(
            "--completions", action="store_true", help="Use POST /v1/completions only."
        )
        eng.add_argument(
            "--chat-first",
            action="store_true",
            help=(
                "Try /v1/chat/completions first, then /v1/completions on missing chat "
                "template."
            ),
        )
        _add_output_args(eng)
        eng.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        try:
            prompt, breakdown = expand_prompt_with_breakdown(args.prompt, args.corpus)
        except ValueError as e:
            _die(str(e))

        model = args.model
        if not model:
            try:
                model = _first_model_id(args.url, args.timeout)
            except RuntimeError as e:
                _die(str(e))

        try:
            stats = _query_with_fallback(
                args.url,
                model,
                prompt,
                args.max_tokens,
                args.timeout,
                completions_only=args.completions,
                chat_first=args.chat_first,
            )
        except RuntimeError as e:
            _die(str(e))

        metrics = self.create_metrics("Query Engine Result", args, width=41)
        _add_prompt_metrics(
            metrics,
            breakdown,
            int(stats["prompt_tokens"]),
            prompt_template=args.prompt,
            corpus_args=args.corpus,
            model_id=model,
        )
        metrics.add("output_tokens", "Output tokens", stats["output_tokens"])
        metrics.add("model", "Model", model)
        lat = metrics.add_section("latency", "Latency Metrics")
        for key, label in _LATENCY_METRIC_ROWS:
            lat.add(key, label, round(stats[key], 2))
        metrics.emit()
