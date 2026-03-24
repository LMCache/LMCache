# SPDX-License-Identifier: Apache-2.0
"""Stream one OpenAI-compatible completion and emit token/latency metrics."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.extend_prompt import (
    add_prompt_metrics,
    expand_prompt_with_breakdown,
)
from lmcache.cli.request import first_model_id, query_with_fallback

_LATENCY_METRIC_ROWS = (
    ("ttft_ms", "TTFT (ms)"),
    ("tpot_ms_per_token", "TPOT (ms/token)"),
    ("total_latency_ms", "Total latency (ms)"),
    ("throughput_tokens_per_s", "Throughput (tokens/s)"),
)


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


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
                "(built-ins: ffmpeg; or define via --documents NAME=PATH)."
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
            "--documents",
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
            prompt, breakdown = expand_prompt_with_breakdown(
                args.prompt, args.documents
            )
        except ValueError as e:
            _die(str(e))

        model = args.model
        if not model:
            try:
                model = first_model_id(args.url, args.timeout)
            except RuntimeError as e:
                _die(str(e))

        try:
            stats = query_with_fallback(
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
        add_prompt_metrics(
            metrics,
            breakdown,
            int(stats["prompt_tokens"]),
            prompt_template=args.prompt,
            documents_args=args.documents,
            model_id=model,
        )
        metrics.add("output_tokens", "Output tokens", stats["output_tokens"])
        metrics.add("model", "Model", model)
        lat = metrics.add_section("latency", "Latency Metrics")
        for key, label in _LATENCY_METRIC_ROWS:
            lat.add(key, label, round(stats[key], 2))
        metrics.emit()
