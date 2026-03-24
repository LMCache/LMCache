# SPDX-License-Identifier: Apache-2.0
"""Run one OpenAI-compatible inference request and report token/latency metrics."""

# Standard
from typing import Any
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand, _add_output_args
from lmcache.cli.prompt import PromptBuilder
from lmcache.cli.request import Request

MetricValue = tuple[str, Any]
MetricMap = dict[str, MetricValue]


class QueryCommand(BaseCommand):
    """CLI command that sends one request to a serving engine."""

    def name(self) -> str:
        return "query"

    def help(self) -> str:
        return "Run one inference request and report metrics."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        pass

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description=(
                "Run one OpenAI-compatible inference request and report metrics."
            ),
        )
        inner = parser.add_subparsers(
            dest="query_target",
            required=True,
            metavar="{engine,kvcache}",
        )
        self._register_engine(inner)
        self._register_kvcache(inner)

    def _register_engine(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "engine",
            help="Send one request to an OpenAI-compatible HTTP API.",
        )
        parser.add_argument("--url", required=True, help="Serving engine base URL.")
        parser.add_argument(
            "--prompt",
            required=True,
            help="Prompt text with optional {name} placeholders.",
        )
        parser.add_argument(
            "--model",
            default=None,
            metavar="ID",
            help="Model ID for the serving engine.",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Maximum completion tokens (default: 128).",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=30.0,
            help="HTTP timeout in seconds (default: 30).",
        )
        parser.add_argument(
            "--documents",
            action="extend",
            nargs="+",
            default=[],
            metavar="NAME=PATH",
            help=(
                "Load file text for {NAME} in --prompt. "
                "Accepts one or more NAME=PATH values."
            ),
        )
        parser.add_argument(
            "--path",
            dest="documents",
            action="extend",
            nargs="+",
            metavar="NAME=PATH",
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--completions",
            action="store_true",
            help="Use POST /v1/completions only.",
        )
        parser.add_argument(
            "--chat-first",
            action="store_true",
            help="Try /v1/chat/completions first, then fall back to /v1/completions.",
        )
        _add_output_args(parser)
        parser.set_defaults(func=self.execute)

    def _register_kvcache(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "kvcache",
            help="Query KV-cache endpoints (not implemented yet).",
        )
        _add_output_args(parser)
        parser.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        handlers = {
            "engine": self.query_engine,
            "kvcache": self.query_kvcache,
        }
        handler = handlers.get(args.query_target)
        if handler is None:
            print(f"Unknown query target: {args.query_target}", file=sys.stderr)
            sys.exit(1)
        handler(args)

    def query_engine(self, args: argparse.Namespace) -> None:
        try:
            prompt_builder = PromptBuilder(args.prompt, args.documents)
            sender = Request(
                base=args.url,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                completions_only=args.completions,
                chat_first=args.chat_first,
            )
            engine_stats = sender.send_request(prompt_builder.complete_prompt)

            model_id = args.model or str(engine_stats["model"][1])
            prompt_stats = prompt_builder.get_token_stats(model_id)
            metrics = self.create_metrics("Query Engine", args)
            metrics.add("model", "Model", model_id)
            for key, (name, value) in prompt_stats.items():
                if key == "prompt_tokens":
                    continue
                metrics.add(key, name, int(value))

            latency = metrics.add_section("latency", "Latency Metrics")
            for key, (name, value) in engine_stats.items():
                if key == "model":
                    continue
                latency.add(key, name, round(float(value), 2))

            metrics.emit()
        except (RuntimeError, ValueError) as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

    def query_kvcache(self, args: argparse.Namespace) -> None:
        pass
