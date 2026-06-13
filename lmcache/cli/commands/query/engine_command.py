# SPDX-License-Identifier: Apache-2.0
"""``lmcache query engine`` — send one request to an OpenAI-compatible API."""

# Standard
from typing import TYPE_CHECKING
import argparse
import sys

# First Party
from lmcache.cli.commands.base import _add_output_args
from lmcache.cli.commands.query.prompt import PromptBuilder
from lmcache.cli.commands.query.request import Request

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand


def register_engine_parser(
    subparsers: argparse._SubParsersAction,
    dispatch_func,
) -> argparse.ArgumentParser:
    """Register the ``lmcache query engine`` subcommand parser.

    Args:
        subparsers: The ``query`` subparsers action.
        dispatch_func: Function to bind via ``set_defaults(func=...)``.

    Returns:
        The created ``ArgumentParser``.
    """
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
    parser.set_defaults(func=dispatch_func)
    return parser


def run_query_engine(cmd: "BaseCommand", args: argparse.Namespace) -> None:
    """Execute the ``lmcache query engine`` subcommand.

    Args:
        cmd: The parent command instance (for metrics creation).
        args: Parsed CLI arguments.
    """
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
        metrics = cmd.create_metrics("Query Engine", args)
        metrics.add("model", "Model", model_id)
        prompt_name, prompt_value = engine_stats["prompt_tokens"]
        metrics.add("prompt_tokens", prompt_name, int(prompt_value))
        output_name, output_value = engine_stats["output_tokens"]
        metrics.add("output_tokens", output_name, int(output_value))

        latency = metrics.add_section("latency", "Latency Metrics")
        for key, (name, value) in engine_stats.items():
            if key in ("model", "prompt_tokens", "output_tokens"):
                continue
            latency.add(key, name, round(float(value), 2))

        metrics.emit()
    except (RuntimeError, ValueError) as err:
        print(str(err), file=sys.stderr)
        sys.exit(1)
