# SPDX-License-Identifier: Apache-2.0
"""``lmcache query kvcache`` — report cache coverage for one prompt."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.query._lookup import CacheLookup, CoverageResult
from lmcache.cli.commands.query._prompt import PromptBuilder

_DEFAULT_CHUNK_SIZE = 256


def _format_locations(locations: list[tuple[str, str]]) -> str:
    """Render ``(instance_id, location)`` pairs as ``[location@instance, ...]``."""
    if not locations:
        return "none"
    return "[" + ", ".join(f"{loc}@{inst}" for inst, loc in locations) + "]"


class KVCacheCommand(BaseCommand):
    """Report how much of a prompt's KV cache is already cached."""

    def name(self) -> str:
        """Return the subcommand name."""
        return "kvcache"

    def help(self) -> str:
        """Return short help text shown by ``lmcache query -h``."""
        return "Report KV cache coverage for one prompt."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add ``query kvcache`` arguments.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        parser.add_argument(
            "--url",
            required=True,
            help="Controller HTTP endpoint (e.g. http://localhost:5555).",
        )
        parser.add_argument(
            "--prompt",
            required=True,
            help="Prompt text with optional {name} document placeholders.",
        )
        parser.add_argument(
            "--model",
            required=True,
            metavar="ID",
            help="Tokenizer/model id used to derive token IDs.",
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
            "--chunk-size",
            type=int,
            default=_DEFAULT_CHUNK_SIZE,
            help=(
                "Tokens per cache chunk for chunk-count display "
                f"(default: {_DEFAULT_CHUNK_SIZE}). Must match the server's "
                "configured chunk size to be exact."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Look up and report cache coverage for the prompt.

        Args:
            args: Parsed CLI arguments.
        """
        try:
            prompt_builder = PromptBuilder(args.prompt, args.documents)
            lookup = CacheLookup(
                url=args.url,
                model=args.model,
                chunk_size=args.chunk_size,
            )
            result = lookup.run(prompt_builder.complete_prompt)
            self._emit(result, args)
        except (RuntimeError, ValueError) as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

    def _emit(self, result: CoverageResult, args: argparse.Namespace) -> None:
        """Render *result* through the metrics framework.

        Args:
            result: The cache-coverage summary.
            args: Parsed CLI arguments (for output format/handlers).
        """
        metrics = self.create_metrics("Query KV Cache", args)
        metrics.add("model", "Model", args.model)
        metrics.add("prompt_tokens", "Prompt tokens", result.prompt_tokens)
        metrics.add(
            "cached_tokens",
            "Cached tokens",
            f"{result.cached_tokens}/{result.prompt_tokens}",
        )
        metrics.add(
            "cached_chunks",
            "Cached chunks",
            f"{result.cached_chunks}/{result.total_chunks}",
        )
        metrics.add(
            "cache_locations", "Cache locations", _format_locations(result.locations)
        )
        metrics.add("cache_status", "Cache status", result.cache_status)
        metrics.emit()
