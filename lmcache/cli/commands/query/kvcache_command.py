# SPDX-License-Identifier: Apache-2.0
"""``lmcache query kvcache`` — report how much of a prompt is already cached."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.bench.engine_bench.tokenizers import try_load_tokenizer
from lmcache.cli.commands.query._lookup import post_lookup, summarize_coverage
from lmcache.cli.commands.query._prompt import PromptBuilder


class KVCacheCommand(BaseCommand):
    """Look up how much of one prompt the KV cache already holds."""

    def name(self) -> str:
        return "kvcache"

    def help(self) -> str:
        return "Report how much of one prompt is already in the KV cache."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--url",
            required=True,
            help="Coordinator base URL, e.g. http://localhost:9000.",
        )
        parser.add_argument(
            "--prompt",
            required=True,
            help="Prompt text with optional {name} placeholders.",
        )
        parser.add_argument(
            "--model",
            required=True,
            metavar="ID",
            help=(
                "Model whose tokenizer and rank fan-out the lookup uses. It must "
                "match the model the engine prefills with."
            ),
        )
        parser.add_argument(
            "--world-size",
            type=int,
            default=1,
            metavar="N",
            help="World size selecting the per-rank key fan-out (default: 1).",
        )
        parser.add_argument(
            "--cache-salt",
            default="",
            metavar="SALT",
            help="Per-tenant isolation salt the keys were stored under.",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=10.0,
            help="HTTP timeout in seconds (default: 10).",
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

    def execute(self, args: argparse.Namespace) -> None:
        try:
            prompt = PromptBuilder(args.prompt, args.documents).complete_prompt
            token_ids = self._tokenize(prompt, args.model)
            body = post_lookup(
                args.url,
                {
                    "token_ids": token_ids,
                    "model_name": args.model,
                    "world_size": args.world_size,
                    "cache_salt": args.cache_salt,
                },
                timeout=args.timeout,
            )
            coverage = summarize_coverage(body, prompt_tokens=len(token_ids))

            metrics = self.create_metrics("Query KV Cache", args)
            metrics.add("model", "Model", args.model)
            metrics.add("prompt_tokens", "Prompt tokens", coverage.prompt_tokens)
            metrics.add("cached_chunks_hit", "Cached chunks", coverage.cached_chunks)
            metrics.add("cached_chunks_total", "Total chunks", coverage.total_chunks)
            metrics.add("cached_tokens_hit", "Cached tokens", coverage.cached_tokens)
            metrics.add("cache_status", "Cache status", coverage.status)
            metrics.add(
                "cached_chunk_location",
                "Cache locations",
                ", ".join(coverage.locations) or "--",
            )
            if coverage.tier_chunks:
                table = metrics.add_table(
                    "cached_chunks_by_tier",
                    "Cached chunks by tier",
                    ("tier", "tier"),
                    ("chunks", "chunks"),
                )
                for tier, count in coverage.tier_chunks:
                    table.add_row(tier=tier, chunks=count)
            metrics.emit()
        except (RuntimeError, ValueError, OSError) as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

    def _tokenize(self, prompt: str, model: str) -> list[int]:
        """Tokenize *prompt* with *model*'s tokenizer.

        Args:
            prompt: The expanded prompt text.
            model: HuggingFace repo ID or local path of the model.

        Returns:
            The prompt's token IDs.

        Raises:
            RuntimeError: If no tokenizer is available, since coverage cannot
                be computed without token IDs and a silent zero-coverage
                answer would read as a cache miss.
        """
        tokenizer = try_load_tokenizer(model)
        if tokenizer is None:
            raise RuntimeError(
                f"No tokenizer available for {model!r}: install `transformers` "
                "and make sure the model ID is reachable, or point --model at a "
                "local tokenizer directory."
            )
        return list(tokenizer.encode(prompt))
