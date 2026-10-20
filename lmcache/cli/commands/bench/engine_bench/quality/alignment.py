# SPDX-License-Identifier: Apache-2.0
"""Chunk alignment of prompt blocks for KV-reuse quality measurements.

A passage cached on its own matches inside a longer prompt only if it starts
on a cache-chunk boundary in both places: chunks are content-addressed, so a
passage that lands off-phase shares no chunk with its cached copy and nothing
is reused, silently.  :class:`ChunkAligner` pads the shared system block and
each passage so every block occupies whole chunks.
"""

# Standard
from typing import Any
import math
import random

# First Party
from lmcache.cli.commands.bench.engine_bench.tokenizers import (
    build_single_token_pool,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Padding words, drawn per block so no two blocks share filler (which would
# make their padded chunks collide in a content-addressed cache).
_FILLER_VOCAB_SIZE = 4096

# Absent from any real passage, and does not merge with template text.
_TEMPLATE_SENTINEL = "██SENTINEL██"

# Re-encode attempts per block; a merged boundary token shifts the estimate.
_PAD_ATTEMPTS = 8


class ChunkAligner:
    """Pads prompt blocks so each occupies a whole number of cache chunks.

    Padding is deterministic: the same tokenizer, chunk size, seed and text
    always yield the same padded block, so two runs build byte-identical
    prompts.  Passage padding is seeded from the passage text, so a passage
    shared by several samples is padded once, identically.
    """

    def __init__(
        self,
        tokenizer: Any,  # transformers is optional; no static type
        model_name: str,
        align_tokens: int,
        seed: int,
    ) -> None:
        """Build the filler pool for *tokenizer*.

        Args:
            tokenizer: A HuggingFace tokenizer for the served model.
            model_name: The model's name, used only in log messages.
            align_tokens: Chunk size to align to.  Should equal the
                deployment's LMCache chunk size.
            seed: Seed for filler selection.

        Raises:
            ValueError: If *align_tokens* is not positive, or the tokenizer
                cannot supply a single-token filler pool.
        """
        if align_tokens < 1:
            raise ValueError(f"align_tokens must be >= 1, got {align_tokens}")
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._align_tokens = align_tokens
        self._seed = seed
        self._pool = build_single_token_pool(tokenizer, _FILLER_VOCAB_SIZE, seed=seed)
        self._passage_blocks: dict[str, str] = {}

    def token_length(self, text: str) -> int:
        """Return the token length of *text*, without special tokens.

        Args:
            text: Any prompt text.

        Returns:
            The number of tokens *text* encodes to.
        """
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def system_block(self, system_prompt: str) -> str:
        """Pad *system_prompt* so the chat-template prefix plus it is aligned.

        Passages only start on a chunk boundary if everything ahead of the
        first one is a whole number of chunks, and the template wrapper is
        part of that.

        Args:
            system_prompt: The instructions placed before the passages.

        Returns:
            The padded system block.
        """
        prefix_tokens = self._chat_prefix_tokens()
        block = self._pad_to_multiple(
            system_prompt, prefix_tokens, random.Random(self._seed)
        )
        logger.info(
            "System block: %d tokens after a %d-token chat-template prefix",
            self.token_length(block),
            prefix_tokens,
        )
        return block

    def passage_block(self, passage: str) -> str:
        """Return *passage* padded to a whole number of chunks.

        Args:
            passage: Raw passage text.

        Returns:
            The padded passage; identical across calls for the same text.
        """
        block = self._passage_blocks.get(passage)
        if block is None:
            rng = random.Random(f"{self._seed}:{passage}")
            block = self._pad_to_multiple(passage, 0, rng)
            self._passage_blocks[passage] = block
        return block

    def _chat_prefix_tokens(self) -> int:
        """Count the tokens the chat template inserts before the content.

        Returns:
            The prefix length, or ``0`` when the model has no chat template —
            alignment is then approximate, and cache reuse partial.
        """
        try:
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": _TEMPLATE_SENTINEL}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:  # noqa: BLE001 - a template failure is non-fatal
            logger.warning(
                "Could not render a chat template for %s (%s); passage "
                "alignment will be approximate",
                self._model_name,
                e,
            )
            return 0

        head = str(rendered).split(_TEMPLATE_SENTINEL)[0]
        return self.token_length(head)

    def _pad_to_multiple(self, text: str, offset: int, rng: random.Random) -> str:
        """Pad *text* so ``offset + len(text)`` is a whole chunk count."""
        align = self._align_tokens
        total = offset + self.token_length(text)
        target = math.ceil(total / align) * align
        if target == total:
            return text

        num_words = target - total
        padded = text
        # Correct against a re-encode: a merged boundary token shifts the
        # estimate, and an off-by-one phase error costs a whole chunk.
        for _ in range(_PAD_ATTEMPTS):
            words = [rng.choice(self._pool.words) for _ in range(max(num_words, 0))]
            padded = text + "\n" + self._pool.join(words)
            actual = offset + self.token_length(padded)
            if actual == target:
                return padded
            num_words += target - actual

        logger.warning(
            "Could not pad a block to a %d-token boundary (off by %d); "
            "cache reuse will be partial",
            align,
            offset + self.token_length(padded) - target,
        )
        return padded
