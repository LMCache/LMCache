# SPDX-License-Identifier: Apache-2.0
"""Chunk alignment of prompt blocks for KV-reuse quality measurements.

A passage cached on its own matches inside a longer prompt only if it starts
on a cache-chunk boundary in both places: chunks are content-addressed, so a
passage that lands off-phase shares no chunk with its cached copy and nothing
is reused, silently.  :class:`ChunkAligner` pads the shared system block and
each passage so every block occupies whole chunks, and
:meth:`ChunkAligner.place_blocks` checks, on the rendered chat prompt, that
they really do.

What fills the padding is the :class:`Filler`:

* ``words`` (legacy, and ``rag-qa-quality``'s) appends random single-token
  words.
* ``dots`` / ``dashes`` wrap each passage as
  ``Passage <id>:\\n<text>\\n\\n . . … .\\nend of passage <id>`` -- a repeated
  low-information token the model can read as a divider.  The block starts
  with a letter and ends with a digit, so no tokenizer merges it with its
  neighbours, and no chat template trims it (see :func:`block_id`).
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import enum
import hashlib
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

# Chunk targets tried per block by the unit fillers before giving up: a
# boundary merge can make a target unreachable, never the next one too.
_PAD_TARGETS = 3


class Filler(enum.Enum):
    """What pads a block to a whole number of chunks."""

    WORDS = "words"
    """Random single-token words (legacy; ``rag-qa-quality`` uses it).
    Noisy for the model to read."""

    DOTS = "dots"
    """Repeated ``" ."`` inside a ``Passage <id>: … end of passage <id>``
    frame."""

    DASHES = "dashes"
    """Repeated ``" -"`` inside the same frame as ``dots``."""


_PAD_UNITS = {Filler.DOTS: " .", Filler.DASHES: " -"}

# Ends the padded system block in the framed layouts: a fixed digit, so the
# first passage's "Passage" never merges into it.
_SYSTEM_TAIL = "\nend of instructions 0"


def block_id(text: str) -> str:
    """Return the id a framed block is labelled with.

    Six hex characters of the text's digest plus one decimal digit, so the
    id -- and the block, which ends with it -- always ends in a digit. Every
    tokenizer checked splits a digit from a following letter or newline, while
    a final letter merges into the next block's first word on some, a final
    punctuation mark glues to the newlines after it on others, and final
    whitespace is merged with its neighbours or stripped by chat templates.

    Args:
        text: The raw passage text.

    Returns:
        The id, e.g. ``"4f7a1c9"``. Identical for identical text.
    """
    digest = hashlib.sha256(text.encode()).hexdigest()
    return digest[:6] + str(int(digest, 16) % 10)


@dataclass(frozen=True)
class BlockPlacement:
    """Where one block sits in a rendered prompt.

    Attributes:
        block: The block text.
        start_token: Its first token's index in the tokenized prompt, or
            ``-1`` when the block was not found verbatim.
        tokens: Its token ids in this prompt.
        problem: Why it cannot be reused, or ``""`` when it starts and ends
            on chunk boundaries with no token straddling either.
    """

    block: str
    start_token: int
    tokens: tuple[int, ...]
    problem: str


class ChunkAligner:
    """Pads prompt blocks so each occupies a whole number of cache chunks.

    Padding is deterministic: the same tokenizer, chunk size, filler and text
    always yield the same padded block (and, for ``words``, the same seed),
    so two runs build byte-identical prompts.  A passage shared by several
    samples is padded once, identically.
    """

    def __init__(
        self,
        tokenizer: Any,  # transformers is optional; no static type
        model_name: str,
        align_tokens: int,
        seed: int,
        filler: Filler = Filler.WORDS,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the aligner (and, for ``words``, the filler pool).

        Args:
            tokenizer: A HuggingFace tokenizer for the served model.
            model_name: The model's name, used only in log messages.
            align_tokens: Chunk size to align to.  Should equal the
                deployment's LMCache chunk size.
            seed: Seed for filler selection (``words`` only).
            filler: What pads each block.
            template_kwargs: Chat-template variables the engine renders
                with; they can change the template's length ahead of the
                content.

        Raises:
            ValueError: If *align_tokens* is not positive, or the tokenizer
                cannot supply a single-token filler pool for ``words``.
        """
        if align_tokens < 1:
            raise ValueError(f"align_tokens must be >= 1, got {align_tokens}")
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._align_tokens = align_tokens
        self._seed = seed
        self._filler = filler
        self._template_kwargs = dict(template_kwargs or {})
        self._pool = (
            build_single_token_pool(tokenizer, _FILLER_VOCAB_SIZE, seed=seed)
            if filler is Filler.WORDS
            else None
        )
        self._passage_blocks: dict[str, tuple[str, int]] = {}
        self._system_pad_tokens = 0

    @property
    def filler(self) -> Filler:
        """What pads each block."""
        return self._filler

    @property
    def system_pad_tokens(self) -> int:
        """Padding tokens in the last block :meth:`system_block` returned."""
        return self._system_pad_tokens

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
        if self._filler is Filler.WORDS:
            block = self._pad_to_multiple(
                system_prompt, prefix_tokens, random.Random(self._seed)
            )
            pad = self.token_length(block) - self.token_length(system_prompt)
        else:
            block, pad = self._pad_framed(
                "", system_prompt, _SYSTEM_TAIL, prefix_tokens
            )
        self._system_pad_tokens = pad
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
        return self._padded_passage(passage)[0]

    def pad_tokens(self, passage: str) -> int:
        """Return how many padding tokens :meth:`passage_block` adds.

        Args:
            passage: Raw passage text.

        Returns:
            Tokens of filler (``words``: the words; framed fillers: the
            repeated unit), excluding the frame around the text.
        """
        return self._padded_passage(passage)[1]

    def place_blocks(
        self, messages: list[dict[str, str]], blocks: list[str]
    ) -> tuple[int, list[BlockPlacement]]:
        """Locate *blocks* in the prompt the engine will tokenize.

        Renders *messages* with the chat template (as the engine does),
        tokenizes the result, and checks each block: it must appear
        verbatim, in order, start and end on chunk boundaries, and no token
        may straddle either boundary. Only then do its chunks hash the same
        here as in any other prompt holding it -- which callers check by
        comparing ``tokens`` across prompts.

        Args:
            messages: The chat messages, as sent to the engine.
            blocks: The padded blocks the messages contain, in order.

        Returns:
            The rendered prompt's token count, and one placement per block.
        """
        rendered = self._render(messages)
        ids = self._encode(rendered)
        placements: list[BlockPlacement] = []
        cursor = 0
        for block in blocks:
            start = rendered.find(block, cursor)
            if start < 0:
                placements.append(
                    BlockPlacement(
                        block,
                        -1,
                        (),
                        "not found verbatim in the rendered prompt (the chat "
                        "template changed it, e.g. trimmed whitespace)",
                    )
                )
                continue
            end = start + len(block)
            cursor = end
            first, clean_start = self._token_boundary(rendered, start, ids)
            last, clean_end = self._token_boundary(rendered, end, ids)
            problems = []
            if not clean_start:
                problems.append("a token straddles its start")
            if not clean_end:
                problems.append("a token straddles its end")
            if clean_start and first % self._align_tokens:
                problems.append(
                    f"starts at token {first}, not on a "
                    f"{self._align_tokens}-token chunk boundary"
                )
            if clean_start and clean_end and (last - first) % self._align_tokens:
                problems.append(
                    f"spans {last - first} tokens, not a whole number of chunks"
                )
            placements.append(
                BlockPlacement(
                    block, first, tuple(ids[first:last]), "; ".join(problems)
                )
            )
        return len(ids), placements

    def _padded_passage(self, passage: str) -> tuple[str, int]:
        """Pad *passage* once and cache the block with its padding count."""
        cached = self._passage_blocks.get(passage)
        if cached is None:
            if self._filler is Filler.WORDS:
                rng = random.Random(f"{self._seed}:{passage}")
                block = self._pad_to_multiple(passage, 0, rng)
                cached = (block, self.token_length(block) - self.token_length(passage))
            else:
                ident = block_id(passage)
                cached = self._pad_framed(
                    f"Passage {ident}:\n", passage, f"\nend of passage {ident}", 0
                )
            self._passage_blocks[passage] = cached
        return cached

    def _render(self, messages: list[dict[str, str]]) -> str:
        """Render *messages* as the engine will, or their content without one."""
        try:
            return str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self._template_kwargs,
                )
            )
        except Exception:  # noqa: BLE001 - no template: content only, as below
            return "".join(m["content"] for m in messages)

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _token_boundary(
        self, rendered: str, char_index: int, ids: list[int]
    ) -> tuple[int, bool]:
        """Map a character position to a token boundary.

        Args:
            rendered: The rendered prompt.
            char_index: A character position in it.
            ids: The rendered prompt's token ids.

        Returns:
            The number of tokens before *char_index*, and whether a token
            boundary falls exactly there (the text before it tokenizes to a
            prefix of the whole prompt's tokens).
        """
        head = self._encode(rendered[:char_index])
        return len(head), ids[: len(head)] == head

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
                **self._template_kwargs,
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

    def _pad_framed(
        self, head: str, text: str, tail: str, offset: int
    ) -> tuple[str, int]:
        """Pad ``head + text + "\\n\\n" + unit*n + tail`` to a chunk boundary.

        Args:
            head: Text before *text* (the passage label, or ``""``).
            text: The passage or system prompt.
            tail: Text after the padding (the end marker).
            offset: Tokens already ahead of the block.

        Returns:
            The padded block and its padding token count. When no count
            lands on a boundary the closest attempt is returned; the
            in-context check (:meth:`place_blocks`) then reports it.
        """
        unit = _PAD_UNITS[self._filler]
        align = self._align_tokens

        def build(n: int) -> str:
            return head + text + "\n\n" + unit * n + tail

        base = self.token_length(build(0))
        target = math.ceil((offset + base) / align) * align
        block = build(0)
        for _ in range(_PAD_TARGETS):
            n = target - offset - base
            for _ in range(_PAD_ATTEMPTS):
                n = max(n, 0)
                block = build(n)
                actual = offset + self.token_length(block)
                if actual == target:
                    return block, actual - offset - base
                n += target - actual
            target += align
        logger.warning(
            "Could not pad a block to a %d-token boundary with %r; cache "
            "reuse will be partial",
            align,
            unit,
        )
        return block, self.token_length(block) - base

    def _pad_to_multiple(self, text: str, offset: int, rng: random.Random) -> str:
        """Pad *text* so ``offset + len(text)`` is a whole chunk count."""
        pool = self._pool
        if pool is None:
            raise RuntimeError("word padding needs the words filler")
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
            words = [rng.choice(pool.words) for _ in range(max(num_words, 0))]
            padded = text + "\n" + pool.join(words)
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
