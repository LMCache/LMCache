# SPDX-License-Identifier: Apache-2.0
"""Tokenizer helpers shared by the engine-benchmark workloads.

Workloads that promise a prompt of *N tokens* need the model's tokenizer to
keep that promise: any text-level proxy (word counts, character counts)
drifts by a per-tokenizer expansion factor, so the same flags produce
different-sized prompts on different models.
"""

# Standard
from dataclasses import dataclass
import random
import re

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# A pool entry is a vocabulary token that starts a new word and continues
# with plain lowercase letters.  Starting a word is what makes concatenation
# exact: such a token is segmented independently of its neighbours, so the
# count does not depend on the order a permutation puts the words in.
# Restricting to ASCII letters keeps entries clear of digits, punctuation
# and byte-fallback fragments, which merge with adjacent text.
_WORD_RE = re.compile(r"^[a-z]+$")

# How each tokenizer family spells "this token begins a word": a literal
# space, GPT-2 byte-level BPE's ``Ġ``, or SentencePiece's ``▁``.  Matching on
# the raw vocabulary key rather than on ``decode()`` output matters —
# SentencePiece decodes ``▁the`` to ``"the"``, dropping the very marker that
# identifies it, so a decode-based filter finds nothing on those models.
_WORD_START_MARKERS = (" ", "Ġ", "▁")

# Stand-in for "some word already precedes this one", used to measure a
# candidate's cost in mid-text position.  Any short word would do.
_PROBE_WORD = "x"


@dataclass(frozen=True)
class TokenPool:
    """Words that each cost one token, plus how to join them.

    The joining convention is part of the pool, not a caller's choice: the
    same words cost a different number of tokens under the wrong one, which
    is the whole failure this module exists to prevent.
    """

    words: list[str]
    leading_space: bool
    """Whether the text must open with a space.

    Most tokenizers fold a leading space into the following word's token, so
    every word — the first included — carries one.  A few (Phi-3) emit a
    standalone token for an explicit leading space; there the words are
    merely separated by spaces and the text starts with a letter.
    """

    def join(self, words: list[str]) -> str:
        """Join ``words`` into text that encodes to ``len(words)`` tokens."""
        text = " ".join(words)
        if text and self.leading_space:
            return " " + text
        return text


def _verify(tokenizer, words: list[str], prefix: str, base: int) -> list[str]:
    """Keep the words that still cost exactly one token after ``prefix``.

    Vocabulary keys are a hint, not a contract — a key that looks like a
    word need not survive the tokenizer's own pre-tokenization, so each
    candidate is encoded as it will appear in a prompt.
    """
    encoded = tokenizer([prefix + w for w in words], add_special_tokens=False)
    return [
        w
        for w, ids_ in zip(words, encoded["input_ids"], strict=False)
        if len(ids_) == base + 1
    ]


def try_load_tokenizer(model_name: str | None):
    """Best-effort load of a model's tokenizer.

    Args:
        model_name: HuggingFace repo ID or local path.  ``None`` short-
            circuits to ``None``.

    Returns:
        The tokenizer, or ``None`` if ``transformers`` isn't installed or
        the tokenizer can't be loaded.  Callers decide whether that is
        fatal — workloads whose length contract depends on the tokenizer
        should fail rather than silently generate mis-sized prompts.
    """
    if model_name is None:
        return None
    try:
        # Third Party
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("transformers is not installed; no tokenizer available")
        return None
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load tokenizer for %s (%s)", model_name, e)
        return None


def build_single_token_pool(
    tokenizer,
    size: int,
    seed: int = 42,
) -> TokenPool:
    """Pick ``size`` words that each cost exactly one token.

    Text built with :meth:`TokenPool.join` tokenizes to exactly one token
    per word, so a workload can hit a requested token count exactly instead
    of approximating it.

    Args:
        tokenizer: A HuggingFace tokenizer.
        size: Number of words to return.
        seed: Random seed; the same seed and tokenizer yield the same pool.

    Returns:
        A :class:`TokenPool` of ``size`` words and the convention for
        joining them.

    Raises:
        ValueError: If the tokenizer yields fewer than ``size`` usable
            words under either convention, or if the resulting text does
            not encode to ``size`` tokens after all.
    """
    special = set(tokenizer.all_special_ids or [])

    seen: set[str] = set()
    for key, token_id in tokenizer.get_vocab().items():
        if token_id in special or not key.startswith(_WORD_START_MARKERS):
            continue
        word = key[1:]
        if _WORD_RE.match(word):
            seen.add(word)
    # ``get_vocab()`` is a plain dict; sort so the pool depends only on the
    # tokenizer and the seed.
    words = sorted(seen)

    verified: list[str] = []
    leading_space = True
    if words:
        verified = _verify(tokenizer, words, " ", 0)
        if len(verified) < size:
            # A tokenizer that charges for an explicit leading space (Phi-3)
            # rejects every candidate above; there the words are separated
            # rather than prefixed.
            base = len(tokenizer.encode(_PROBE_WORD, add_special_tokens=False))
            separated = _verify(tokenizer, words, _PROBE_WORD + " ", base)
            if len(separated) > len(verified):
                verified, leading_space = separated, False

    if len(verified) < size:
        raise ValueError(
            f"Tokenizer yielded only {len(verified)} words that stay a "
            f"single token in context, need {size}. Use a smaller "
            f"vocab_size, or a model whose tokenizer marks word starts "
            f"(byte-level BPE or SentencePiece do; WordPiece does not)."
        )

    pool = TokenPool(random.Random(seed).sample(verified, size), leading_space)

    # What is promised is a total, and per-word costs are not obliged to
    # compose.  Check the total rather than trusting that they do.
    actual = len(tokenizer.encode(pool.join(pool.words), add_special_tokens=False))
    if actual != size:
        raise ValueError(
            f"Built a {size}-word pool but its text encodes to {actual} "
            f"tokens; this tokenizer does not segment words independently, "
            f"so exact token lengths cannot be honoured for it."
        )

    logger.debug(
        "Single-token pool: %d candidates, %d verified, sampling %d (leading_space=%s)",
        len(words),
        len(verified),
        size,
        leading_space,
    )
    return pool
