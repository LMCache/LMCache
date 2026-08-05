# SPDX-License-Identifier: Apache-2.0
"""A tokenizer stand-in for engine-benchmark tests.

Workloads that size their prompts in tokens need a tokenizer.  Loading a
real one would pull in ``transformers`` and hit the network, so these
tests use a stand-in that reproduces the one property the workloads rely
on: GPT-style pre-tokenization, where a leading space binds to the
following word and each ``" word"`` piece is segmented independently of
its neighbours.
"""

# Standard
import re

# Newlines are their own piece; everything else is a run of non-space
# characters plus the single space in front of it.
_PIECE_RE = re.compile(r"\n+| ?[^\s]+")


class FakeTokenizer:
    """Minimal HuggingFace-tokenizer stand-in for offline tests."""

    def __init__(
        self,
        words: list[str],
        extra_tokens: tuple[str, ...] = ("\n\n", " Capitalized", " digit9"),
        split_words: tuple[str, ...] = (),
        marker: str = " ",
        space_is_token: bool = False,
    ) -> None:
        """Build a vocabulary from ``words`` plus some unusable entries.

        Args:
            words: Words that become single ``" word"`` tokens.
            extra_tokens: Literal token texts a pool builder must reject
                (newlines, capitals, digits).
            split_words: Words present in the vocabulary that nonetheless
                re-encode to two tokens, exercising round-trip checks.
            marker: How a vocabulary key spells "this token begins a word".
                ``" "`` models byte-level BPE, whose keys survive
                ``decode()``; ``"▁"`` models SentencePiece, whose keys do
                not — ``decode()`` drops the marker instead of turning it
                back into a space.
            space_is_token: Model a tokenizer (Phi-3) that emits a standalone
                token for an explicit leading space, so prefixing a word with
                one costs two tokens rather than one.
        """
        self._marker = marker
        self._space_is_token = space_is_token
        self._id_to_text: dict[int, str] = {}
        for i, word in enumerate(words):
            self._id_to_text[i] = f" {word}"
        for j, text in enumerate(extra_tokens, start=len(words)):
            self._id_to_text[j] = text
        if space_is_token:
            self._id_to_text[len(self._id_to_text)] = " "
        self._text_to_id = {t: i for i, t in self._id_to_text.items()}
        self._split = {f" {w}" for w in split_words}
        self.all_special_ids: list[int] = []

    def get_vocab(self) -> dict[str, int]:
        return {
            (self._marker + t[1:] if t.startswith(" ") else t): i
            for t, i in self._text_to_id.items()
        }

    def batch_decode(self, ids_list: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in ids_list]

    def decode(self, ids: list[int], **_kwargs) -> str:
        text = "".join(self._id_to_text[i] for i in ids)
        if self._marker != " ":
            # SentencePiece decoding drops the word-start marker rather than
            # rendering it as a space.
            text = text[1:] if text.startswith(" ") else text
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        if self._space_is_token and text.startswith(" "):
            ids.append(self._text_to_id[" "])
            text = text[1:]
        for piece in _PIECE_RE.findall(text):
            if piece in self._split:
                # Two tokens: the vocabulary entry exists but does not
                # survive a decode/encode round trip.
                ids.extend([self._text_to_id[piece]] * 2)
            elif piece in self._text_to_id:
                ids.append(self._text_to_id[piece])
            elif self._space_is_token and f" {piece}" in self._text_to_id:
                # Opening the text with a bare word still makes it
                # word-initial, so it costs the same single token.
                ids.append(self._text_to_id[f" {piece}"])
            else:
                # Unknown text falls back to one token per character.
                ids.extend(range(len(piece)))
        return ids

    def __call__(self, texts: list[str], add_special_tokens: bool = False) -> dict:
        if not texts:
            # Real HuggingFace tokenizers raise here rather than returning an
            # empty batch; callers must not reach this with no candidates.
            raise IndexError("list index out of range")
        return {"input_ids": [self.encode(t) for t in texts]}


def make_fake_tokenizer(num_words: int = 300, **kwargs) -> FakeTokenizer:
    """Build a :class:`FakeTokenizer` with ``num_words`` usable words."""
    words = [f"w{i:04d}word" for i in range(num_words)]
    # Strip digits so the words pass a letters-only pool filter.
    words = [w.translate(str.maketrans("0123456789", "abcdefghij")) for w in words]
    return FakeTokenizer(words, **kwargs)
