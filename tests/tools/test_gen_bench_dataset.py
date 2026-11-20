# SPDX-License-Identifier: Apache-2.0
"""Public token generation and dataset conversion contracts."""

# Standard
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from lmcache.tools.cache_simulator.gen_bench_dataset import (
    gen_bench_dataset,
    hash_to_tokens,
)

CHUNK_A = "0x0000000000000946"
CHUNK_B = "0x0000000000005c31"


class ReversibleTokenizer:
    """Synthetic codec isolating prefix identity from tokenizer boundary merges."""

    vocab_size = 32

    def decode(self, token_ids: list[int], **kwargs: bool) -> str:
        """Decode each ID to a distinct space-prefixed fixture token."""
        return "".join(f" token{token_id}" for token_id in token_ids)

    def encode(self, text: str, **kwargs: bool) -> list[int]:
        """Recover IDs from the fixture's space-separated tokens."""
        return [int(token.removeprefix("token")) for token in text.split()]


@pytest.mark.parametrize("n_tokens", [16, 256])
def test_distinct_chunks_do_not_collapse(n_tokens: int) -> None:
    """This distinct-hash pair must not become the same synthetic prefix."""
    vocab_ids = list(range(32))

    assert hash_to_tokens(CHUNK_A, n_tokens, vocab_ids) != hash_to_tokens(
        CHUNK_B, n_tokens, vocab_ids
    )


@pytest.mark.parametrize("n_tokens", [1, 16, 256])
def test_generation_is_deterministic_across_calls(n_tokens: int) -> None:
    """Other hashes and requested lengths do not change a hash's token prefix."""
    vocab_ids = list(range(32))
    tokens = hash_to_tokens(CHUNK_A, n_tokens, vocab_ids)

    hash_to_tokens(CHUNK_B, n_tokens, vocab_ids)
    longer_tokens = hash_to_tokens(CHUNK_A, n_tokens + 1, vocab_ids)

    assert len(tokens) == n_tokens
    assert set(tokens) <= set(vocab_ids)
    assert hash_to_tokens(CHUNK_A, n_tokens, vocab_ids) == tokens
    assert longer_tokens[:n_tokens] == tokens


def test_dataset_preserves_distinct_and_repeated_prefixes(tmp_path: Path) -> None:
    """Generated A/B/A records retain exactly that prefix-sharing pattern."""
    events = [
        {"chunk_hashes": [chunk_hash], "chunk_size": 256, "seq_len": 256}
        for chunk_hash in (CHUNK_A, CHUNK_B, CHUNK_A)
    ]
    tokenizer = ReversibleTokenizer()
    output_path = tmp_path / "dataset.jsonl"

    count = gen_bench_dataset(events, tokenizer, 128, output_path)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert count == len(records) == 3
    assert records[0]["prompt"] != records[1]["prompt"]
    assert records[0]["prompt"] == records[2]["prompt"]
    assert all(record["output_tokens"] == 128 for record in records)
    assert all(len(tokenizer.encode(record["prompt"])) == 256 for record in records)
