#!/usr/bin/env python3
"""Utility script to verify hash stability when skipping leading tokens.

The script demonstrates that computing prefix hashes for the full token
sequence and then skipping the first ``n`` tokens (where ``n`` is a multiple
of the chunk size) produces the same hash values for the remaining chunks as
computing the hashes for the entire sequence.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Iterable, List, Sequence, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lmcache.v1.token_database import ChunkedTokenDatabase


def collect_hashes(
    db: ChunkedTokenDatabase, tokens: Sequence[int]
) -> List[Tuple[int, int, int]]:
    """Collect (start, end, hash) tuples for the given token sequence."""

    return list(db.process_tokens(tokens=tokens, make_key=False))


def filter_after_skip(
    hashes: Iterable[Tuple[int, int, int]], skip_n_tokens: int
) -> List[Tuple[int, int, int]]:
    """Filter hash tuples whose end position is beyond ``skip_n_tokens``."""

    return [entry for entry in hashes if entry[1] > skip_n_tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size used by the ChunkedTokenDatabase (default: 256).",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=4,
        help="Number of chunks to generate for the random token sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for token generation (default: 0).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    db = ChunkedTokenDatabase()
    db.chunk_size = args.chunk_size

    num_tokens = args.chunk_size * args.num_chunks
    tokens = [random.randint(0, 32000) for _ in range(num_tokens)]

    full_hashes = collect_hashes(db, tokens)

    for multiplier in range(args.num_chunks + 1):
        skip_n_tokens = args.chunk_size * multiplier
        filtered_hashes = filter_after_skip(full_hashes, skip_n_tokens)

        truncated_tokens = tokens[skip_n_tokens:]
        truncated_hashes = collect_hashes(db, truncated_tokens)

        # ``filtered_hashes`` should match the hashes we would send to the
        # lookup server after skipping ``skip_n_tokens`` tokens while still
        # computing hashes on the entire prefix.
        # ``truncated_hashes`` shows the hashes when the prefix is removed
        # before hashing, which is expected to differ due to the prefix hash
        # dependency.
        print(f"Skip {skip_n_tokens} tokens:")
        print(f"  Remaining chunks counted via full prefix: {len(filtered_hashes)}")
        print(
            "  Matches hashes computed on truncated prefix?",
            filtered_hashes
            == [(start + skip_n_tokens, end + skip_n_tokens, value)
                for start, end, value in truncated_hashes],
        )
        print(
            "  First hash with truncated prefix differs?",
            (filtered_hashes[0][2] if filtered_hashes else None)
            != (truncated_hashes[0][2] if truncated_hashes else None),
        )
        print()


if __name__ == "__main__":
    main()
