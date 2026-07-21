# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple, Union
import hashlib
import string

# Third Party
import torch


def hex_hash_to_int16(s: Optional[str]) -> int:
    """Convert a hash identifier into a 16-bit integer.

    Handles hex strings (optionally prefixed with ``0x``) and falls back
    to a stable SHA-256 hash for arbitrary string identifiers.

    Args:
        s: The multimodal hash identifier string. May be None.

    Returns:
        A 16-bit integer in ``[0, 0xFFFF]``.
    """
    s = "" if s is None else str(s)
    s_stripped = s.strip()

    hex_part = s_stripped[2:] if s_stripped.lower().startswith("0x") else s_stripped
    if hex_part and all(c in string.hexdigits for c in hex_part):
        try:
            return int(hex_part, 16) & 0xFFFF
        except ValueError:
            pass

    digest = hashlib.sha256(s_stripped.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], byteorder="big", signed=False)


def apply_mm_hashes_to_token_ids(
    token_ids: Union[List[int], torch.Tensor],
    mm_hashes: Optional[List[str]],
    mm_positions: Optional[List[Tuple[int, int]]],
) -> Union[List[int], torch.Tensor]:
    """Overwrite token positions with multimodal content hashes.

    Each multimodal placeholder position (specified by ``(offset, length)``
    tuples in ``mm_positions``) is replaced with the corresponding 16-bit
    hash of the multimodal content identifier.

    Args:
        token_ids: Token ID list or tensor. Modified in-place and returned.
        mm_hashes: Multimodal content identifier strings, or None.
        mm_positions: List of ``(offset, length)`` tuples identifying
            placeholder token ranges, or None.

    Returns:
        The modified ``token_ids`` (same type as input).
    """
    if not mm_hashes or not mm_positions:
        return token_ids

    if isinstance(token_ids, torch.Tensor):
        n = token_ids.size(0)
        for hash_str, (start, length) in zip(mm_hashes, mm_positions, strict=False):
            if start >= n:
                continue
            end = min(start + length, n)
            token_ids[start:end] = hex_hash_to_int16(hash_str)
        return token_ids

    n = len(token_ids)
    for hash_str, (start, length) in zip(mm_hashes, mm_positions, strict=False):
        if start >= n:
            continue
        end = min(start + length, n)
        fill_value = hex_hash_to_int16(hash_str)
        for i in range(start, end):
            token_ids[i] = fill_value
    return token_ids
