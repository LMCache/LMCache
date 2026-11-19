# SPDX-License-Identifier: Apache-2.0
"""Packed binary representation of a token id sequence.

Token ids cross three process boundaries on every scheduler step, and as a
``list[int]`` each hop materializes one Python ``int`` per token. Packed,
each hop is a memcpy.

The layout is big-endian ``uint32``, one word per token: the exact byte
string ``_make_blake3_hash_func`` already fed to blake3, so chunk hashes are
identical either way and no cached entry is invalidated.
"""

# Standard
from collections.abc import Sequence
import array
import struct
import sys

TOKEN_STRIDE = 4
"""Bytes per packed token id (big-endian ``uint32``)."""

_NATIVE_U32 = array.array("I").itemsize == TOKEN_STRIDE
"""Whether ``array('I')`` is a 4-byte word here, enabling the fast path."""

_SWAP = sys.byteorder == "little"
"""Whether native words need byte-swapping to reach big-endian order."""


def pack_token_ids(token_ids: Sequence[int]) -> bytes:
    """Pack token ids into the big-endian ``uint32`` wire buffer.

    Args:
        token_ids: The token ids to pack. Each must fit in a ``uint32``,
            which every real vocabulary does.

    Returns:
        ``TOKEN_STRIDE * len(token_ids)`` bytes.

    Raises:
        OverflowError: If a token id does not fit in a ``uint32``.
    """
    if not _NATIVE_U32:
        return struct.pack(f">{len(token_ids)}I", *token_ids)
    # ~2x faster than struct.pack for long sequences: one bulk conversion
    # instead of unpacking the whole sequence as varargs.
    words = array.array("I", token_ids)
    if _SWAP:
        words.byteswap()
    return words.tobytes()


def unpack_token_ids(packed: bytes) -> list[int]:
    """Unpack a wire buffer back into token ids.

    Only callers that genuinely need Python ints should do this; hashing
    works directly on the packed form.

    Args:
        packed: A buffer whose length is a multiple of ``TOKEN_STRIDE``.

    Returns:
        The token ids it holds.

    Raises:
        ValueError: If ``packed`` is not a whole number of tokens.
    """
    if len(packed) % TOKEN_STRIDE:
        raise ValueError(
            f"packed token buffer of {len(packed)} byte(s) is not a multiple "
            f"of {TOKEN_STRIDE}"
        )
    if not _NATIVE_U32:
        return list(struct.unpack(f">{num_packed_tokens(packed)}I", packed))
    words = array.array("I")
    words.frombytes(packed)
    if _SWAP:
        words.byteswap()
    return words.tolist()


def num_packed_tokens(packed: bytes) -> int:
    """Count the token ids in a wire buffer.

    Args:
        packed: A packed token buffer.

    Returns:
        How many whole token ids it holds; a ragged tail is not counted.
    """
    return len(packed) // TOKEN_STRIDE
