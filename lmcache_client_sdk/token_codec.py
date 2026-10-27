# SPDX-License-Identifier: Apache-2.0
"""Packed binary representation of a token id sequence.

The wire layout mirrors ``lmcache.v1.multiprocess.token_codec``: big-endian
``uint32`` words, one word per token. The client sends the packed bytes to the
server, which computes chunk hashes; the packing format is a frozen protocol
contract and must not change.
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
