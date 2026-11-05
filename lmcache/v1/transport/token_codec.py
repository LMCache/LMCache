# SPDX-License-Identifier: Apache-2.0
"""Packed binary representation of a token id sequence.

The layout is big-endian ``uint32``, one word per token. This is the exact byte
sequence used by the chunk-hash contract, so changing it would rekey caches.
"""

# Standard
from collections.abc import Sequence
import array
import struct
import sys

TOKEN_STRIDE = 4
"""Bytes per packed token id (big-endian ``uint32``)."""

_NATIVE_U32 = array.array("I").itemsize == TOKEN_STRIDE
_SWAP = sys.byteorder == "little"


def pack_token_ids(token_ids: Sequence[int]) -> bytes:
    """Pack token ids into the big-endian ``uint32`` wire buffer.

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
    """Return the number of whole token ids in a packed wire buffer."""
    return len(packed) // TOKEN_STRIDE
