# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible import surface for the packed token codec.

The codec is shared by multiprocessing and other LMCache components, so its
implementation lives in :mod:`lmcache.v1.common.token_codec`.  Keep this
module as a forwarding shim for external integrations that still import the
former multiprocessing path.
"""

# First Party
from lmcache.v1.common.token_codec import (
    TOKEN_STRIDE,
    num_packed_tokens,
    pack_token_ids,
    unpack_token_ids,
)

__all__ = [
    "TOKEN_STRIDE",
    "num_packed_tokens",
    "pack_token_ids",
    "unpack_token_ids",
]
