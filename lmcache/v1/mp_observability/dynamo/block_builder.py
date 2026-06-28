# SPDX-License-Identifier: Apache-2.0

"""Pure block-building logic for Dynamo KV cache events.

Slices a token sequence into full blocks and chain-hashes each block (via
:class:`~lmcache.v1.multiprocess.token_hasher.TokenHasher`), producing the
``(parent_hash, blocks)`` shape used to build Dynamo ``BlockStored`` events.
The chain is carried in the hasher's *native* form (``bytes`` for blake3,
``int`` for sha256_cbor/builtin) to avoid losing entropy; only the per-block
value returned to the caller is reduced to Dynamo's signed-i64 block hash.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.token_hasher import TokenHasher

# A hasher's native hash value: ``bytes`` (blake3) or ``int`` (sha256_cbor /
# builtin). Opaque; only ever fed back into the hasher's own methods.
NativeHash = object


def _to_signed_i64(native_hash: NativeHash, hasher: TokenHasher) -> int:
    """Reduce a native hash to a signed 64-bit integer.

    Takes the leading 8 bytes of ``hasher.hash_to_bytes(native_hash)`` so the
    result fits Dynamo's ``i64`` regardless of digest width (blake3 yields 32
    bytes; sha256_cbor / builtin already yield 8).

    Args:
        native_hash: A hash value in the hasher's native form.
        hasher: The hasher that produced ``native_hash``.

    Returns:
        The hash reduced to a signed 64-bit integer.
    """
    raw = hasher.hash_to_bytes(native_hash)
    return int.from_bytes(raw[:8], byteorder="big", signed=True)


def build_blocks(
    token_ids: list[int],
    kv_block_size: int,
    prefix_hash: NativeHash,
    hasher: TokenHasher,
) -> tuple[int | None, list[tuple[list[int], int]]]:
    """Slice ``token_ids`` into full blocks and chain-hash each block.

    Trailing tokens that do not fill a block are discarded. Block 0 is hashed
    against ``prefix_hash``, each later block against the previous block's
    native hash, forming the same prefix chain the storage path computes -- so
    two calls sharing a token prefix produce identical hashes for the shared
    blocks.

    Args:
        token_ids: Token sequence to slice. Only full blocks are used.
        kv_block_size: Tokens per block. Must be positive.
        prefix_hash: Native hash of the block preceding ``token_ids``; pass
            ``hasher.none_hash`` at the sequence start.
        hasher: Hasher used for all hashing.

    Returns:
        ``(parent_hash, blocks)``. ``parent_hash`` is the signed-i64 form of
        ``prefix_hash``, or ``None`` at the sequence start. ``blocks`` is an
        ordered list of ``(block_token_ids, block_hash_i64)``, one per block.

    Raises:
        ValueError: If ``kv_block_size`` is not positive.
    """
    if kv_block_size <= 0:
        raise ValueError(f"kv_block_size must be positive (got {kv_block_size})")

    if prefix_hash == hasher.none_hash:
        parent_hash: int | None = None
    else:
        parent_hash = _to_signed_i64(prefix_hash, hasher)

    blocks: list[tuple[list[int], int]] = []
    prev_native = prefix_hash
    num_full_blocks = len(token_ids) // kv_block_size
    for i in range(num_full_blocks):
        block_tokens = token_ids[i * kv_block_size : (i + 1) * kv_block_size]
        block_native = hasher.hash_tokens(block_tokens, prev_native)
        blocks.append((block_tokens, _to_signed_i64(block_native, hasher)))
        prev_native = block_native

    return parent_hash, blocks
