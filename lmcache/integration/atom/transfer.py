# SPDX-License-Identifier: Apache-2.0
"""Convert ATOM's transfer request into LMCache's transfer spec.

ATOM describes a transfer by a prefix chunk-hash chain plus the engine block
ids of the transferred range; LMCache submits it as
:class:`AtomMPTransferSpec`. The request shape mirrors ``TransferRequest`` in
the external-connector design (v2): the scheduler computes the chain once, so
``hashes`` covers ``[0, end)`` and ``start`` only selects which block ids the
range uses. LMCache's chunk hashes are chained, so a transfer that begins
mid-prefix still needs the hashes before its first chunk.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# Local
from .multi_process_adapter import AtomMPTransferSpec

_CHUNK_HASH_BYTES = 32
_SALT_FORBIDDEN = "@/\\\x00"


@dataclass(frozen=True)
class AtomTransferRequest:
    """One ATOM transfer, as the external-connector API defines it.

    Attributes:
        hashes: Concatenated chunk hashes for the prefix ``[0, end)``, 32 bytes
            each. ``start`` selects the block ids of the range, not a slice of
            this chain: LMCache's chunk hashes are chained from the first
            token.
        block_ids: Engine block ids for the range; one list per engine group,
            or a flat list when the deployment has a single group.
        start: First token of the transferred range, inclusive.
        end: Last token of the transferred range, exclusive.
    """

    hashes: bytes
    block_ids: Sequence[int] | Sequence[Sequence[int]]
    start: int = 0
    end: int = 0
    cache_salt: str = ""


def _field(request: Any, name: str) -> Any:
    """Read one request field, failing loudly when it is absent."""
    try:
        return getattr(request, name)
    except AttributeError as exc:
        raise ValueError(
            f"ATOM transfer request {request!r} has no {name!r}; expected "
            "hashes, block_ids, start and end"
        ) from exc


def _optional_field(request: Any, name: str, default: Any) -> Any:
    """Read an optional request field, falling back to the API's default."""
    return getattr(request, name, default)


def _chunk_hashes(hashes: bytes) -> list[bytes]:
    """Split the request's hash blob into one 32-byte hash per chunk."""
    if len(hashes) % _CHUNK_HASH_BYTES:
        raise ValueError(
            f"ATOM transfer carries {len(hashes)} hash bytes, which is not a "
            f"multiple of {_CHUNK_HASH_BYTES}"
        )
    return [
        hashes[offset : offset + _CHUNK_HASH_BYTES]
        for offset in range(0, len(hashes), _CHUNK_HASH_BYTES)
    ]


def _block_id_lists(request: Any, *, num_engine_groups: int | None) -> list[list[int]]:
    """Normalise the block-id field: flat for one group, a list per group."""
    raw = _field(request, "block_ids")
    if isinstance(raw, (str, bytes)):
        raise ValueError("ATOM block_ids must be block ids, not a string")
    groups: list[list[int]] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, int):
            groups = [[int(block_id) for block_id in raw]]  # type: ignore[union-attr]
            break
        if isinstance(entry, (str, bytes)):
            raise ValueError(f"ATOM block_ids[{index}] must be block ids, not a string")
        groups.append([int(block_id) for block_id in entry])
    if num_engine_groups is not None and len(groups) != num_engine_groups:
        raise ValueError(
            f"ATOM transfer names {len(groups)} block-id list(s) but "
            f"{num_engine_groups} engine group(s) were registered"
        )
    return groups


def _cache_salt(request: Any) -> str:
    """Read and validate the tenant salt a request carries.

    The salt is not part of the chunk hashes: LMCache hashes cover the tokens,
    and the salt is a separate identity field, so the converter has to forward
    it or two tenants share objects.
    """
    salt = str(_optional_field(request, "cache_salt", "") or "")
    if any(char in salt for char in _SALT_FORBIDDEN):
        raise ValueError(f"ATOM cache_salt must not contain any of {_SALT_FORBIDDEN!r}")
    return salt


def transfer_spec_from_atom(
    request: Any,
    *,
    num_engine_groups: int | None = None,
    chunk_tokens: int | None = None,
) -> AtomMPTransferSpec:
    """Build an LMCache transfer spec from an ATOM transfer request.

    Args:
        request: An ATOM ``TransferRequest`` (or :class:`AtomTransferRequest`)
            with ``hashes``, ``block_ids``, ``start``, ``end`` and optionally
        num_engine_groups: Number of registered engine groups, when the caller
            knows it. Passing it turns a missing or extra block-id list into an
            error here rather than inside the adapter.
        chunk_tokens: The server's chunk size in tokens, when the caller knows
            it. The hash cover ``[0, end)``, so a request must carry
            ``end // chunk_tokens`` hashes; the server rejects that mismatch
            too, and checking it here fails the request before submission.

    Returns:
        The :class:`AtomMPTransferSpec` LMCache's adapter submits. ``token_ids``
        stays empty: the external API carries chunk hashes, and the backend keys
        a transfer by them instead of re-hashing tokens.

    Raises:
        ValueError: If the hash blob is not a whole number of chunk hashes, the
            token range or salt is invalid, the block-id lists do not match the
            engine group count, or a non-empty range names no blocks.
    """
    hashes = _field(request, "hashes")
    if not isinstance(hashes, (bytes, bytearray)):
        raise ValueError("ATOM transfer hashes must be bytes")
    chunk_hashes = _chunk_hashes(bytes(hashes))

    start = int(_field(request, "start"))
    end = int(_field(request, "end"))
    if start < 0:
        raise ValueError(f"transfer start must be >= 0, got {start}")
    if start > end:
        raise ValueError(f"transfer start {start} is past its end {end}")

    if chunk_tokens is not None:
        if chunk_tokens <= 0:
            raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
        expected = end // chunk_tokens
        if len(chunk_hashes) != expected:
            raise ValueError(
                f"transfer ends at token {end} and carries {len(chunk_hashes)} "
                f"chunk hash(es), but {chunk_tokens}-token chunks cover "
                f"{expected}"
            )

    block_ids = _block_id_lists(request, num_engine_groups=num_engine_groups)
    if end > start:
        if not chunk_hashes:
            raise ValueError(
                f"transfer covers tokens [{start}, {end}) but carries no chunk hashes"
            )
        if not block_ids:
            raise ValueError(
                f"transfer covers tokens [{start}, {end}) but names no block-id "
                "list; a flat list is one group, an empty list names none"
            )
        for index, blocks in enumerate(block_ids):
            if not blocks:
                raise ValueError(
                    f"transfer covers tokens [{start}, {end}) but engine group "
                    f"{index} names no blocks"
                )

    return AtomMPTransferSpec(
        token_ids=[],
        block_ids=block_ids,
        start=start,
        end=end,
        cache_salt=_cache_salt(request),
        chunk_hashes=b"".join(chunk_hashes),
    )


__all__ = ["AtomTransferRequest", "transfer_spec_from_atom"]
