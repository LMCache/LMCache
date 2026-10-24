# SPDX-License-Identifier: Apache-2.0
"""Payloads of the ``l1_residency.query`` server-module method.

Residency here means committed residency: an object counts once its write
finished and it is not temporary, as in
:class:`~lmcache.v1.distributed.internal_api.L1CommittedUsage`.
"""

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.token_hasher import TokenHasher

L1_RESIDENCY_QUERY_METHOD = "l1_residency.query"

_PROBE_TOKENS = [1, 2, 3, 4]


def compute_probe_hash(hasher: TokenHasher) -> bytes:
    """Hash the fixed probe tokens as the first chunk of a sequence.

    The hash goes through the same entry points as a real chunk, starting
    from the hasher's ``none_hash``, so two hashers agree on it only when
    both the hash function and its chain seed agree.

    Args:
        hasher: The hasher whose chunk hashes are being checked.

    Returns:
        The probe chunk's hash, as an ``ObjectKey.chunk_hash``.
    """
    return hasher.hash_to_bytes(hasher.hash_tokens(_PROBE_TOKENS))


class L1ResidencyQuery(msgspec.Struct):
    """Ask how much committed L1 the given chunks and their model hold.

    The server expands every chunk into its exact object keys: one per kv
    rank of ``world_size``, in object group 0, salted with the dict key, as
    a lookup of the same model and world size would.

    Args:
        model_name: The model's ``ObjectKey.model_name``.
        world_size: The kv world size whose ranks the keys fan out to.
        chunk_hashes_by_salt: Per cache salt, distinct ``ObjectKey.chunk_hash``
            values to count.
    """

    model_name: str
    world_size: int
    chunk_hashes_by_salt: dict[str, list[bytes]]


class L1ResidencyReply(msgspec.Struct):
    """Committed L1 usage of the queried chunks and of their whole model.

    The usage fields mirror
    :class:`~lmcache.v1.distributed.internal_api.L1CommittedUsage`.

    Args:
        probe_hash: The server's :func:`compute_probe_hash`. A client whose
            own probe hash differs hashes chunks differently and matches
            nothing.
        matched_objects: Committed objects among the queried chunks' keys.
        matched_bytes: Their bytes.
        model_objects: Committed objects of the queried model.
        model_bytes: Their bytes.
    """

    probe_hash: bytes
    matched_objects: int
    matched_bytes: int
    model_objects: int
    model_bytes: int
