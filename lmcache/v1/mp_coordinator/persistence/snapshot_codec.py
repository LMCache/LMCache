# SPDX-License-Identifier: Apache-2.0
"""Binary codec for a coordinator checkpoint.

Layout (version 1)::

    magic "LMCKDIR\\0" | u32 version | u32 length | msgpack structure
    raw uint32 token arrays, one per binding, in structure order

Everything but the token arrays goes through msgpack (``msgspec``, as the
trace format does), so the field-level encoding is declared rather than
written out. The arrays stay outside it: they are the bulk of the file, and
writing them raw keeps the encoder from copying a gigabyte to hand back.

The reverse index and component sections name their keys by position in the
key list, so a key referenced from several places is stored once.

See ``docs/design/v1/mp_coordinator/key_directory.md``.
"""

# Standard
from collections.abc import Mapping
from typing import BinaryIO, cast
import struct

# Third Party
import msgspec
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.ingest.event_gate import InstanceStreamStats
from lmcache.v1.mp_coordinator.key_directory import DirectorySnapshot, Placement

logger = init_logger(__name__)

_MAGIC = b"LMCKDIR\0"
_FORMAT_VERSION = 1

# Token ids go to disk as little-endian uint32 regardless of host order,
# so a snapshot moves between machines.
_TOKENS = np.dtype("<u4")

_PREFIX = struct.Struct("<8sII")

_ENCODER = msgspec.msgpack.Encoder()


class SnapshotFormatError(Exception):
    """A snapshot stream is malformed, truncated, or of an unknown version."""


def write_snapshot(
    snapshot: DirectorySnapshot,
    cursors: Mapping[str, InstanceStreamStats],
    sections: Mapping[str, Mapping[str, object]],
    stream: BinaryIO,
) -> None:
    """Encode the directory, the gate's cursors, and component sections.

    Args:
        snapshot: The captured directory state.
        cursors: The ingest gate's per-emitter cursors, from
            :meth:`EventGate.stats`.
        sections: Each registered component's ``capture()``, keyed by
            its ``name``.
        stream: Binary stream to write to; positioned at the end on return.

    Raises:
        OSError: If the stream rejects a write.
    """
    key_indices = {key: index for index, key in enumerate(snapshot.keys)}
    arrays = [token_ids for token_ids, _ in snapshot.bindings.values()]
    structure = _Structure(
        keys=[
            _KeyRecord(key=key, last_access=last_access, placements=list(placements))
            for key, (last_access, placements) in snapshot.keys.items()
        ],
        bindings=[
            _Binding(
                chunk_hash=chunk_hash,
                token_offset=token_offset,
                num_tokens=int(token_ids.size),
            )
            for chunk_hash, (token_ids, token_offset) in snapshot.bindings.items()
        ],
        l1_keys_by_instance={
            instance_id: _indices(key_indices, l1_keys, instance_id)
            for instance_id, l1_keys in snapshot.l1_keys_by_instance.items()
        },
        cursors=dict(cursors),
        sections={key: _ENCODER.encode(payload) for key, payload in sections.items()},
    )
    payload = _ENCODER.encode(structure)
    stream.write(_PREFIX.pack(_MAGIC, _FORMAT_VERSION, len(payload)))
    stream.write(payload)
    for token_ids in arrays:
        stream.write(token_ids.astype(_TOKENS, copy=False).tobytes())


def read_snapshot(
    stream: BinaryIO,
) -> tuple[
    DirectorySnapshot,
    dict[str, InstanceStreamStats],
    dict[str, Mapping[str, object]],
]:
    """Decode a snapshot written by :func:`write_snapshot`.

    Args:
        stream: Binary stream positioned at the start of the snapshot.

    Returns:
        The directory state for :meth:`KeyDirectory.restore`, the cursors
        for :meth:`EventGate.restore_cursors`, and each component's section
        keyed by its ``name``.

    Raises:
        SnapshotFormatError: If the stream is not a snapshot, carries an
            unsupported version, is truncated, or violates an
            :class:`ObjectKey` invariant.
    """
    magic, version, length = _PREFIX.unpack(_read(stream, _PREFIX.size))
    if magic != _MAGIC:
        raise SnapshotFormatError(f"not a directory snapshot (magic {magic!r})")
    if version != _FORMAT_VERSION:
        raise SnapshotFormatError(
            f"unsupported snapshot version {version} "
            f"(this build reads {_FORMAT_VERSION})"
        )
    try:
        structure = msgspec.msgpack.Decoder(_Structure).decode(_read(stream, length))
        key_table = [record.key for record in structure.keys]
        keys = {
            record.key: (record.last_access, tuple(record.placements))
            for record in structure.keys
        }
        bindings = {
            binding.chunk_hash: (
                _read_tokens(stream, binding.num_tokens),
                binding.token_offset,
            )
            for binding in structure.bindings
        }
        l1_keys_by_instance = {
            instance_id: tuple(_key_at(key_table, index, instance_id) for index in idx)
            for instance_id, idx in structure.l1_keys_by_instance.items()
        }
        sections = {
            key: cast("Mapping[str, object]", msgspec.msgpack.decode(payload))
            for key, payload in structure.sections.items()
        }
    except (msgspec.MsgspecError, ValueError) as e:
        # Covers a truncated or non-conforming payload and ObjectKey's own
        # field invariants, which msgspec enforces on decode.
        raise SnapshotFormatError(f"malformed directory snapshot: {e}") from e
    return (
        DirectorySnapshot(
            keys=keys, bindings=bindings, l1_keys_by_instance=l1_keys_by_instance
        ),
        structure.cursors,
        sections,
    )


# -- Internals ----------------------------------------------------------------


class _KeyRecord(msgspec.Struct):
    """One key with its recency and live placements."""

    key: ObjectKey
    last_access: float
    placements: list[Placement]


class _Binding(msgspec.Struct):
    """One chunk's token metadata; its array follows the structure."""

    chunk_hash: bytes
    token_offset: int
    num_tokens: int


class _Structure(msgspec.Struct):
    """Everything in the checkpoint except the token arrays."""

    keys: list[_KeyRecord]
    bindings: list[_Binding]
    l1_keys_by_instance: dict[str, list[int]]
    cursors: dict[str, InstanceStreamStats]
    sections: dict[str, bytes]


def _indices(
    key_indices: Mapping[ObjectKey, int], keys: tuple[ObjectKey, ...], owner: str
) -> list[int]:
    """Resolve ``keys`` to key-list positions, dropping any with no record."""
    resolved = [key_indices[key] for key in keys if key in key_indices]
    if len(resolved) != len(keys):
        # Tracked keys holding no placement; dropping them is what a
        # restore would converge to anyway.
        logger.warning(
            "Snapshot: %s tracks %d unplaced keys; dropping them",
            owner,
            len(keys) - len(resolved),
        )
    return resolved


def _key_at(key_table: list[ObjectKey], index: int, owner: str) -> ObjectKey:
    """Resolve one key-list position.

    Raises:
        SnapshotFormatError: If ``index`` is out of range.
    """
    if index >= len(key_table):
        raise SnapshotFormatError(
            f"{owner!r} references key index {index} of {len(key_table)}"
        )
    return key_table[index]


def _read_tokens(stream: BinaryIO, num_tokens: int) -> np.ndarray:
    """Read one chunk's token array, as written after the structure."""
    raw = _read(stream, num_tokens * _TOKENS.itemsize)
    token_ids = np.frombuffer(raw, dtype=_TOKENS).astype(np.uint32, copy=False)
    token_ids.flags.writeable = False
    return token_ids


def _read(stream: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes.

    Raises:
        SnapshotFormatError: If the stream ends first.
    """
    data = stream.read(size)
    if len(data) != size:
        raise SnapshotFormatError(
            f"truncated snapshot: wanted {size} bytes, got {len(data)}"
        )
    return data
