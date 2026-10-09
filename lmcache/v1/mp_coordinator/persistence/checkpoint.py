# SPDX-License-Identifier: Apache-2.0
"""The coordinator's checkpoint: derived state that outlives the process.

Written on a timer and on a clean stop, read once at startup. A restart
that finds one comes back knowing what the fleet has cached; one that
does not starts blind and relearns only what gets stored again.

The format is a short header and one msgpack document holding every
section. Because a capture is plain data, the codec never learns what a
section means -- adding a component changes nothing here.
"""

# Standard
from collections.abc import Mapping, Sequence
from typing import BinaryIO, cast
import struct

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.persistence.durable_component import DurableComponent
from lmcache.v1.mp_coordinator.persistence.quiesce import (
    DEFAULT_QUIESCE_TIMEOUT,
    QuiesceLock,
)
from lmcache.v1.mp_coordinator.persistence.store import (
    ArtifactNotFoundError,
    ArtifactStore,
)

logger = init_logger(__name__)

_MAGIC = b"LMCKPT\0\0"
_FORMAT_VERSION = 1
_HEADER = struct.Struct("<8sI")


class CheckpointFormatError(Exception):
    """A checkpoint is malformed, truncated, or of an unknown version."""


def save_checkpoint(
    store: ArtifactStore,
    quiesce: QuiesceLock,
    components: Sequence[DurableComponent],
    timeout: float = DEFAULT_QUIESCE_TIMEOUT,
) -> None:
    """Capture ``components`` and replace the stored checkpoint.

    Ingest is quiesced while the components are read, which is the only
    way the sections agree with each other -- one batch is applied by
    several of them. It is released before the write: a disk write under
    the quiesce would park the fleet's event stream for its duration.
    That is why ``capture`` must return copies -- by the time the
    sections are encoded, the components they came from are moving again.

    Failures are logged rather than raised: a checkpoint is an
    optimization, and a coordinator that dies because it could not write
    one is strictly worse than one that keeps serving.

    Args:
        store: Where the checkpoint lives.
        quiesce: The lock the ingest path holds while applying.
        components: The state to persist.
        timeout: Seconds to wait for an in-flight batch.
    """
    try:
        sections: dict[str, Mapping[str, object]] = {}
        with quiesce.quiesced(timeout):
            for component in components:
                if component.name in sections:
                    raise ValueError(f"two components named {component.name!r}")
                sections[component.name] = component.capture()
        with store.open_write() as stream:
            _write_checkpoint(sections, stream)
    except (OSError, TimeoutError, ValueError) as e:
        logger.warning("Failed to write checkpoint %s: %s", store.location(), e)
        return
    logger.debug("Checkpointed %d section(s) to %s", len(sections), store.location())


def load_checkpoint(
    store: ArtifactStore, components: Sequence[DurableComponent]
) -> None:
    """Restore ``components`` from the stored checkpoint, if there is one.

    Call once at startup, before the gate admits anything. Each component
    loads its own section, so the order here does not matter and a
    component whose section is absent keeps whatever it started with.

    Every failure is survivable: a missing, corrupt, or future-version
    checkpoint logs and leaves the coordinator cold rather than refusing
    to boot.

    Args:
        store: Where the checkpoint lives.
        components: The state to restore.
    """
    try:
        with store.open_read() as stream:
            sections = _read_checkpoint(stream)
    except ArtifactNotFoundError:
        logger.info("No checkpoint at %s; starting cold", store.location())
        return
    except (OSError, CheckpointFormatError) as e:
        logger.warning("Ignoring checkpoint %s: %s", store.location(), e)
        return

    restored = []
    for component in components:
        section = sections.get(component.name)
        if section is None:
            continue
        try:
            component.restore(section)
        except (ValueError, TypeError, KeyError) as e:
            # One unreadable section must not cost the others.
            logger.warning("Ignoring %r section: %s", component.name, e)
            continue
        restored.append(component.name)
    logger.info(
        "Restored %s from checkpoint %s",
        ", ".join(restored) or "nothing",
        store.location(),
    )


# -- Internals ----------------------------------------------------------------


def _write_checkpoint(
    sections: Mapping[str, Mapping[str, object]], stream: BinaryIO
) -> None:
    """Encode ``sections`` and write them to ``stream``.

    Args:
        sections: Captures keyed by component name.
        stream: Where to write; positioned at the end on return.

    Raises:
        OSError: If the stream rejects a write.
    """
    payload = msgspec.msgpack.encode(sections)
    stream.write(_HEADER.pack(_MAGIC, _FORMAT_VERSION))
    stream.write(payload)


def _read_checkpoint(stream: BinaryIO) -> dict[str, Mapping[str, object]]:
    """Decode a checkpoint written by :func:`_write_checkpoint`.

    Args:
        stream: Positioned at the start of the checkpoint.

    Returns:
        Sections keyed by component name.

    Raises:
        CheckpointFormatError: If the stream is not a checkpoint, carries
            an unsupported version, or is malformed.
    """
    header = stream.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise CheckpointFormatError(f"truncated header ({len(header)} bytes)")
    magic, version = _HEADER.unpack(header)
    if magic != _MAGIC:
        raise CheckpointFormatError(f"not a checkpoint (magic {magic!r})")
    if version != _FORMAT_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint version {version} "
            f"(this build reads {_FORMAT_VERSION})"
        )
    try:
        decoded = msgspec.msgpack.decode(stream.read())
    except msgspec.MsgspecError as e:
        raise CheckpointFormatError(f"malformed checkpoint: {e}") from e
    if not isinstance(decoded, dict):
        raise CheckpointFormatError(f"checkpoint holds a {type(decoded).__name__}")
    return cast("dict[str, Mapping[str, object]]", decoded)
