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
from dataclasses import dataclass, field, replace
from typing import BinaryIO, Literal, cast
import struct
import threading
import time

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.metric_sources import register_metric_source
from lmcache.v1.mp_coordinator.persistence.durable_component import DurableComponent
from lmcache.v1.mp_coordinator.persistence.quiesce import (
    DEFAULT_QUIESCE_TIMEOUT,
    QuiesceLock,
)
from lmcache.v1.mp_coordinator.persistence.store import (
    ArtifactNotFoundError,
    ArtifactStore,
)
from lmcache.v1.mp_coordinator.utils.histogram import DURATION_SECONDS_BOUNDS, Histogram

logger = init_logger(__name__)

_MAGIC = b"LMCKPT\0\0"
_FORMAT_VERSION = 1
_HEADER = struct.Struct("<8sI")


RestoreOutcome = Literal["cold", "restored", "ignored"]
WriteResult = Literal["ok", "quiesce_timeout", "error"]


class CheckpointFormatError(Exception):
    """A checkpoint is malformed, truncated, or of an unknown version."""


class CheckpointWriteError(Exception):
    """A periodic checkpoint write failed; the message says why."""


@dataclass(frozen=True)
class CheckpointWrite:
    """What one :func:`save_checkpoint` call did.

    Attributes:
        result: ``ok``; ``quiesce_timeout`` when a batch was still being
            applied after the timeout; or ``error`` when capturing or
            writing failed.
        error: The failure, as ``Type: message``; ``None`` when ``ok``.
        pause_ingest_seconds: Time waiting for in-flight batches before
            capturing; ``None`` if the quiesce timed out.
        copy_seconds: Time capturing the components, with ingest paused;
            ``None`` if capturing did not finish.
        write_seconds: Time encoding and writing; ``None`` unless written.
        size_bytes: Bytes written; ``None`` unless written.
    """

    result: WriteResult
    error: str | None = None
    pause_ingest_seconds: float | None = None
    copy_seconds: float | None = None
    write_seconds: float | None = None
    size_bytes: int | None = None


@dataclass
class CheckpointStats:
    """What the checkpoint has done since this process started.

    Attributes:
        writes: Checkpoint writes by result (``ok``, ``quiesce_timeout``,
            ``error``); every result is present, from 0.
        last_write_timestamp: Wall-clock time of the last good write;
            ``None`` before the first.
        size_bytes: Size of the last good write; ``None`` before the first.
        write_duration_seconds: Time per write phase: ``pause_ingest``,
            ``copy`` (how long ingest was paused for) and ``write``.
        restored_on_start: What startup did: ``cold`` (no checkpoint),
            ``restored``, or ``ignored`` (unreadable); ``None`` until
            :func:`load_checkpoint` has run.
    """

    writes: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(("ok", "quiesce_timeout", "error"), 0)
    )
    last_write_timestamp: float | None = None
    size_bytes: int | None = None
    write_duration_seconds: dict[str, Histogram] = field(
        default_factory=lambda: {
            phase: Histogram.with_bounds(DURATION_SECONDS_BOUNDS)
            for phase in ("pause_ingest", "copy", "write")
        }
    )
    restored_on_start: RestoreOutcome | None = None


class CheckpointMonitor:
    """Keeps :class:`CheckpointStats` from what saves and the load report.

    Locked: the periodic save and the final one on shutdown run on worker
    threads, while metrics read from another.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats = CheckpointStats()
        register_metric_source(self)

    def record_write(self, checkpoint_write: CheckpointWrite) -> None:
        """Count one :func:`save_checkpoint` call.

        Args:
            checkpoint_write: What it did.
        """
        seconds_per_phase = {
            "pause_ingest": checkpoint_write.pause_ingest_seconds,
            "copy": checkpoint_write.copy_seconds,
            "write": checkpoint_write.write_seconds,
        }
        with self._lock:
            self._stats.writes[checkpoint_write.result] += 1
            for phase, seconds in seconds_per_phase.items():
                if seconds is not None:
                    self._stats.write_duration_seconds[phase].observe(seconds)
            if checkpoint_write.result == "ok":
                self._stats.last_write_timestamp = time.time()
                self._stats.size_bytes = checkpoint_write.size_bytes

    def record_restore(self, outcome: RestoreOutcome) -> None:
        """Record what :func:`load_checkpoint` did at startup.

        Args:
            outcome: Its return value.
        """
        with self._lock:
            self._stats.restored_on_start = outcome

    def stats(self) -> CheckpointStats:
        """Return a copy of :class:`CheckpointStats`."""
        with self._lock:
            return replace(
                self._stats,
                writes=dict(self._stats.writes),
                write_duration_seconds={
                    phase: histogram.copy()
                    for phase, histogram in self._stats.write_duration_seconds.items()
                },
            )


def save_checkpoint(
    store: ArtifactStore,
    quiesce: QuiesceLock,
    components: Sequence[DurableComponent],
    timeout: float = DEFAULT_QUIESCE_TIMEOUT,
) -> CheckpointWrite:
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

    Returns:
        What the call did, including how long each phase took.
    """
    pause_ingest_seconds = copy_seconds = None
    started_at = time.monotonic()
    try:
        sections: dict[str, Mapping[str, object]] = {}
        with quiesce.quiesced(timeout):
            quiesced_at = time.monotonic()
            pause_ingest_seconds = quiesced_at - started_at
            for component in components:
                if component.name in sections:
                    raise ValueError(f"two components named {component.name!r}")
                sections[component.name] = component.capture()
        copied_at = time.monotonic()
        copy_seconds = copied_at - quiesced_at
        with store.open_write() as stream:
            size_bytes = _write_checkpoint(sections, stream)
    except (OSError, ValueError) as e:
        logger.warning("Failed to write checkpoint %s: %s", store.location(), e)
        # TimeoutError is an OSError; only one raised before ingest paused
        # is the quiesce's -- a later one came from the store.
        timed_out_quiescing = (
            isinstance(e, TimeoutError) and pause_ingest_seconds is None
        )
        return CheckpointWrite(
            result="quiesce_timeout" if timed_out_quiescing else "error",
            error=_describe(e),
            pause_ingest_seconds=pause_ingest_seconds,
            copy_seconds=copy_seconds,
        )
    logger.debug("Checkpointed %d section(s) to %s", len(sections), store.location())
    return CheckpointWrite(
        result="ok",
        pause_ingest_seconds=pause_ingest_seconds,
        copy_seconds=copy_seconds,
        write_seconds=time.monotonic() - copied_at,
        size_bytes=size_bytes,
    )


def load_checkpoint(
    store: ArtifactStore, components: Sequence[DurableComponent]
) -> RestoreOutcome:
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

    Returns:
        ``cold`` when there is no checkpoint, ``ignored`` when it could not
        be read, else ``restored`` (even if some sections were skipped).
    """
    try:
        with store.open_read() as stream:
            sections = _read_checkpoint(stream)
    except ArtifactNotFoundError:
        logger.info("No checkpoint at %s; starting cold", store.location())
        return "cold"
    except (OSError, CheckpointFormatError) as e:
        logger.warning("Ignoring checkpoint %s: %s", store.location(), e)
        return "ignored"

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
    return "restored"


# -- Internals ----------------------------------------------------------------


def _write_checkpoint(
    sections: Mapping[str, Mapping[str, object]], stream: BinaryIO
) -> int:
    """Encode ``sections`` and write them to ``stream``.

    Args:
        sections: Captures keyed by component name.
        stream: Where to write; positioned at the end on return.

    Returns:
        The number of bytes written.

    Raises:
        OSError: If the stream rejects a write.
    """
    payload = msgspec.msgpack.encode(sections)
    stream.write(_HEADER.pack(_MAGIC, _FORMAT_VERSION))
    stream.write(payload)
    return _HEADER.size + len(payload)


def _describe(error: Exception) -> str:
    """Return ``error`` as ``Type: message``."""
    return f"{type(error).__name__}: {error}"


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
