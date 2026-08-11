# SPDX-License-Identifier: Apache-2.0
"""Where a durable artifact's bytes live, and who can produce them.

One storage contract serves both artifacts: each is a single whole object,
replaced atomically, at one location. The seam is a byte stream rather
than a value, because the directory snapshot runs to gigabytes and neither
side should hold two copies — the metadata document layers its JSON on top
of the same primitive.

See ``docs/design/v1/mp_coordinator/key_directory.md``.
"""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import BinaryIO, Protocol
import io
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import PersistenceType

logger = init_logger(__name__)

_TMP_SUFFIX = ".tmp"


class ArtifactNotFoundError(Exception):
    """Nothing has been stored at this location yet."""


class ArtifactStore(ABC):
    """Storage for one whole artifact, replaced atomically."""

    @abstractmethod
    def open_read(self) -> AbstractContextManager[BinaryIO]:
        """Open the stored artifact for reading.

        Returns:
            A context manager yielding the artifact's byte stream.

        Raises:
            ArtifactNotFoundError: If nothing has been written yet.
            OSError: If the backend is unreachable or unreadable.
        """
        raise NotImplementedError

    @abstractmethod
    def open_write(self) -> AbstractContextManager[BinaryIO]:
        """Open a stream that replaces the stored artifact.

        The replacement happens on clean exit: a reader sees either the
        previous artifact or the new one, never a partial write. Leaving
        the context via an exception discards whatever was written.

        Returns:
            A context manager yielding the byte stream to write into.

        Raises:
            OSError: If the backend rejects the write.
        """
        raise NotImplementedError

    @abstractmethod
    def location(self) -> str:
        """Return a human-readable location, for log messages."""
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    """A file on the coordinator's own filesystem.

    Writes land beside the target and are renamed into place, so a crash
    mid-write leaves at most a stale temporary file. On Kubernetes this
    needs a volume outliving the pod.

    Args:
        path: The artifact file; its parent is created on write.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @contextmanager
    def open_read(self) -> Iterator[BinaryIO]:
        """Open the file. See :meth:`ArtifactStore.open_read`."""
        try:
            stream = self._path.open("rb")
        except FileNotFoundError as e:
            raise ArtifactNotFoundError(f"nothing stored at {self._path}") from e
        with stream:
            yield stream

    @contextmanager
    def open_write(self) -> Iterator[BinaryIO]:
        """Write beside the target, then rename it into place. See
        :meth:`ArtifactStore.open_write`."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + _TMP_SUFFIX)
        try:
            with tmp.open("wb") as stream:
                yield stream
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, self._path)

    def location(self) -> str:
        """Return the file path."""
        return str(self._path)


class NullArtifactStore(ArtifactStore):
    """Store for an artifact with no location configured.

    Reads find nothing and writes go nowhere, so the persistence path
    needs no "is it enabled" branch.
    """

    @contextmanager
    def open_read(self) -> Iterator[BinaryIO]:
        """Raise, as nothing is ever stored."""
        raise ArtifactNotFoundError("persistence is not configured")
        yield  # pragma: no cover - unreachable, keeps this a generator

    @contextmanager
    def open_write(self) -> Iterator[BinaryIO]:
        """Yield a stream whose contents are discarded."""
        with io.BytesIO() as stream:
            yield stream

    def location(self) -> str:
        """Return a placeholder location."""
        return "(not configured)"


class DurableComponent(Protocol):
    """Coordinator state that outlives the process by serializing itself.

    Owners advertise their components (see
    ``FleetEvictionController.get_durable_components``) and the
    ``persistence_type`` routes each one, so neither persister needs to know
    what any section means.
    """

    @property
    def name(self) -> str:
        """Name of this component's section in its artifact."""
        ...

    @property
    def persistence_type(self) -> PersistenceType:
        """Which artifact this component's state rides in."""
        ...

    def capture(self) -> Mapping[str, object]:
        """Return the current state in the form the document holds."""
        ...

    def restore(self, state: Mapping[str, object]) -> None:
        """Replace the current state with a captured one.

        Args:
            state: A :meth:`capture` value, as decoded from the
                document; implementations know their own shape.
        """
        ...
