# SPDX-License-Identifier: Apache-2.0
"""Where a durable artifact's bytes live.

One contract serves both artifacts: each is a single whole object,
replaced atomically, at one location. The seam is a byte stream rather
than a value, because a checkpoint runs to gigabytes and neither side
should hold two copies of it.
"""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import BinaryIO
import io
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_TMP_SUFFIX = ".tmp"


class ArtifactNotFoundError(Exception):
    """Nothing is stored at the artifact's location."""


class ArtifactStore(ABC):
    """A single named artifact, read and replaced whole."""

    @abstractmethod
    def open_read(self) -> AbstractContextManager[BinaryIO]:
        """Open the stored artifact for reading.

        Returns:
            A context manager yielding the artifact's bytes.

        Raises:
            ArtifactNotFoundError: If nothing is stored yet.
            OSError: If the artifact cannot be opened.
        """

    @abstractmethod
    def open_write(self) -> AbstractContextManager[BinaryIO]:
        """Open a stream that replaces the artifact on clean exit.

        A reader sees either the previous artifact or the new one, never
        a partial write, and an exception inside the block leaves the
        previous one in place.

        Returns:
            A context manager yielding the stream to write.

        Raises:
            OSError: If the artifact cannot be written.
        """

    @abstractmethod
    def location(self) -> str:
        """Where this artifact lives, for logs and errors."""


class LocalArtifactStore(ArtifactStore):
    """An artifact in a local file.

    Writes land beside the target and are renamed over it, so a crash
    mid-write leaves at most a stale temporary file. On Kubernetes the
    path is typically a PersistentVolumeClaim.
    """

    def __init__(self, path: Path) -> None:
        """Args:
        path: The file to read and replace.
        """
        self._path = path

    @contextmanager
    def open_read(self) -> Iterator[BinaryIO]:
        """See :meth:`ArtifactStore.open_read`."""
        try:
            with self._path.open("rb") as stream:
                yield stream
        except FileNotFoundError as e:
            raise ArtifactNotFoundError(str(self._path)) from e

    @contextmanager
    def open_write(self) -> Iterator[BinaryIO]:
        """See :meth:`ArtifactStore.open_write`."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + _TMP_SUFFIX)
        try:
            with tmp.open("wb") as stream:
                yield stream
                stream.flush()
                # Durable before the rename, or a crash can publish a
                # name pointing at unwritten blocks.
                os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, self._path)

    def location(self) -> str:
        """See :meth:`ArtifactStore.location`."""
        return str(self._path)


class NullArtifactStore(ArtifactStore):
    """The store for an artifact nobody configured.

    Reads report nothing stored and writes are discarded, so persistence
    being off is not a special case every caller has to test for.
    """

    @contextmanager
    def open_read(self) -> Iterator[BinaryIO]:
        """See :meth:`ArtifactStore.open_read`; always raises."""
        raise ArtifactNotFoundError(self.location())
        yield  # pragma: no cover - unreachable, keeps this a generator

    @contextmanager
    def open_write(self) -> Iterator[BinaryIO]:
        """See :meth:`ArtifactStore.open_write`; discards what is written."""
        with io.BytesIO() as sink:
            yield sink

    def location(self) -> str:
        """See :meth:`ArtifactStore.location`."""
        return "<not configured>"
