# SPDX-License-Identifier: Apache-2.0
"""Tests for the artifact store: atomicity, absence, and the null case."""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.persistence.store import (
    ArtifactNotFoundError,
    LocalArtifactStore,
    NullArtifactStore,
)


class TestLocalArtifactStore:
    def test_a_written_artifact_reads_back(self, tmp_path: Path):
        store = LocalArtifactStore(tmp_path / "nested" / "artifact")

        with store.open_write() as stream:
            stream.write(b"payload")
        with store.open_read() as stream:
            assert stream.read() == b"payload"

    def test_reading_nothing_says_so(self, tmp_path: Path):
        """Absence is normal on a first boot, so it is its own error
        rather than an OSError the caller has to classify."""
        with pytest.raises(ArtifactNotFoundError):
            with LocalArtifactStore(tmp_path / "absent").open_read() as _:
                pass

    def test_a_failed_write_leaves_the_previous_artifact(self, tmp_path: Path):
        """The point of writing beside and renaming: a half-written
        checkpoint must never replace a good one."""
        path = tmp_path / "artifact"
        store = LocalArtifactStore(path)
        with store.open_write() as stream:
            stream.write(b"first")

        with pytest.raises(RuntimeError):
            with store.open_write() as stream:
                stream.write(b"second, interrupted")
                raise RuntimeError("boom")

        assert path.read_bytes() == b"first"
        assert list(tmp_path.iterdir()) == [path], "left a temporary file behind"


class TestNullArtifactStore:
    def test_writes_are_discarded_and_reads_find_nothing(self):
        """Persistence being unconfigured is not a special case every
        caller has to test for."""
        store = NullArtifactStore()

        with store.open_write() as stream:
            stream.write(b"ignored")
        with pytest.raises(ArtifactNotFoundError):
            with store.open_read() as _:
                pass
