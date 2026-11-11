# SPDX-License-Identifier: Apache-2.0
"""Startup scan of an fs_native cache directory (no native extension required).

After a restart, the files a previous server wrote are counted toward usage and
handed to the eviction policy. These tests cover the scan that finds them.
"""

# Standard
import os

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import encode_fs_filename
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    _scan_persisted_entries,
)


def _key(chunk_id: int, model_name: str = "org/model", salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=3,
        cache_salt=salt,
    )


def _write(base, key: ObjectKey, size: int, mtime_s: int) -> None:
    path = os.path.join(base, encode_fs_filename(key))
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    os.utime(path, ns=(mtime_s * 10**9, mtime_s * 10**9))


class TestScanPersistedEntries:
    def test_finds_cache_files_oldest_first(self, tmp_path):
        newest, oldest, middle = _key(1), _key(2, salt="user-a"), _key(3)
        _write(tmp_path, newest, 300, mtime_s=3_000)
        _write(tmp_path, oldest, 100, mtime_s=1_000)
        _write(tmp_path, middle, 200, mtime_s=2_000)

        entries, skipped = _scan_persisted_entries(str(tmp_path))

        assert entries == [(oldest, 100), (middle, 200), (newest, 300)]
        assert skipped == 0

    def test_ignores_the_tmp_dir_and_other_files(self, tmp_path):
        kept = _key(1)
        _write(tmp_path, kept, 64, mtime_s=1_000)
        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()
        _write(tmp_dir, _key(2), 64, mtime_s=1_000)
        (tmp_path / "notes.txt").write_text("not a cache file")
        (tmp_path / "subdir.data").mkdir()

        entries, skipped = _scan_persisted_entries(str(tmp_path))

        assert entries == [(kept, 64)]
        assert skipped == 0

    def test_counts_names_that_do_not_decode(self, tmp_path):
        (tmp_path / "garbage.data").write_bytes(b"x")

        entries, skipped = _scan_persisted_entries(str(tmp_path))

        assert entries == []
        assert skipped == 1

    def test_missing_directory_recovers_nothing(self, tmp_path):
        entries, skipped = _scan_persisted_entries(str(tmp_path / "absent"))

        assert entries == []
        assert skipped == 0
