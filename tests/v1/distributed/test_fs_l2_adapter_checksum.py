# SPDX-License-Identifier: Apache-2.0
"""Tests for the FS L2 adapter's opt-in CRC32 integrity check.

With ``checksum="crc32"`` every stored file carries a trailer, and a load
whose payload or trailer does not verify is served as a miss. Loads never
modify files; the key's next store rewrites the file atomically.
``checksum="none"`` (default) keeps the historical raw-bytes format.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
from typing import cast
import os
import time
import zlib

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj

_TRAILER_SIZE = 8


class _RecordingListener:
    def __init__(self) -> None:
        self.stored: list[tuple[ObjectKey, int]] = []
        self.accessed: list[ObjectKey] = []
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.stored.extend(zip(keys, sizes, strict=True))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.extend(keys)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.extend(keys)


class _Buf:
    """Minimal MemoryObj stand-in exposing ``byte_array``."""

    def __init__(self, data: bytes) -> None:
        self.data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self.data)


_KEY = ObjectKey(
    chunk_hash=b"\xde\xad\xbe\xef", model_name="llama", kv_rank=0, cache_salt=""
)


def _make_adapter(
    base_path: Path, checksum: str = "crc32", use_odirect: bool = False
) -> tuple[FSL2Adapter, _RecordingListener]:
    adp = FSL2Adapter(
        FSL2AdapterConfig(
            base_path=str(base_path), checksum=checksum, use_odirect=use_odirect
        )
    )
    listener = _RecordingListener()
    adp.register_listener(listener)  # type: ignore[arg-type]
    return adp, listener


@pytest.fixture
def crc_adapter(tmp_path: Path) -> Iterator[tuple[FSL2Adapter, _RecordingListener]]:
    adp, listener = _make_adapter(tmp_path)
    try:
        yield adp, listener
    finally:
        adp.close()


def _store(adp: FSL2Adapter, payload: bytes) -> None:
    task_id = adp.submit_store_task([_KEY], cast("list[MemoryObj]", [_Buf(payload)]))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = adp.pop_completed_store_tasks()
        if task_id in completed:
            assert completed[task_id].is_successful()
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 5s")


def _load(adp: FSL2Adapter, size: int) -> tuple[bool, bytes]:
    """Load ``_KEY`` into a zeroed buffer; return (hit, loaded bytes)."""
    buf = _Buf(b"\x00" * size)
    task_id = adp.submit_load_task([_KEY], cast("list[MemoryObj]", [buf]))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        bitmap = adp.query_load_result(task_id)
        if bitmap is not None:
            return bitmap.test(0), bytes(buf.data)
        time.sleep(0.01)
    raise AssertionError("load task did not complete within 5s")


def _only_data_file(base_path: Path) -> Path:
    paths = list(base_path.glob("*.data"))
    assert len(paths) == 1, f"expected one .data file, found {paths}"
    return paths[0]


def _flip_byte(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


class TestConfig:
    def test_default_is_none(self, tmp_path: Path) -> None:
        assert FSL2AdapterConfig(base_path=str(tmp_path)).checksum == "none"

    def test_from_dict_parses_checksum(self, tmp_path: Path) -> None:
        cfg = FSL2AdapterConfig.from_dict(
            {"base_path": str(tmp_path), "checksum": "crc32"}
        )
        assert cfg.checksum == "crc32"

    @pytest.mark.parametrize("value", ["md5", "", "CRC32"])
    def test_unknown_algorithm_rejected(self, tmp_path: Path, value: str) -> None:
        with pytest.raises(ValueError, match="checksum"):
            FSL2AdapterConfig(base_path=str(tmp_path), checksum=value)


class TestCrc32:
    def test_round_trip(self, crc_adapter, tmp_path: Path) -> None:
        adp, listener = crc_adapter
        payload = os.urandom(64)
        _store(adp, payload)

        assert _only_data_file(tmp_path).stat().st_size == 64 + _TRAILER_SIZE
        assert _load(adp, 64) == (True, payload)
        assert listener.accessed == [_KEY]

    @pytest.mark.parametrize(
        "offset",
        [0, 63, 64, 64 + _TRAILER_SIZE - 1],
        ids=["payload_first", "payload_last", "trailer_magic", "trailer_crc"],
    )
    def test_single_bit_flip_is_miss_and_file_untouched(
        self, crc_adapter, tmp_path: Path, offset: int
    ) -> None:
        adp, listener = crc_adapter
        _store(adp, os.urandom(64))
        path = _only_data_file(tmp_path)
        _flip_byte(path, offset)
        corrupted = path.read_bytes()

        hit, _ = _load(adp, 64)

        assert not hit
        assert listener.accessed == []
        assert listener.deleted == []
        assert path.read_bytes() == corrupted

    def test_failed_key_is_rewritten_by_next_store(
        self, crc_adapter, tmp_path: Path
    ) -> None:
        adp, listener = crc_adapter
        _store(adp, b"a" * 64)
        _flip_byte(_only_data_file(tmp_path), 0)
        assert not _load(adp, 64)[0]

        _store(adp, b"b" * 64)

        assert _load(adp, 64) == (True, b"b" * 64)
        # The rewrite replaces the old file's bytes rather than adding to them.
        assert adp.get_usage().total_bytes_used == 64 + _TRAILER_SIZE
        assert listener.deleted == [_KEY]

    def test_replacement_during_verification_is_kept(
        self, crc_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A writer replacing the file while its old bytes are being verified
        must not have its newer, valid file removed or modified."""
        adp, _ = crc_adapter
        _store(adp, b"a" * 64)
        path = _only_data_file(tmp_path)
        valid = path.read_bytes()
        _flip_byte(path, 0)

        real_crc32 = zlib.crc32

        def replace_then_crc32(data, value=0):
            # The corrupt bytes are already read; a concurrent writer wins now.
            staged = path.with_name(path.name + ".writer")
            staged.write_bytes(valid)
            os.replace(staged, path)
            monkeypatch.setattr(zlib, "crc32", real_crc32)
            return real_crc32(data, value)

        monkeypatch.setattr(zlib, "crc32", replace_then_crc32)

        assert not _load(adp, 64)[0]
        assert path.read_bytes() == valid
        assert _load(adp, 64) == (True, b"a" * 64)

    def test_usage_returns_to_zero_after_delete(self, crc_adapter) -> None:
        adp, listener = crc_adapter
        _store(adp, b"x" * 100)
        assert listener.stored == [(_KEY, 100 + _TRAILER_SIZE)]

        adp.delete([_KEY])

        assert adp.get_usage().total_bytes_used == 0


class TestFormatCompatibility:
    def test_none_keeps_raw_format(self, tmp_path: Path) -> None:
        adp, _ = _make_adapter(tmp_path, checksum="none")
        try:
            _store(adp, b"r" * 32)
            assert _only_data_file(tmp_path).stat().st_size == 32
        finally:
            adp.close()

    def test_crc32_rewrites_files_without_trailer(self, tmp_path: Path) -> None:
        old, _ = _make_adapter(tmp_path, checksum="none")
        try:
            _store(old, b"o" * 32)
        finally:
            old.close()

        adp, listener = _make_adapter(tmp_path)
        try:
            assert not _load(adp, 32)[0]
            assert listener.deleted == []
            assert _only_data_file(tmp_path).stat().st_size == 32

            _store(adp, b"o" * 32)

            assert _only_data_file(tmp_path).stat().st_size == 32 + _TRAILER_SIZE
            assert _load(adp, 32) == (True, b"o" * 32)
        finally:
            adp.close()

    def test_none_reads_crc32_files(self, tmp_path: Path) -> None:
        new, _ = _make_adapter(tmp_path)
        try:
            _store(new, b"n" * 32)
        finally:
            new.close()

        adp, _ = _make_adapter(tmp_path, checksum="none")
        try:
            assert _load(adp, 32) == (True, b"n" * 32)
        finally:
            adp.close()


class TestODirectLayout:
    """O_DIRECT writes pad the trailer to one block.

    tmpfs rejects O_DIRECT, so the flag is removed to exercise the aligned
    code path and on-disk layout without it.
    """

    @pytest.fixture
    def block_size(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
        monkeypatch.delattr(os, "O_DIRECT", raising=False)
        return os.statvfs(tmp_path).f_bsize

    @pytest.mark.parametrize("checksum", ["none", "crc32"])
    def test_stale_larger_temp_file_is_truncated(
        self, tmp_path: Path, block_size: int, checksum: str
    ) -> None:
        payload = os.urandom(block_size)
        expected = block_size * (2 if checksum == "crc32" else 1)
        adp, listener = _make_adapter(tmp_path, checksum=checksum, use_odirect=True)
        try:
            _store(adp, payload)
            final = _only_data_file(tmp_path)
            adp.delete([_KEY])
            # Leftover from a crashed writer, longer than the new file.
            final.with_suffix(".tmp").write_bytes(b"\xff" * (4 * block_size))
            listener.stored.clear()

            _store(adp, payload)

            assert final.stat().st_size == expected
            assert listener.stored == [(_KEY, expected)]
            assert _load(adp, len(payload)) == (True, payload)
        finally:
            adp.close()

    def test_trailer_padded_to_block_and_verified(
        self, tmp_path: Path, block_size: int
    ) -> None:
        adp, _ = _make_adapter(tmp_path, use_odirect=True)
        try:
            payload = os.urandom(2 * block_size)
            _store(adp, payload)

            assert _only_data_file(tmp_path).stat().st_size == 3 * block_size
            assert _load(adp, len(payload)) == (True, payload)

            _flip_byte(_only_data_file(tmp_path), 0)
            assert not _load(adp, len(payload))[0]
        finally:
            adp.close()

    def test_odirect_reads_buffered_written_file(
        self, tmp_path: Path, block_size: int
    ) -> None:
        payload = os.urandom(block_size)
        writer, _ = _make_adapter(tmp_path)
        try:
            _store(writer, payload)
        finally:
            writer.close()

        reader, _ = _make_adapter(tmp_path, use_odirect=True)
        try:
            assert _load(reader, len(payload)) == (True, payload)
        finally:
            reader.close()
