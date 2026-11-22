# SPDX-License-Identifier: Apache-2.0
"""Exercise adapter shutdown with a read blocked in the filesystem."""

# Standard
from pathlib import Path
from threading import Event
from typing import BinaryIO, cast
import os
import time

# Third Party
import pytest

# First Party
from lmcache.v1.common.errors import LMCacheTimeoutError
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import encode_fs_filename
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj


class _Destination:
    def __init__(self) -> None:
        self.byte_array = memoryview(bytearray(4))


def test_close_reports_stalled_fifo_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close times out while the worker owns the read buffer, then succeeds."""
    key = ObjectKey(chunk_hash=b"fifo", model_name="close", kv_rank=0)
    path = tmp_path / encode_fs_filename(key)
    os.mkfifo(path)
    opening = Event()

    def track_open(file: Path, mode: str, **kwargs: object) -> BinaryIO:
        assert mode == "rb"
        if file == path:
            opening.set()
        return file.open("rb")

    monkeypatch.setattr("aiofiles.threadpool.sync_open", track_open)
    monkeypatch.setattr(
        "lmcache.v1.distributed.l2_adapters.fs_l2_adapter._CLOSE_TIMEOUT_SECONDS",
        0.25,
    )
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    adapter.submit_load_task([key], [cast(MemoryObj, _Destination())])
    try:
        assert opening.wait(5)
        start = time.monotonic()
        with pytest.raises(LMCacheTimeoutError, match="Filesystem workers"):
            adapter.close()
        assert time.monotonic() - start < 2
    finally:
        writer = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
        adapter.close()
