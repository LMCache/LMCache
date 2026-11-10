# SPDX-License-Identifier: Apache-2.0
"""Physical DAX release must wait for confirmed unmapping."""

# Standard
from pathlib import Path
import ctypes
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.l2_adapters.dax.core import DaxCore, DaxPutFromPtrResult
from lmcache.v1.memory_management import MemoryFormat


def test_close_retries_exported_mapping_and_refuses_new_work(tmp_path: Path) -> None:
    """An exported public view must prevent successful detach until released."""
    path = tmp_path / "dax"
    path.write_bytes(bytes(4096))
    core = DaxCore[str](device_path=str(path), max_dax_size_bytes=4096, slot_bytes=64)
    mapping, fd = core.mmap_obj, core.fd
    assert core.arena_view is not None and mapping is not None and fd is not None
    exported = core.arena_view[:1]
    try:
        with pytest.raises(BufferError):
            core.close()
        assert core.mmap_obj is mapping and not mapping.closed
        assert core.fd == fd and os.fstat(fd).st_size == 4096
        assert core.report_status()["closing"]
        assert not core.report_status()["is_healthy"]
        with pytest.raises(RuntimeError, match="closing"):
            core.remap(4096)
        data = ctypes.create_string_buffer(b"payload")
        assert (
            core.put_reserved_from_ptr(
                "late",
                ctypes.addressof(data),
                8,
                torch.Size([8]),
                torch.uint8,
                MemoryFormat.KV_2LTD,
                None,
            )
            is DaxPutFromPtrResult.CLOSED
        )
    finally:
        exported.release()
        core.close()
    assert mapping.closed and core.mmap_obj is None and core.fd is None
    core.close()


def test_close_error_does_not_retry_a_reused_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed close syscall must not let a retry close an unrelated descriptor."""
    path = tmp_path / "dax"
    path.write_bytes(bytes(4096))
    core = DaxCore[str](device_path=str(path), max_dax_size_bytes=4096, slot_bytes=64)
    fd = core.fd
    assert fd is not None
    real_close = os.close

    def uncertain_close(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == fd:
            raise OSError("close acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(os, "close", uncertain_close)
        with pytest.raises(OSError, match="acknowledgement lost"):
            core.close()
    assert core.mmap_obj is None and core.fd is None
    replacement = os.open(path, os.O_RDONLY)
    if replacement != fd:
        os.dup2(replacement, fd)
        os.close(replacement)
    try:
        core.close()
        assert os.fstat(fd).st_size == 4096
    finally:
        os.close(fd)
