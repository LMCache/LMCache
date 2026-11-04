# SPDX-License-Identifier: Apache-2.0
"""Diagnostics of the legacy CUDA IPC export path.

The export itself needs a CUDA device, but the failure handling is pure
Python: storage stubs exercise it on CPU-only runners.
"""

# Standard
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.cuda.ipc_wrapper import _share_storage_for_ipc

DEVICE = torch.device("cuda", 0)


class _UnexportableStorage:
    """Storage whose allocation has no legacy CUDA IPC handle."""

    def __init__(self, nbytes: int) -> None:
        self._nbytes = nbytes

    def nbytes(self) -> int:
        return self._nbytes

    def _share_cuda_(self) -> Any:
        raise RuntimeError("CUDA error: out of memory")


class _ExportableStorage:
    """Storage that exports a handle."""

    def __init__(self, handle: tuple[Any, ...], nbytes: int) -> None:
        self._handle = handle
        self._nbytes = nbytes

    def nbytes(self) -> int:
        return self._nbytes

    def _share_cuda_(self) -> Any:
        return self._handle


def test_export_failure_names_the_size_and_the_workaround():
    """`cudaErrorMemoryAllocation` must not read as plain memory pressure."""
    storage = cast(torch.UntypedStorage, _UnexportableStorage(4 * 2**30))

    with pytest.raises(RuntimeError) as err:
        _share_storage_for_ipc(storage, DEVICE)

    message = str(err.value)
    assert "4.00 GiB" in message
    assert "cuda:0" in message
    assert "not necessarily memory pressure" in message
    assert "cudaErrorMemoryAllocation" in message
    assert "engine_driven" in message
    assert isinstance(err.value.__cause__, RuntimeError)


def test_successful_export_is_returned_unchanged():
    """The wrapper keeps using the handle tuple torch returns."""
    handle = (0, 0, 4096, 0, 1)
    storage = cast(torch.UntypedStorage, _ExportableStorage(handle, 4096))

    assert _share_storage_for_ipc(storage, DEVICE) == handle
