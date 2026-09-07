# SPDX-License-Identifier: Apache-2.0
"""Deterministic concurrency and cancellation checks for raw L1 downloads."""

# Standard
from types import SimpleNamespace
from typing import Any
import asyncio
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey, L1BackendType, L1ObjectSnapshot
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.multiprocess.cache_control.errors import NotFound
from lmcache.v1.multiprocess.cache_control.object_service import ObjectService
from lmcache.v1.multiprocess.config import HTTPFrontendConfig


def _key() -> EncodedObjectKey:
    return EncodedObjectKey(chunk_hash_hex="01", model_name="test", kv_rank=0)


def _snapshot() -> L1ObjectSnapshot:
    return L1ObjectSnapshot(
        data=b"kv",
        size_bytes=2,
        backend=L1BackendType.DRAM,
        memory_format="undefined",
        shapes=((2,),),
        dtypes=("torch.uint8",),
    )


def _service(copy: Any) -> ObjectService:
    return ObjectService(
        SimpleNamespace(storage_manager=SimpleNamespace(snapshot_l1_object=copy)),
        HTTPFrontendConfig(
            enable_l1_cache_download=True, l1_cache_download_max_concurrency=1
        ),
    )


@pytest.mark.asyncio
async def test_concurrent_copies_are_limited() -> None:
    loop = asyncio.get_running_loop()
    copying = asyncio.Event()
    finish_copy = threading.Event()
    copies = 0

    def copy(*args: Any, **kwargs: Any) -> tuple[L1Error, L1ObjectSnapshot]:
        nonlocal copies
        copies += 1
        if copies == 1:
            loop.call_soon_threadsafe(copying.set)
            assert finish_copy.wait(5), "test failed to unblock copy"
        return L1Error.SUCCESS, _snapshot()

    service = _service(copy)
    first = asyncio.create_task(service.download_object(_key()))
    await asyncio.wait_for(copying.wait(), 5)
    second = asyncio.create_task(service.download_object(_key()))
    try:
        await asyncio.sleep(0)
        assert copies == 1
        assert not second.done()
    finally:
        finish_copy.set()
    first_snapshot, second_snapshot = await asyncio.gather(first, second)
    assert copies == 2
    assert first_snapshot.data == second_snapshot.data == b"kv"


@pytest.mark.asyncio
async def test_cancelled_copy_keeps_slot_until_worker_exits() -> None:
    loop = asyncio.get_running_loop()
    copying = asyncio.Event()
    attempting = asyncio.Event()
    finish_copy = threading.Event()
    copies = 0

    def copy(*args: Any, **kwargs: Any) -> tuple[L1Error, L1ObjectSnapshot]:
        nonlocal copies
        copies += 1
        if copies == 1:
            loop.call_soon_threadsafe(copying.set)
            assert finish_copy.wait(5), "test failed to unblock copy"
        return L1Error.SUCCESS, _snapshot()

    service = _service(copy)

    async def download() -> None:
        attempting.set()
        await service.download_object(_key())

    first = asyncio.create_task(download())
    try:
        await asyncio.wait_for(copying.wait(), 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        attempting.clear()
        second = asyncio.create_task(download())
        await asyncio.wait_for(attempting.wait(), 5)
        assert copies == 1
        assert not second.done()
    finally:
        finish_copy.set()
    await asyncio.wait_for(second, 5)
    assert copies == 2


@pytest.mark.asyncio
async def test_copy_error_releases_slot() -> None:
    copies = 0

    def copy(*args: Any, **kwargs: Any) -> tuple[L1Error, L1ObjectSnapshot | None]:
        nonlocal copies
        copies += 1
        if copies == 1:
            return L1Error.KEY_NOT_EXIST, None
        return L1Error.SUCCESS, _snapshot()

    service = _service(copy)
    with pytest.raises(NotFound):
        await service.download_object(_key())

    async def retry() -> None:
        snapshot = await service.download_object(_key())
        assert snapshot.data == b"kv"

    await asyncio.wait_for(retry(), 5)
