# SPDX-License-Identifier: Apache-2.0
"""Async behavior tests for bytes-level KV store/retrieve helpers."""

# Standard
from collections.abc import AsyncIterator
from typing import Any, cast
import asyncio
import threading
import time

# Third Party
from fastapi import FastAPI
import httpx
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.http_apis.kv_api import router as kv_router
from lmcache.v1.multiprocess.kv_bytes import (
    RetrieveBytesResult,
    store_kv_bytes_by_tokens,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 4
LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size((2, 1, CHUNK_SIZE, 4))],
    dtypes=[torch.float32],
)


class _BlockingStoreStorageManager:
    """Storage manager fake whose reserve call blocks long enough to detect."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def reserve_write(
        self,
        keys: list[Any],
        layout_desc: MemoryLayoutDesc,
        mode: str,
    ) -> dict[Any, Any]:
        """Block in the synchronous write-reservation call."""
        self.entered.set()
        self.release.wait(0.3)
        return {}

    def finish_write(self, keys: list[Any]) -> None:
        """Finish is a no-op because this fake never reserves objects."""


class _BlockingRetrieveEngine:
    """Engine fake whose retrieve call blocks long enough to detect."""

    chunk_size = CHUNK_SIZE

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def retrieve_kv_bytes_by_tokens(
        self,
        model_name: str,
        tokens: list[int],
        *,
        cache_salt: str = "",
    ) -> RetrieveBytesResult:
        """Block in retrieve and then return an empty hit result."""
        self.entered.set()
        self.release.wait(0.3)
        return RetrieveBytesResult(
            total_tokens=CHUNK_SIZE,
            total_chunks=1,
            hit_tokens=0,
            hit_chunks=0,
            world_size=1,
            per_shard_shape=(2, 1, CHUNK_SIZE, 4),
            dtype=torch.float32,
            shard_iter_factory=lambda: iter(()),
            close_callback=lambda: None,
        )


def _resolve_model(model_name: str) -> tuple[MemoryLayoutDesc, int]:
    """Return the fixed test layout for any model name."""
    return LAYOUT, 1


async def _single_chunk(payload: bytes) -> AsyncIterator[bytes]:
    """Yield one store chunk."""
    yield payload


def _chunk_payload() -> bytes:
    """Return one chunk of canonical KV_2LTD tensor bytes."""
    tensor = torch.zeros((2, 1, CHUNK_SIZE, 4), dtype=torch.float32)
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


async def _assert_loop_stays_responsive() -> None:
    """Assert a short sleep is not delayed by a synchronous blocking call."""
    start = time.monotonic()
    await asyncio.sleep(0.05)
    assert time.monotonic() - start < 0.25


async def _assert_blocking_call_entered(event: threading.Event) -> None:
    """Wait briefly for the worker-thread fake to enter its blocking section."""
    if event.is_set():
        return
    assert await asyncio.wait_for(
        asyncio.to_thread(event.wait, 0.2),
        timeout=0.3,
    )


def test_store_does_not_block_event_loop_while_reserving() -> None:
    """Store offloads synchronous storage writes from the asyncio loop."""

    async def run() -> None:
        storage_manager = _BlockingStoreStorageManager()
        task = asyncio.create_task(
            store_kv_bytes_by_tokens(
                model_name="m",
                tokens=list(range(CHUNK_SIZE)),
                chunks=_single_chunk(_chunk_payload()),
                full_shape=(2, 1, CHUNK_SIZE, 4),
                dtype=torch.float32,
                cache_salt="",
                chunk_size=CHUNK_SIZE,
                token_hasher=TokenHasher(chunk_size=CHUNK_SIZE),
                storage_manager=cast(StorageManager, storage_manager),
                resolve_model=_resolve_model,
            )
        )

        await _assert_loop_stays_responsive()
        await _assert_blocking_call_entered(storage_manager.entered)
        assert not task.done()
        storage_manager.release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result.stored_chunks == 0

    asyncio.run(run())


def test_http_retrieve_does_not_block_event_loop_while_waiting() -> None:
    """HTTP retrieve offloads synchronous engine waiting from the event loop."""

    async def run() -> None:
        engine = _BlockingRetrieveEngine()
        app = FastAPI()
        app.state.engine = engine
        app.include_router(kv_router)

        @app.get("/ping")
        async def ping() -> dict[str, bool]:
            return {"ok": True}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            try:
                task = asyncio.create_task(
                    client.post(
                        "/api/kv/retrieve",
                        json={
                            "model_name": "m",
                            "tokens": list(range(CHUNK_SIZE)),
                        },
                    )
                )

                await _assert_loop_stays_responsive()
                await _assert_blocking_call_entered(engine.entered)
                assert not task.done()

                ping_started = time.monotonic()
                ping_response = await client.get("/ping")
                assert time.monotonic() - ping_started < 0.25
                assert ping_response.status_code == 200

                engine.release.set()
                retrieve_response = await asyncio.wait_for(task, timeout=1.0)
                assert retrieve_response.status_code == 200
            finally:
                engine.release.set()

    asyncio.run(run())
