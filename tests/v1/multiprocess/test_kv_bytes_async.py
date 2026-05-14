# SPDX-License-Identifier: Apache-2.0
"""HTTP-level responsiveness tests for bytes-level KV cache APIs."""

# Standard
import asyncio
import threading
import time

# Third Party
from fastapi import FastAPI
import httpx
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.http_apis.kv_api import router as kv_router
from lmcache.v1.multiprocess.kv_bytes import RetrieveBytesResult

CHUNK_SIZE = 4


class _BlockingRetrieveEngine:
    """Engine fake whose public retrieve API blocks until released."""

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


async def _wait_for_blocking_retrieve(engine: _BlockingRetrieveEngine) -> None:
    """Wait until the test engine has entered its blocking retrieve call."""
    if engine.entered.is_set():
        return
    assert await asyncio.wait_for(
        asyncio.to_thread(engine.entered.wait, 0.2),
        timeout=0.3,
    )


@pytest.mark.parametrize("path", ["/api/kv/retrieve", "/api/kv/lookup"])
def test_http_kv_read_does_not_block_unrelated_http_requests(path: str) -> None:
    """A slow KV read request does not block another HTTP request."""

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
                blocked_request = asyncio.create_task(
                    client.post(
                        path,
                        json={
                            "model_name": "m",
                            "tokens": list(range(CHUNK_SIZE)),
                        },
                    )
                )

                await _wait_for_blocking_retrieve(engine)
                assert not blocked_request.done()

                started = time.monotonic()
                ping_response = await client.get("/ping")
                assert time.monotonic() - started < 0.25
                assert ping_response.status_code == 200

                engine.release.set()
                kv_response = await asyncio.wait_for(blocked_request, timeout=1.0)
                assert kv_response.status_code == 200
            finally:
                engine.release.set()

    asyncio.run(run())
